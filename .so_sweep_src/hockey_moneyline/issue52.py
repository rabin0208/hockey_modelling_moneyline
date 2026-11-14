"""Issue #52 frozen prospective artifact + pregame scoring proof.

Usage:
    python -m hockey_moneyline.issue52

The command uses only the existing frozen Cycle-2 MoneyPuck cache. It does not
refresh data and does not select or inspect 2026-27 outcomes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import (
    AdvancedTeamGame,
    inspect_advanced_stats_csv_for_cycle2_development,
)
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.issue45 import (
    _canonical_frozen_rows,
    _eligible,
    _pick,
    _vector_hash,
)
from hockey_moneyline.prospective_artifact import (
    ARTIFACT_VERSION,
    COMMITTED_FROZEN_ARTIFACT_RELATIVE_PATH,
    EXPECTED_FITTED_ARTIFACT_SHA256,
    EXPECTED_FROZEN_TRAINING_ROW_COUNT,
    EXPECTED_TRAINING_ROW_IDENTITY_SHA256,
    FEATURE_SCHEMA_VERSION,
    FROZEN_CANDIDATE_FAMILY,
    FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT,
    FROZEN_FEATURE_COLUMNS,
    FrozenFittedArtifact,
    assert_frozen_prospective_contract,
    compare_refit_to_committed_artifact,
    committed_frozen_artifact_path,
    fit_frozen_prospective_artifact,
    load_committed_frozen_artifact,
    write_frozen_artifact,
)
from hockey_moneyline.prospective_scoring import (
    FORBIDDEN_MARKET_FIELDS,
    FORBIDDEN_TARGET_OUTCOME_FIELDS,
    score_historical_replay_prediction,
)
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    RESERVED_FUTURE_SEASON,
    ScheduleHistoryGame,
    schedule_history_from_moneypuck_csv,
)
from hockey_moneyline.schedule_load_ablation import (
    KNOWN_CYCLE2_MONEYPUCK_SHA256,
    SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS,
    require_existing_cycle2_cache,
    sha256_path,
)
from hockey_moneyline.schema import NormalizedGame

DEFAULT_OUTPUT = Path("data/derived/issue52_frozen_prospective_scoring_report.json")
DEFAULT_ARTIFACT_OUTPUT = Path(
    "data/derived/frozen_prospective_fitted_artifact.json"
)
READY = "READY FOR PROSPECTIVE 2026-27 PREGAME SCORING"
NOT_READY = "NOT READY — ONE MATERIAL PROSPECTIVE-SCORING DEPENDENCY"


def _dry_run_timestamps(game_date) -> tuple[datetime, datetime]:
    scheduled = datetime.combine(game_date, time(23, 0), tzinfo=timezone.utc)
    observed = datetime.combine(game_date, time(12, 0), tzinfo=timezone.utc)
    return scheduled, observed


def build_issue52_prospective_report(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    *,
    historical_cache_sha256: str,
    created_at_utc: datetime | None = None,
) -> tuple[dict[str, Any], FrozenFittedArtifact]:
    assert_frozen_prospective_contract()
    if not games:
        raise ValueError("Issue #52 proof requires historical modeling games")
    if any(game.season >= RESERVED_FUTURE_SEASON for game in games):
        raise ValueError("Issue #52 proof cannot load 2026-27 outcomes")
    seasons = sorted({game.season for game in games})
    if tuple(seasons) != tuple(CYCLE2_DEVELOPMENT_SEASONS):
        raise ValueError(
            "Issue #52 proof requires exactly development seasons "
            "2021-22 through 2025-26"
        )

    artifact = load_committed_frozen_artifact()
    refit = fit_frozen_prospective_artifact(
        games,
        records,
        schedule_history,
        historical_cache_sha256=historical_cache_sha256,
        created_at_utc=created_at_utc,
    )
    blockers: list[str] = []
    blockers.extend(compare_refit_to_committed_artifact(artifact, refit))

    canonical_rows = _canonical_frozen_rows(games, records, schedule_history)
    eligible_rows = [row for row in canonical_rows if _eligible(row)]
    by_game = {game.game_id: game for game in games}
    date_counts: dict[str, int] = {}
    for game in games:
        key = game.game_date.isoformat()
        date_counts[key] = date_counts.get(key, 0) + 1

    target_ids = set(by_game)
    labeled_history = [
        item for item in schedule_history if item.game_id in target_ids
    ]
    from hockey_moneyline.schedule_load import build_prior_schedule_load_features

    schedule_without_history_only = build_prior_schedule_load_features(
        labeled_history,
        games,
    )
    schedule_without_by_id = {
        str(row["game_id"]): row for row in schedule_without_history_only
    }

    def tied_history_changes(row: Mapping[str, Any]) -> bool:
        reduced = schedule_without_by_id[str(row["game_id"])]
        return any(
            float(row[column]) != float(reduced[column])
            for column in SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
        )

    case_specs = (
        ("ordinary_prior_history", 20212022, lambda row: True),
        (
            "same_day_isolation",
            20222023,
            lambda row: date_counts[str(row["game_date"])] > 1,
        ),
        (
            "season_boundary",
            20222023,
            lambda row: (
                float(row["home_season_games"]) == 0.0
                or float(row["away_season_games"]) == 0.0
            ),
        ),
        (
            "prior_xgoal_availability",
            20232024,
            lambda row: (
                row.get("home_prior_xgoal_share") is not None
                and row.get("away_prior_xgoal_share") is not None
            ),
        ),
        (
            "back_to_back_and_seven_day_load",
            20242025,
            lambda row: (
                float(row["home_back_to_back"]) == 1.0
                or float(row["away_back_to_back"]) == 1.0
            )
            and (
                float(row["home_games_prior_7d"]) >= 2.0
                or float(row["away_games_prior_7d"]) >= 2.0
            ),
        ),
        ("tied_unlabeled_prior_history", 20252026, tied_history_changes),
    )

    cases: list[dict[str, Any]] = []
    used: set[str] = set()
    for name, preferred_season, predicate in case_specs:
        selected = _pick(
            eligible_rows,
            preferred_season=preferred_season,
            used=used,
            predicate=predicate,
        )
        if selected is None:
            blockers.append(f"historical dry-run case unavailable: {name}")
            continue
        game_id = str(selected["game_id"])
        used.add(game_id)
        target_game = by_game[game_id]
        records_without_target = {
            key: value for key, value in records.items() if key[0] != game_id
        }
        scheduled, observed = _dry_run_timestamps(target_game.game_date)
        snapshot, record = score_historical_replay_prediction(
            target_game,
            games,
            records_without_target,
            schedule_history,
            artifact,
            scheduled_start_time_utc=scheduled,
            prediction_timestamp_utc=observed,
            source_versions={"historical_replay_dry_run": True},
        )
        expected = tuple(
            float(selected[column]) for column in FROZEN_FEATURE_COLUMNS
        )
        equal = (
            snapshot.feature_values == expected
            and record.feature_values == snapshot.feature_values
        )
        if not equal:
            differing = [
                column
                for column, actual, canonical in zip(
                    FROZEN_FEATURE_COLUMNS,
                    snapshot.feature_values,
                    expected,
                    strict=True,
                )
                if actual != canonical
            ]
            blockers.append(
                f"{name} target {game_id} snapshot/scorer mismatch: columns={differing}"
            )
        leaked = [
            field
            for field in (*FORBIDDEN_MARKET_FIELDS, *FORBIDDEN_TARGET_OUTCOME_FIELDS)
            if field in record.to_dict()
        ]
        if leaked:
            blockers.append(f"{name} prediction record leaked fields: {leaked}")
        cases.append(
            {
                "case": name,
                "game_id": game_id,
                "game_date": str(selected["game_date"]),
                "season": int(selected["season"]),
                "exact_python_float_equality": equal,
                "scorer_consumed_issue45_snapshot_unchanged": (
                    record.feature_values == snapshot.feature_values
                ),
                "prospective_vector_sha256": _vector_hash(snapshot.feature_values),
                "canonical_vector_sha256": _vector_hash(expected),
                "model_home_win_probability": record.model_home_win_probability,
                "probability_kind": record.probability_kind,
                "prediction_before_scheduled_start": (
                    record.prediction_timestamp_utc < record.scheduled_start_time_utc
                ),
                "target_outcome_fields_in_prediction_record": False,
            }
        )

    covered_seasons = sorted({int(case["season"]) for case in cases})
    if covered_seasons != list(CYCLE2_DEVELOPMENT_SEASONS):
        blockers.append(
            "selected dry-run cases do not span every development season: "
            f"{covered_seasons}"
        )
    if artifact.fitted_artifact_sha256 != refit.fitted_artifact_sha256:
        blockers.append("refit on the same frozen development rows changed the artifact hash")
    if artifact.training_row_count != len(eligible_rows):
        blockers.append(
            "fitted training-row count disagrees with eligible frozen development rows"
        )
    if tuple(artifact.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        blockers.append("fitted artifact columns drifted from the frozen 25-column order")

    outcome = READY if not blockers else NOT_READY
    report: dict[str, Any] = {
        "issue": 52,
        "outcome": outcome,
        "blockers": blockers,
        "frozen_feature_family": FROZEN_CANDIDATE_FAMILY,
        "frozen_feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "frozen_feature_count": len(FROZEN_FEATURE_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "frozen_model": artifact.model_family,
        "code_freeze_commit": FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT,
        "fitted_artifact": {
            "artifact_version": ARTIFACT_VERSION,
            "committed_path": COMMITTED_FROZEN_ARTIFACT_RELATIVE_PATH,
            "committed_path_resolved": str(committed_frozen_artifact_path()),
            "scored_from_committed_artifact_without_refit": True,
            "fitted_artifact_sha256": artifact.fitted_artifact_sha256,
            "expected_fitted_artifact_sha256": EXPECTED_FITTED_ARTIFACT_SHA256,
            "matches_expected_fitted_artifact_sha256": (
                artifact.fitted_artifact_sha256 == EXPECTED_FITTED_ARTIFACT_SHA256
            ),
            "refit_artifact_sha256": refit.fitted_artifact_sha256,
            "reproducible_refit": (
                artifact.fitted_artifact_sha256 == refit.fitted_artifact_sha256
            ),
            "training_seasons": list(artifact.training_seasons),
            "training_window_start": artifact.training_window_start,
            "training_window_end": artifact.training_window_end,
            "training_row_count": artifact.training_row_count,
            "expected_training_row_count": EXPECTED_FROZEN_TRAINING_ROW_COUNT,
            "training_row_identity_sha256": artifact.training_row_identity_sha256,
            "expected_training_row_identity_sha256": EXPECTED_TRAINING_ROW_IDENTITY_SHA256,
            "matches_expected_training_row_identity_sha256": (
                artifact.training_row_identity_sha256
                == EXPECTED_TRAINING_ROW_IDENTITY_SHA256
            ),
            "historical_cache_sha256": artifact.historical_cache_sha256,
            "created_at_utc": artifact.created_at_utc,
            "fitted_before_first_2026_27_scored_prediction": True,
        },
        "historical_replay": {
            "development_seasons": list(CYCLE2_DEVELOPMENT_SEASONS),
            "2026_27_outcomes_accessed": False,
            "eligible_canonical_rows": len(eligible_rows),
            "case_count": len(cases),
            "cases": cases,
            "all_exact": bool(cases)
            and all(bool(case["exact_python_float_equality"]) for case in cases),
            "scorer_consumed_issue45_snapshot_without_changing_values": bool(cases)
            and all(
                bool(case["scorer_consumed_issue45_snapshot_unchanged"])
                for case in cases
            ),
        },
        "prediction_contract": {
            "requires_regular_season": True,
            "requires_pregame_timestamps": True,
            "model_probability_only": True,
            "market_fields_present": False,
            "champion_promoted": False,
            "production_deployed": False,
            "background_prediction_automation": False,
        },
        "remaining_dependency": (
            None
            if not blockers
            else {
                "owner": "03 — Prediction, Odds & Recommendation Engine",
                "dependency": blockers[0],
                "unblock_signal": (
                    "committed frozen artifact loads with the expected fingerprint, "
                    "a reproducibility refit matches it, and every historical dry-run "
                    "case consumes the Issue #45 25-column snapshot unchanged"
                ),
            }
        ),
        "notes": [
            "Durable fitted scaler/classifier parameters are committed at src/hockey_moneyline/artifacts/frozen_prospective_fitted_artifact.json.",
            "Prospective scoring loaded that committed artifact with integrity verification and did not refit to score.",
            "A reproducibility refit on eligible frozen development history through consumed 2025-26 was compared to the committed freeze.",
            "The prospective scorer consumes the merged Issue #45 unscored snapshot without changing values.",
            "No 2026-27 outcomes were loaded, inspected, or used for fitting, tuning, or artifact regeneration.",
            "No odds, market, ROI, staking, recommendation, Production, or schedule automation is included.",
        ],
    }
    return report, artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #52 frozen prospective artifact and historical scoring proof"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/moneypuck"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = require_existing_cycle2_cache(args.cache_dir)
        actual_sha = sha256_path(path)
        if actual_sha != KNOWN_CYCLE2_MONEYPUCK_SHA256:
            raise ValueError(
                "Issue #52 frozen MoneyPuck cache SHA-256 mismatch: "
                f"{actual_sha} != {KNOWN_CYCLE2_MONEYPUCK_SHA256}"
            )
        inspection = inspect_advanced_stats_csv_for_cycle2_development(
            path,
            CYCLE2_DEVELOPMENT_SEASONS,
        )
        schedule_history = schedule_history_from_moneypuck_csv(
            path,
            seasons=CYCLE2_DEVELOPMENT_SEASONS,
        )
        report, artifact = build_issue52_prospective_report(
            inspection.games,
            inspection.records,
            schedule_history,
            historical_cache_sha256=actual_sha,
        )
        report["source"] = {
            "cache_path": str(path),
            "cache_sha256": actual_sha,
            "refresh": False,
            "raw_provider_file_committed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_frozen_artifact(args.artifact_output, artifact)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        print(f"Wrote {args.artifact_output}")
        return 0 if report["outcome"] == READY else 1
    except (HockeyMoneylineError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
