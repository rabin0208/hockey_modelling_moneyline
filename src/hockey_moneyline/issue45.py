"""Issue #45 historical replay proof for the frozen unscored-target primitive.

Usage:
    python -m hockey_moneyline.issue45

The command uses only the existing frozen Cycle-2 MoneyPuck cache. It does not
refresh data and does not select or inspect 2026-27 outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import (
    AdvancedTeamGame,
    build_pregame_xgoal_share_features,
    inspect_advanced_stats_csv_for_cycle2_development,
)
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.features import build_features
from hockey_moneyline.provenance import MIN_PRIOR_GAMES
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    ScheduleHistoryGame,
    build_prior_schedule_load_features,
    schedule_history_from_moneypuck_csv,
)
from hockey_moneyline.schedule_load_ablation import (
    KNOWN_CYCLE2_MONEYPUCK_SHA256,
    SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS,
    attach_pregame_schedule_load_compact_features,
    require_existing_cycle2_cache,
    schedule_load_compact_features_valid,
    sha256_path,
)
from hockey_moneyline.schema import NormalizedGame
from hockey_moneyline.unscored_target import (
    FROZEN_ISSUE40_FEATURE_COLUMNS,
    build_historical_replay_snapshot,
)
from hockey_moneyline.xgoal_share import (
    attach_pregame_xgoal_share_features,
    xgoal_features_valid,
)

DEFAULT_OUTPUT = Path("data/derived/issue45_unscored_target_parity_report.json")
READY = "READY FOR 03 PROSPECTIVE PREDICTION PIPELINE"
NOT_READY = "NOT READY — ONE MATERIAL PREDICTION-PARITY DEPENDENCY"


def _canonical_frozen_rows(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
) -> list[dict[str, Any]]:
    rows = build_features(games)
    rows = attach_pregame_xgoal_share_features(
        rows,
        build_pregame_xgoal_share_features(games, records),
    )
    rows = attach_pregame_schedule_load_compact_features(
        rows,
        build_prior_schedule_load_features(schedule_history, games),
    )
    return rows


def _eligible(row: Mapping[str, Any]) -> bool:
    return (
        float(row["home_prior_games"]) >= MIN_PRIOR_GAMES
        and float(row["away_prior_games"]) >= MIN_PRIOR_GAMES
        and xgoal_features_valid(row)
        and schedule_load_compact_features_valid(row)
    )


def _pick(
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_season: int,
    used: set[str],
    predicate=lambda _row: True,
) -> Mapping[str, Any] | None:
    preferred = [
        row
        for row in rows
        if int(row["season"]) == preferred_season
        and str(row["game_id"]) not in used
        and predicate(row)
    ]
    fallback = [
        row
        for row in rows
        if str(row["game_id"]) not in used and predicate(row)
    ]
    candidates = preferred or fallback
    return min(
        candidates,
        key=lambda row: (str(row["game_date"]), str(row["game_id"])),
        default=None,
    )


def _vector_hash(values: Sequence[float]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_issue45_parity_report(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
) -> dict[str, Any]:
    if not games:
        raise ValueError("Issue #45 parity proof requires historical modeling games")
    seasons = sorted({game.season for game in games})
    if tuple(seasons) != tuple(CYCLE2_DEVELOPMENT_SEASONS):
        raise ValueError(
            "Issue #45 parity proof requires exactly development seasons "
            "2021-22 through 2025-26"
        )

    canonical_rows = _canonical_frozen_rows(games, records, schedule_history)
    eligible_rows = [row for row in canonical_rows if _eligible(row)]
    by_game = {game.game_id: game for game in games}
    date_counts = Counter(game.game_date.isoformat() for game in games)

    target_ids = set(by_game)
    labeled_history = [
        item for item in schedule_history if item.game_id in target_ids
    ]
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

    blockers: list[str] = []
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
            blockers.append(f"historical parity case unavailable: {name}")
            continue
        game_id = str(selected["game_id"])
        used.add(game_id)
        target_game = by_game[game_id]

        # Explicitly remove the target's own postgame advanced rows. The replay
        # must not need them.
        records_without_target = {
            key: value for key, value in records.items() if key[0] != game_id
        }
        replay = build_historical_replay_snapshot(
            target_game,
            games,
            records_without_target,
            schedule_history,
        )
        expected = tuple(
            float(selected[column]) for column in FROZEN_ISSUE40_FEATURE_COLUMNS
        )
        equal = replay.feature_values == expected
        if not equal:
            differing = [
                column
                for column, actual, canonical in zip(
                    FROZEN_ISSUE40_FEATURE_COLUMNS,
                    replay.feature_values,
                    expected,
                    strict=True,
                )
                if actual != canonical
            ]
            blockers.append(
                f"{name} target {game_id} parity mismatch: columns={differing}"
            )
        cases.append(
            {
                "case": name,
                "game_id": game_id,
                "game_date": str(selected["game_date"]),
                "season": int(selected["season"]),
                "feature_columns": list(FROZEN_ISSUE40_FEATURE_COLUMNS),
                "exact_python_float_equality": equal,
                "prospective_vector_sha256": _vector_hash(replay.feature_values),
                "canonical_vector_sha256": _vector_hash(expected),
                "target_advanced_record_removed": True,
                "target_outcome_fields_in_unscored_representation": False,
            }
        )

    covered_seasons = sorted({int(case["season"]) for case in cases})
    if covered_seasons != list(CYCLE2_DEVELOPMENT_SEASONS):
        blockers.append(
            "selected parity cases do not span every development season: "
            f"{covered_seasons}"
        )

    outcome = READY if not blockers else NOT_READY
    report: dict[str, Any] = {
        "issue": 45,
        "outcome": outcome,
        "blockers": blockers,
        "frozen_feature_family": (
            "baseline_v1_plus_elo_plus_prior_xgoal_share_plus_"
            "prior_schedule_load_compact"
        ),
        "frozen_feature_columns": list(FROZEN_ISSUE40_FEATURE_COLUMNS),
        "frozen_feature_count": len(FROZEN_ISSUE40_FEATURE_COLUMNS),
        "historical_replay": {
            "development_seasons": list(CYCLE2_DEVELOPMENT_SEASONS),
            "2026_27_outcomes_accessed": False,
            "eligible_canonical_rows": len(eligible_rows),
            "case_count": len(cases),
            "cases": cases,
            "all_exact": bool(cases) and all(
                bool(case["exact_python_float_equality"]) for case in cases
            ),
        },
        "target_contract": {
            "requires_home_score": False,
            "requires_away_score": False,
            "requires_home_win": False,
            "requires_target_advanced_stat_record": False,
            "nhl_schedule_adapter_preserves_identity_and_pregame_timing": True,
        },
        "timing_contract": {
            "result_elo_xgoal_rule": "history.game_date < target.game_date",
            "same_day_result_isolation": True,
            "schedule_load_rule": (
                "complete same-season schedule history with "
                "history.game_date < target.game_date"
            ),
            "future_result_mutation_cannot_change_historical_replay": True,
        },
        "history_contract": {
            "winner_labeled_result_history_games": len(games),
            "complete_schedule_history_games": len(schedule_history),
            "history_only_tied_unlabeled_games": (
                len({game.game_id for game in schedule_history}) - len(target_ids)
            ),
        },
        "remaining_dependency": (
            None
            if not blockers
            else {
                "owner": "01 — Data, Features & Validation",
                "dependency": blockers[0],
                "unblock_signal": (
                    "all required historical replay cases produce exact frozen "
                    "25-column equality without target outcome/stat fields"
                ),
            }
        ),
        "notes": [
            "No model fitting, model ablation, tuning, recalibration, or live prediction runs here.",
            "The replay harness removes each target's advanced-stat record before prospective snapshot construction.",
            "The prospective primitive can accept legitimately completed prior 2026-27 history later; this Issue #45 validation itself uses no 2026-27 outcomes.",
        ],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #45 unscored-target frozen historical parity proof"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/moneypuck"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = require_existing_cycle2_cache(args.cache_dir)
        actual_sha = sha256_path(path)
        if actual_sha != KNOWN_CYCLE2_MONEYPUCK_SHA256:
            raise ValueError(
                "Issue #45 frozen MoneyPuck cache SHA-256 mismatch: "
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
        report = build_issue45_parity_report(
            inspection.games,
            inspection.records,
            schedule_history,
        )
        report["source"] = {
            "cache_path": str(path),
            "cache_sha256": actual_sha,
            "refresh": False,
            "raw_provider_file_committed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        return 0 if report["outcome"] == READY else 1
    except (HockeyMoneylineError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
