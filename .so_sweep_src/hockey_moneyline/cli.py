"""Command-line entry points for the canonical Hockey Moneyline baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from hockey_moneyline.acquire import DEFAULT_CACHE_DIR, acquire_source_csv
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.advanced_stats import (
    assert_corsi_source_ready,
    inspect_advanced_stats_csv,
    inspect_advanced_stats_csv_for_authorized_heldout,
    inspect_advanced_stats_csv_for_cycle2_development,
)
from hockey_moneyline.evaluate import (
    VALIDATION_HISTORY_SEASONS,
    evaluate_forward_chaining_games,
    evaluate_corsi_share_ablation_games,
    evaluate_hgb_comparison_games,
    evaluate_linear_recency_ablation_games,
    evaluate_ordered_lag_ablation_games,
    evaluate_validation_ablation_games,
    evaluate_xgoal_share_ablation_games,
)
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    schedule_history_from_moneypuck_csv,
)
from hockey_moneyline.schedule_load_ablation import (
    CYCLE2_EXACT_SEASONS_CLI,
    DEFAULT_CYCLE2_SCHEDULE_LOAD_OUTPUT,
    KNOWN_CYCLE2_MONEYPUCK_SHA256,
    evaluate_schedule_load_compact_ablation_games,
    require_existing_cycle2_cache,
    sha256_path as cycle2_sha256_path,
)
from hockey_moneyline.features import build_features
from hockey_moneyline.frozen_heldout import (
    DEFAULT_FROZEN_HELDOUT_OUTPUT,
    EXACT_SEASONS_CLI,
    assert_frozen_heldout_request,
    evaluate_frozen_heldout_games,
    require_existing_frozen_cache,
    sha256_path,
)
from hockey_moneyline.normalize import NormalizeReport, games_from_moneypuck_csv
from hockey_moneyline.provenance import DEFAULT_SEASONS, FINAL_TEST_SEASON, provenance_record


def _parse_seasons(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_SEASONS
    seasons = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seasons:
        raise SystemExit("No seasons provided")
    return seasons


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {path}")


def _load_games(
    args: argparse.Namespace,
    *,
    default_seasons: tuple[int, ...] = DEFAULT_SEASONS,
) -> NormalizeReport:
    seasons = _parse_seasons(args.seasons) if args.seasons else default_seasons
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    return games_from_moneypuck_csv(path, seasons=seasons)


def cmd_acquire(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons)
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    report = games_from_moneypuck_csv(path, seasons=seasons)
    games = report.games
    summary = {
        "source": provenance_record(),
        "seasons": list(seasons),
        "cached_file": str(path),
        "n_normalized_games": len(games),
        "normalized_games_by_season": report.normalized_games_by_season,
        "skipped_tied_scores": report.skipped_tied_scores,
        "skipped_tied_scores_by_season": report.skipped_tied_scores_by_season,
        "notes": report.notes,
        "start": games[0].game_date.isoformat() if games else None,
        "end": games[-1].game_date.isoformat() if games else None,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        _write_json(Path(args.output), [game.to_dict() for game in games])
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    loaded = _load_games(args)
    report = evaluate_forward_chaining_games(
        loaded.games,
        min_prior_games=args.min_prior_games,
    )
    report["normalization"] = {
        "normalized_games_by_season": loaded.normalized_games_by_season,
        "skipped_tied_scores": loaded.skipped_tied_scores,
        "skipped_tied_scores_by_season": loaded.skipped_tied_scores_by_season,
    }
    report.setdefault("notes", []).extend(loaded.notes)
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/baseline_report.json")
    _write_json(output, report)
    if args.features_output:
        _write_json(Path(args.features_output), build_features(loaded.games))
    return 0


def cmd_ablate_elo(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else VALIDATION_HISTORY_SEASONS
    if FINAL_TEST_SEASON in seasons:
        raise SystemExit(
            "ERROR: Issue #13 validation-only ablation cannot include the 2025-26 held-out test"
        )
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    loaded = games_from_moneypuck_csv(path, seasons=seasons)
    report = evaluate_validation_ablation_games(
        loaded.games,
        min_prior_games=args.min_prior_games,
    )
    report["normalization"] = {
        "normalized_games_by_season": loaded.normalized_games_by_season,
        "skipped_tied_scores": loaded.skipped_tied_scores,
        "skipped_tied_scores_by_season": loaded.skipped_tied_scores_by_season,
    }
    report.setdefault("notes", []).extend(loaded.notes)
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/elo_ablation_report.json")
    _write_json(output, report)
    return 0



def cmd_ablate_ordered_lags(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else VALIDATION_HISTORY_SEASONS
    if FINAL_TEST_SEASON in seasons:
        raise SystemExit(
            "ERROR: Issue #15 validation-only ablation cannot include the 2025-26 held-out test"
        )
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    loaded = games_from_moneypuck_csv(path, seasons=seasons)
    report = evaluate_ordered_lag_ablation_games(
        loaded.games,
        min_prior_games=args.min_prior_games,
    )
    report["normalization"] = {
        "normalized_games_by_season": loaded.normalized_games_by_season,
        "skipped_tied_scores": loaded.skipped_tied_scores,
        "skipped_tied_scores_by_season": loaded.skipped_tied_scores_by_season,
    }
    report.setdefault("notes", []).extend(loaded.notes)
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/ordered_lag_ablation_report.json")
    _write_json(output, report)
    return 0


def cmd_ablate_linear_recency(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else VALIDATION_HISTORY_SEASONS
    if FINAL_TEST_SEASON in seasons:
        raise SystemExit(
            "ERROR: Issue #17 validation-only ablation cannot include the 2025-26 held-out test"
        )
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    loaded = games_from_moneypuck_csv(path, seasons=seasons)
    report = evaluate_linear_recency_ablation_games(
        loaded.games,
        min_prior_games=args.min_prior_games,
    )
    report["normalization"] = {
        "normalized_games_by_season": loaded.normalized_games_by_season,
        "skipped_tied_scores": loaded.skipped_tied_scores,
        "skipped_tied_scores_by_season": loaded.skipped_tied_scores_by_season,
    }
    report.setdefault("notes", []).extend(loaded.notes)
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/linear_recency_ablation_report.json")
    _write_json(output, report)
    return 0


def cmd_ablate_prior_xgoal_share(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else VALIDATION_HISTORY_SEASONS
    if FINAL_TEST_SEASON in seasons:
        raise SystemExit(
            "ERROR: Issue #21 validation-only ablation cannot include the 2025-26 held-out test"
        )
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    inspection = inspect_advanced_stats_csv(path, seasons=seasons)
    if inspection.report["outcome"] != "READY FOR 02 ABLATION":
        raise SystemExit(
            "ERROR: Issue #19 advanced-stat contract is not READY FOR 02 ABLATION: "
            f"{inspection.report.get('blockers')}"
        )
    report = evaluate_xgoal_share_ablation_games(
        inspection.games,
        inspection.records,
        min_prior_games=args.min_prior_games,
        assert_incumbent_reproduction=True,
    )
    report["issue19_contract"] = {
        "outcome": inspection.report["outcome"],
        "seasons_profiled": inspection.report["seasons_profiled"],
        "held_out_2025_26_profiled": inspection.report["held_out_2025_26_profiled"],
        "normalized_game_team_unmatched": inspection.report["join"][
            "normalized_game_team_unmatched"
        ],
        "home_away_pair_conflicts": inspection.report["join"]["home_away_pair_conflicts"],
        "raw_file_committed": False,
    }
    report["normalization"] = {
        "normalized_games_by_season": {
            season: count
            for season, count in inspection.report["join"]["normalized_games_by_season"].items()
        },
        "matched_team_game_rows_by_season": inspection.report["join"][
            "matched_team_game_rows_by_season"
        ],
    }
    report.setdefault("notes", []).append(
        "Issue #21 reused the validated Issue #19 inspect/join primitive; raw MoneyPuck CSV remains uncommitted."
    )
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/prior_xgoal_share_ablation_report.json")
    _write_json(output, report)
    return 0


def cmd_compare_histgradientboosting(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else VALIDATION_HISTORY_SEASONS
    if FINAL_TEST_SEASON in seasons:
        raise SystemExit(
            "ERROR: Issue #23 validation-only comparison cannot include the 2025-26 held-out test"
        )
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    inspection = inspect_advanced_stats_csv(path, seasons=seasons)
    if inspection.report["outcome"] != "READY FOR 02 ABLATION":
        raise SystemExit(
            "ERROR: Issue #19 advanced-stat contract is not READY FOR 02 ABLATION: "
            f"{inspection.report.get('blockers')}"
        )
    report = evaluate_hgb_comparison_games(
        inspection.games,
        inspection.records,
        min_prior_games=args.min_prior_games,
        assert_logistic_reproduction=True,
        assert_issue21_row_counts=True,
    )
    report["issue19_contract"] = {
        "outcome": inspection.report["outcome"],
        "seasons_profiled": inspection.report["seasons_profiled"],
        "held_out_2025_26_profiled": inspection.report["held_out_2025_26_profiled"],
        "normalized_game_team_unmatched": inspection.report["join"][
            "normalized_game_team_unmatched"
        ],
        "home_away_pair_conflicts": inspection.report["join"]["home_away_pair_conflicts"],
        "raw_file_committed": False,
    }
    report.setdefault("notes", []).append(
        "Issue #23 reused the Issue #19/#21 feature contract; raw MoneyPuck CSV remains uncommitted."
    )
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/histgradientboosting_comparison_report.json")
    _write_json(output, report)
    return 0


def cmd_ablate_prior_corsi_share(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else VALIDATION_HISTORY_SEASONS
    if FINAL_TEST_SEASON in seasons:
        raise SystemExit(
            "ERROR: Issue #25 validation-only ablation cannot include the 2025-26 held-out test"
        )
    path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    inspection = inspect_advanced_stats_csv(path, seasons=seasons)
    if inspection.report["outcome"] != "READY FOR 02 ABLATION":
        raise SystemExit(
            "ERROR: Issue #19 advanced-stat contract is not READY FOR 02 ABLATION: "
            f"{inspection.report.get('blockers')}"
        )
    corsi_ready = assert_corsi_source_ready(inspection)
    report = evaluate_corsi_share_ablation_games(
        inspection.games,
        inspection.records,
        min_prior_games=args.min_prior_games,
        assert_incumbent_reproduction=True,
        assert_issue21_row_counts=True,
    )
    report["issue19_contract"] = {
        "outcome": inspection.report["outcome"],
        "seasons_profiled": inspection.report["seasons_profiled"],
        "held_out_2025_26_profiled": inspection.report["held_out_2025_26_profiled"],
        "normalized_game_team_unmatched": inspection.report["join"][
            "normalized_game_team_unmatched"
        ],
        "home_away_pair_conflicts": inspection.report["join"]["home_away_pair_conflicts"],
        "raw_file_committed": False,
        "note": (
            "Issue #19 READY FOR 02 ABLATION is xGoal-oriented and is not treated "
            "as proof of Corsi readiness."
        ),
    }
    report["corsi_source_readiness"] = corsi_ready
    report["normalization"] = {
        "normalized_games_by_season": {
            season: count
            for season, count in inspection.report["join"]["normalized_games_by_season"].items()
        },
        "matched_team_game_rows_by_season": inspection.report["join"][
            "matched_team_game_rows_by_season"
        ],
    }
    report.setdefault("notes", []).append(
        "Issue #25 reused the validated Issue #19 inspect/join primitive and "
        "explicitly verified shotAttemptsFor/shotAttemptsAgainst completeness; "
        "raw MoneyPuck CSV remains uncommitted."
    )
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or "data/derived/prior_corsi_share_ablation_report.json")
    _write_json(output, report)
    return 0


def cmd_ablate_prior_schedule_load_compact(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons) if args.seasons else CYCLE2_DEVELOPMENT_SEASONS
    if seasons != CYCLE2_DEVELOPMENT_SEASONS:
        raise SystemExit(
            "ERROR: Issue #38 requires exactly "
            f"{CYCLE2_EXACT_SEASONS_CLI}; got {','.join(str(s) for s in seasons)}"
        )
    path = require_existing_cycle2_cache(Path(args.cache_dir))
    cache_sha256 = cycle2_sha256_path(path)
    inspection = inspect_advanced_stats_csv_for_cycle2_development(path, seasons)
    if inspection.report["outcome"] != "READY FOR 02 ABLATION":
        raise SystemExit(
            "ERROR: Issue #19 advanced-stat contract is not READY FOR 02 ABLATION: "
            f"{inspection.report.get('blockers')}"
        )
    schedule_history = schedule_history_from_moneypuck_csv(path, seasons=seasons)
    report = evaluate_schedule_load_compact_ablation_games(
        inspection.games,
        inspection.records,
        schedule_history,
        min_prior_games=args.min_prior_games,
    )
    report["command"] = (
        "python -m hockey_moneyline ablate-prior-schedule-load-compact "
        f"--seasons {CYCLE2_EXACT_SEASONS_CLI} "
        f"--output {args.output or DEFAULT_CYCLE2_SCHEDULE_LOAD_OUTPUT}"
    )
    report["cycle2_cache"] = {
        "path": str(path),
        "sha256": cache_sha256,
        "size_bytes": path.stat().st_size,
        "refresh": False,
        "raw_file_committed": False,
        "matches_known_issue32_34_identity": cache_sha256 == KNOWN_CYCLE2_MONEYPUCK_SHA256,
        "known_issue32_34_sha256": KNOWN_CYCLE2_MONEYPUCK_SHA256,
    }
    report["issue19_contract"] = {
        "outcome": inspection.report["outcome"],
        "seasons_profiled": inspection.report["seasons_profiled"],
        "consumed_2025_26_as_cycle2_development_evidence": True,
        "held_out_2025_26_profiled_flag_from_issue19_primitive": inspection.report[
            "held_out_2025_26_profiled"
        ],
        "normalized_game_team_unmatched": inspection.report["join"][
            "normalized_game_team_unmatched"
        ],
        "home_away_pair_conflicts": inspection.report["join"]["home_away_pair_conflicts"],
        "raw_file_committed": False,
        "note": (
            "Issue #38 used a Cycle-2 development wrapper over the Issue #19 "
            "parse/join/timing primitive. 2025-26 is consumed Cycle-2 development "
            "evidence here, not a fresh heldout."
        ),
    }
    report["normalization"] = {
        "normalized_games_by_season": {
            season: count
            for season, count in inspection.report["join"][
                "normalized_games_by_season"
            ].items()
        },
        "matched_team_game_rows_by_season": inspection.report["join"][
            "matched_team_game_rows_by_season"
        ],
    }
    report.setdefault("notes", []).append(
        "Issue #38 reused Issue #19 advanced-stat join/timing and Issue #34 "
        "complete schedule-history construction; raw MoneyPuck CSV remains uncommitted; "
        "no --refresh was used."
    )
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or DEFAULT_CYCLE2_SCHEDULE_LOAD_OUTPUT)
    _write_json(output, report)
    return 0


def cmd_evaluate_frozen_heldout(args: argparse.Namespace) -> int:
    seasons = _parse_seasons(args.seasons)
    try:
        assert_frozen_heldout_request(
            seasons,
            confirm_consume_2025_26=bool(args.confirm_consume_2025_26),
        )
    except HockeyMoneylineError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    path = require_existing_frozen_cache(Path(args.cache_dir))
    cache_sha256 = sha256_path(path)
    inspection = inspect_advanced_stats_csv_for_authorized_heldout(
        path,
        seasons,
        confirm_consume_2025_26=True,
    )
    if inspection.report["outcome"] != "READY FOR 02 ABLATION":
        raise SystemExit(
            "ERROR: Issue #19 advanced-stat contract is not READY FOR 02 ABLATION: "
            f"{inspection.report.get('blockers')}"
        )
    report = evaluate_frozen_heldout_games(
        inspection.games,
        inspection.records,
        confirm_consume_2025_26=True,
    )
    report["frozen_cache"] = {
        "path": str(path),
        "sha256": cache_sha256,
        "size_bytes": path.stat().st_size,
        "refresh": False,
        "raw_file_committed": False,
    }
    report["issue19_contract"] = {
        "outcome": inspection.report["outcome"],
        "seasons_profiled": inspection.report["seasons_profiled"],
        "held_out_2025_26_profiled": inspection.report["held_out_2025_26_profiled"],
        "normalized_game_team_unmatched": inspection.report["join"][
            "normalized_game_team_unmatched"
        ],
        "home_away_pair_conflicts": inspection.report["join"]["home_away_pair_conflicts"],
        "raw_file_committed": False,
    }
    report.setdefault("notes", []).append(
        "Issue #32 reused the validated Issue #19 inspect/join/timing primitive "
        "through an explicitly gated heldout pathway; raw MoneyPuck CSV remains uncommitted."
    )
    print(json.dumps(report, indent=2, default=str))
    output = Path(args.output or DEFAULT_FROZEN_HELDOUT_OUTPUT)
    _write_json(output, report)
    return 0


def cmd_provenance(_: argparse.Namespace) -> int:
    print(json.dumps(provenance_record(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hockey Moneyline canonical historical baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    acquire = sub.add_parser("acquire", help="Download and normalize the MoneyPuck source")
    acquire.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is canonical 2021-22 through 2025-26",
    )
    acquire.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    acquire.add_argument("--refresh", action="store_true")
    acquire.add_argument("--output", help="Optional path for normalized games JSON")
    acquire.set_defaults(func=cmd_acquire)

    evaluate = sub.add_parser("evaluate", help="Run canonical strict forward-chaining baseline evaluation")
    evaluate.add_argument(
        "--seasons",
        help="Comma-separated season ids; canonical evaluation requires 2021-22 through 2025-26",
    )
    evaluate.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    evaluate.add_argument("--refresh", action="store_true")
    evaluate.add_argument("--min-prior-games", type=int, default=5)
    evaluate.add_argument("--output", help="Report JSON path")
    evaluate.add_argument("--features-output", help="Optional feature-row JSON path")
    evaluate.set_defaults(func=cmd_evaluate)

    ablate = sub.add_parser(
        "ablate-elo",
        help="Issue #13 validation-only Elo feature ablation (folds A/B/C only)",
    )
    ablate.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is 2021-22 through 2024-25. 2025-26 is rejected.",
    )
    ablate.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ablate.add_argument("--refresh", action="store_true")
    ablate.add_argument("--min-prior-games", type=int, default=5)
    ablate.add_argument("--output", help="Report JSON path")
    ablate.set_defaults(func=cmd_ablate_elo)


    ordered = sub.add_parser(
        "ablate-ordered-lags",
        help="Issue #15 validation-only ordered recent-form lag ablation (folds A/B/C only)",
    )
    ordered.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is 2021-22 through 2024-25. 2025-26 is rejected.",
    )
    ordered.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ordered.add_argument("--refresh", action="store_true")
    ordered.add_argument("--min-prior-games", type=int, default=5)
    ordered.add_argument("--output", help="Report JSON path")
    ordered.set_defaults(func=cmd_ablate_ordered_lags)

    recency = sub.add_parser(
        "ablate-linear-recency",
        help="Issue #17 validation-only linear recency-weighted recent-form ablation (folds A/B/C only)",
    )
    recency.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is 2021-22 through 2024-25. 2025-26 is rejected.",
    )
    recency.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    recency.add_argument("--refresh", action="store_true")
    recency.add_argument("--min-prior-games", type=int, default=5)
    recency.add_argument("--output", help="Report JSON path")
    recency.set_defaults(func=cmd_ablate_linear_recency)

    xgoal = sub.add_parser(
        "ablate-prior-xgoal-share",
        help="Issue #21 validation-only prior xGoal share ablation (folds A/B/C only)",
    )
    xgoal.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is 2021-22 through 2024-25. 2025-26 is rejected.",
    )
    xgoal.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    xgoal.add_argument("--refresh", action="store_true")
    xgoal.add_argument("--min-prior-games", type=int, default=5)
    xgoal.add_argument("--output", help="Report JSON path")
    xgoal.set_defaults(func=cmd_ablate_prior_xgoal_share)

    hgb = sub.add_parser(
        "compare-histgradientboosting",
        help="Issue #23 validation-only logistic vs HistGradientBoosting comparison (folds A/B/C only)",
    )
    hgb.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is 2021-22 through 2024-25. 2025-26 is rejected.",
    )
    hgb.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    hgb.add_argument("--refresh", action="store_true")
    hgb.add_argument("--min-prior-games", type=int, default=5)
    hgb.add_argument("--output", help="Report JSON path")
    hgb.set_defaults(func=cmd_compare_histgradientboosting)

    corsi = sub.add_parser(
        "ablate-prior-corsi-share",
        help="Issue #25 validation-only prior Corsi share ablation (folds A/B/C only)",
    )
    corsi.add_argument(
        "--seasons",
        help="Comma-separated season ids; default is 2021-22 through 2024-25. 2025-26 is rejected.",
    )
    corsi.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    corsi.add_argument("--refresh", action="store_true")
    corsi.add_argument("--min-prior-games", type=int, default=5)
    corsi.add_argument("--output", help="Report JSON path")
    corsi.set_defaults(func=cmd_ablate_prior_corsi_share)

    schedule_load = sub.add_parser(
        "ablate-prior-schedule-load-compact",
        help=(
            "Issue #38 Cycle-2 prior schedule-load compact ablation "
            "(folds A/B/C/D; 2025-26 consumed development evidence; rejects 2026-27). "
            "Uses existing local MoneyPuck cache only; never refreshes."
        ),
    )
    schedule_load.add_argument(
        "--seasons",
        default=CYCLE2_EXACT_SEASONS_CLI,
        help=f"Must be exactly {CYCLE2_EXACT_SEASONS_CLI}",
    )
    schedule_load.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    schedule_load.add_argument("--min-prior-games", type=int, default=5)
    schedule_load.add_argument(
        "--output",
        default=str(DEFAULT_CYCLE2_SCHEDULE_LOAD_OUTPUT),
        help="Derived report JSON path",
    )
    schedule_load.set_defaults(func=cmd_ablate_prior_schedule_load_compact)

    heldout = sub.add_parser(
        "evaluate-frozen-heldout",
        help=(
            "Issue #32 dedicated frozen-candidate 2025-26 evaluation. "
            "Requires --confirm-consume-2025-26. Never refreshes provider data."
        ),
    )
    heldout.add_argument(
        "--seasons",
        required=True,
        help=f"Must be exactly {EXACT_SEASONS_CLI}",
    )
    heldout.add_argument(
        "--confirm-consume-2025-26",
        dest="confirm_consume_2025_26",
        action="store_true",
        help="Required explicit confirmation before opening 2025-26 for this frozen candidate",
    )
    heldout.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    heldout.add_argument(
        "--output",
        default=str(DEFAULT_FROZEN_HELDOUT_OUTPUT),
        help="Derived report JSON path",
    )
    heldout.set_defaults(func=cmd_evaluate_frozen_heldout)

    provenance = sub.add_parser("provenance", help="Print the bounded source record")
    provenance.set_defaults(func=cmd_provenance)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
