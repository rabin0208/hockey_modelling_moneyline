"""Reproducible Issue #34 Cycle-2 schedule-load contract inspection.

Usage from an existing network-enabled development environment:

    python -m hockey_moneyline.issue34 --refresh

Only 2021-22 through 2025-26 are selected. 2026-27 is never selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hockey_moneyline.acquire import DEFAULT_CACHE_DIR, acquire_source_csv
from hockey_moneyline.errors import FeatureError, HockeyMoneylineError
from hockey_moneyline.normalize import games_from_moneypuck_csv
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    inspect_prior_schedule_load_contract,
    schedule_history_from_moneypuck_csv,
)

DEFAULT_OUTPUT = Path("data/derived/issue34_schedule_load_contract_report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #34 Cycle-2 prior-only schedule-load contract validation"
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)

        # Modeling targets intentionally require a winner label and therefore
        # exclude tied situation=all rows whose shootout winner is unavailable.
        normalized = games_from_moneypuck_csv(
            path,
            seasons=CYCLE2_DEVELOPMENT_SEASONS,
        )

        # Schedule history intentionally does not require a winner label.
        # Those tied/unlabeled games still affect later rest/load state.
        schedule_history = schedule_history_from_moneypuck_csv(
            path,
            seasons=CYCLE2_DEVELOPMENT_SEASONS,
        )

        report = inspect_prior_schedule_load_contract(
            schedule_history,
            normalized.games,
        )

        history_only = report["cardinality"]["schedule_history_only_games_by_season"]
        tied_exclusions = {
            str(season): normalized.skipped_tied_scores_by_season.get(season, 0)
            for season in CYCLE2_DEVELOPMENT_SEASONS
        }
        if history_only != tied_exclusions:
            raise FeatureError(
                "Schedule-history-only game counts do not match the canonical "
                f"winner-label exclusions: history_only={history_only}, "
                f"tied_exclusions={tied_exclusions}"
            )

        report["normalization"] = {
            "modeling_target_games_by_season": {
                str(season): normalized.normalized_games_by_season.get(season, 0)
                for season in CYCLE2_DEVELOPMENT_SEASONS
            },
            "winner_label_exclusions_by_season": tied_exclusions,
            "schedule_history_games_by_season": report["cardinality"][
                "schedule_history_games_by_season"
            ],
            "raw_provider_file_committed": False,
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {output}")
        return 0
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
