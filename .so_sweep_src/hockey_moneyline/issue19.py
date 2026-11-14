"""Reproducible Issue #19 MoneyPuck advanced-stat contract validation command.

Usage from a network-enabled development environment:

    python -m hockey_moneyline.issue19 --refresh

The command is hard-bounded to 2021-22 through 2024-25. It never accepts the
2025-26 held-out season as an inspection target and writes derived JSON only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hockey_moneyline.acquire import DEFAULT_CACHE_DIR, acquire_source_csv
from hockey_moneyline.advanced_stats import ISSUE19_SEASONS, inspect_advanced_stats_csv
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.provenance import SOURCE_FILE_URL, SOURCE_PAGE

DEFAULT_OUTPUT = Path("data/derived/issue19_advanced_contract_report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #19 MoneyPuck advanced team-stat source/schema/join/timing validation"
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = acquire_source_csv(cache_dir=Path(args.cache_dir), refresh=args.refresh)
        inspection = inspect_advanced_stats_csv(path, seasons=ISSUE19_SEASONS)
        report = dict(inspection.report)
        report["source"] = {
            "page": SOURCE_PAGE,
            "file": SOURCE_FILE_URL,
            "credit": "MoneyPuck.com",
            "raw_file_committed": False,
            "definition_reference": "https://www.moneypuck.com/glossary.htm",
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
