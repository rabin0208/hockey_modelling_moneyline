"""Fit and commit the promoted SO-branch frozen artifact, then refresh Results."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import create_features as cf  # noqa: E402
from hockey_moneyline.prospective_artifact import (  # noqa: E402
    FROZEN_FEATURE_COLUMNS,
    FROZEN_HISTORICAL_CACHE_SHA256,
    committed_frozen_artifact_path,
    fit_frozen_prospective_artifact,
    write_frozen_artifact,
)


def _patch_expected_constants(
    *,
    fitted_sha: str,
    row_sha: str,
    row_count: int,
) -> None:
    path = SRC / "hockey_moneyline" / "prospective_artifact.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'EXPECTED_FITTED_ARTIFACT_SHA256 = "[^"]*"',
        f'EXPECTED_FITTED_ARTIFACT_SHA256 = "{fitted_sha}"',
        text,
        count=1,
    )
    text = re.sub(
        r'EXPECTED_TRAINING_ROW_IDENTITY_SHA256 = "[^"]*"',
        f'EXPECTED_TRAINING_ROW_IDENTITY_SHA256 = "{row_sha}"',
        text,
        count=1,
    )
    text = re.sub(
        r"EXPECTED_FROZEN_TRAINING_ROW_COUNT = \d+",
        f"EXPECTED_FROZEN_TRAINING_ROW_COUNT = {row_count}",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    print("Loading SO-inclusive source...")
    cf._SOURCE = None
    cf._SOURCE_STAT = None
    source = cf.load_source(refresh=False)
    development = [game for game in source.games if game.season < 20262027]
    print(
        f"Fitting frozen artifact on {len(development)} development games; "
        f"{len(FROZEN_FEATURE_COLUMNS)} features; "
        f"cache={FROZEN_HISTORICAL_CACHE_SHA256[:12]}..."
    )
    artifact = fit_frozen_prospective_artifact(
        development,
        source.records,
        [item for item in source.schedule if item.season < 20262027],
        historical_cache_sha256=FROZEN_HISTORICAL_CACHE_SHA256,
    )
    out = write_frozen_artifact(committed_frozen_artifact_path(), artifact)
    _patch_expected_constants(
        fitted_sha=artifact.fitted_artifact_sha256,
        row_sha=artifact.training_row_identity_sha256,
        row_count=artifact.training_row_count,
    )
    print(
        f"Wrote {out}\n"
        f"  rows={artifact.training_row_count}\n"
        f"  fitted_sha={artifact.fitted_artifact_sha256}\n"
        f"  row_sha={artifact.training_row_identity_sha256}\n"
        f"  columns={len(artifact.feature_columns)}"
    )

    print("\nRegenerating data/game_features.csv...")
    cf.main()

    print("\nRegenerating Results CSVs...")
    import evaluate_historical as eh  # noqa: E402

    eh.main()
    print("Promotion complete.")


if __name__ == "__main__":
    main()
