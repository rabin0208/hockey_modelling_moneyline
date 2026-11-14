"""Issue #21 prior xGoal share feature-family contract.

This experiment keeps the Issue #13 incumbent family unchanged and adds only
the already-validated Issue #19 ``prior_xgoal_share`` columns. It reuses
``build_pregame_xgoal_share_features`` and does not create a second rolling
xGoal implementation.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import XGOAL_FEATURE_COLUMNS, XGOAL_RAW_FIELDS
from hockey_moneyline.elo import ELO_FEATURE_COLUMN, PLUS_ELO_FEATURE_COLUMNS
from hockey_moneyline.errors import FeatureError
from hockey_moneyline.linear_recency import LINEAR_RECENCY_COLUMNS
from hockey_moneyline.ordered_lags import ORDERED_LAG_COLUMNS, RECENT_WIN_RATE_COLUMNS
from hockey_moneyline.schema import FEATURE_COLUMNS

BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE = (
    "baseline_v1_plus_elo_plus_prior_xgoal_share"
)

PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS = (
    PLUS_ELO_FEATURE_COLUMNS + XGOAL_FEATURE_COLUMNS
)

FORBIDDEN_EXPERIMENTAL_COLUMNS = (
    "xGoalsPercentage",
    "corsiPercentage",
    "fenwickPercentage",
    "shotAttemptsFor",
    "shotAttemptsAgainst",
    "unblockedShotAttemptsFor",
    "unblockedShotAttemptsAgainst",
    "lowDangerShotsFor",
    "lowDangerShotsAgainst",
    "mediumDangerShotsFor",
    "mediumDangerShotsAgainst",
    "highDangerShotsFor",
    "highDangerShotsAgainst",
) + ORDERED_LAG_COLUMNS + LINEAR_RECENCY_COLUMNS

# Published Issue #13 / PR #14 validation metrics for baseline_v1_plus_elo.
ISSUE13_PLUS_ELO_VALIDATION_METRICS = {
    "fold_a": {
        "accuracy": 0.603944,
        "roc_auc": 0.643312,
        "log_loss": 0.665197,
        "brier": 0.235870,
        "ece": 0.045684,
    },
    "fold_b": {
        "accuracy": 0.588618,
        "roc_auc": 0.628965,
        "log_loss": 0.668318,
        "brier": 0.237555,
        "ece": 0.030934,
    },
    "fold_c": {
        "accuracy": 0.581301,
        "roc_auc": 0.603217,
        "log_loss": 0.677601,
        "brier": 0.241852,
        "ece": 0.049449,
    },
    "macro": {
        "accuracy": 0.591288,
        "roc_auc": 0.625165,
        "log_loss": 0.670372,
        "brier": 0.238426,
        "ece": 0.042022,
    },
}

INCUMBENT_REPRODUCTION_ATOL = 1e-6

XGOAL_SHARE_CONFIG = {
    "name": "prior_xgoal_share",
    "raw_fields": list(XGOAL_RAW_FIELDS),
    "added_columns": list(XGOAL_FEATURE_COLUMNS),
    "removed_from_incumbent": [],
    "incumbent_family": "baseline_v1_plus_elo",
    "experimental_family": BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    "aggregation": (
        "prior_xgoal_share = sum(prior xGoalsFor) / "
        "(sum(prior xGoalsFor) + sum(prior xGoalsAgainst))"
    ),
    "timing": (
        "All baseline, Elo, and xGoal-share snapshots for a calendar date are "
        "emitted before any same-day result updates team, Elo, or advanced "
        "state. Features continue to use only history.game_date < target.game_date."
    ),
    "minimum_history": (
        "Eligible train/evaluation rows still require at least five prior "
        "canonical games for both teams. Experimental rows also require finite "
        "prior xGoal share features; missing values are not imputed."
    ),
    "reuses_issue19_primitive": True,
    "identities_independent": ("ARI", "UTA"),
}


def xgoal_features_valid(row: Mapping[str, Any]) -> bool:
    """Return True when both approved xGoal-share columns are finite."""
    for column in XGOAL_FEATURE_COLUMNS:
        if column not in row:
            return False
        value = row[column]
        if value is None:
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number):
            return False
    return True


def attach_pregame_xgoal_share_features(
    modeling_rows: Sequence[Mapping[str, Any]],
    xgoal_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join Issue #19 xGoal snapshots onto incumbent modeling rows.

    Identity, date, and cardinality disagreements fail visibly. Missing xGoal
    shares remain ``None``; they are not imputed.
    """
    if len(modeling_rows) != len(xgoal_rows):
        raise FeatureError(
            "Issue #19 xGoal primitive produced "
            f"{len(xgoal_rows)} rows but incumbent modeling has {len(modeling_rows)} rows"
        )

    xgoal_by_id: dict[str, Mapping[str, Any]] = {}
    for xgoal_row in xgoal_rows:
        game_id = str(xgoal_row["game_id"])
        if game_id in xgoal_by_id:
            raise FeatureError(f"Duplicate xGoal feature row for game {game_id}")
        xgoal_by_id[game_id] = xgoal_row

    modeling_ids = [str(row["game_id"]) for row in modeling_rows]
    if len(set(modeling_ids)) != len(modeling_ids):
        raise FeatureError("Incumbent modeling rows contain duplicate game_id values")
    if set(xgoal_by_id) != set(modeling_ids):
        missing = sorted(set(modeling_ids) - set(xgoal_by_id))
        extra = sorted(set(xgoal_by_id) - set(modeling_ids))
        raise FeatureError(
            "Issue #19 xGoal primitive cannot attach cleanly to incumbent "
            f"modeling rows; missing={missing[:10]} extra={extra[:10]}"
        )

    attached: list[dict[str, Any]] = []
    for row in modeling_rows:
        xgoal_row = xgoal_by_id[str(row["game_id"])]
        if (
            xgoal_row["home_team_id"] != row["home_team_id"]
            or xgoal_row["away_team_id"] != row["away_team_id"]
            or str(xgoal_row["game_date"]) != str(row["game_date"])
            or int(xgoal_row["season"]) != int(row["season"])
        ):
            raise FeatureError(
                "Issue #19 xGoal identity/date/season disagrees with modeling "
                f"row {row['game_id']}"
            )
        merged = dict(row)
        for column in XGOAL_FEATURE_COLUMNS:
            merged[column] = xgoal_row[column]
        merged["home_advanced_prior_games"] = xgoal_row["home_advanced_prior_games"]
        merged["away_advanced_prior_games"] = xgoal_row["away_advanced_prior_games"]
        attached.append(merged)
    return attached


def assert_prior_xgoal_share_contract() -> None:
    """Guard incumbent columns and the Issue #21 experimental family."""
    if PLUS_ELO_FEATURE_COLUMNS != FEATURE_COLUMNS + (ELO_FEATURE_COLUMN,):
        raise RuntimeError("baseline_v1_plus_elo feature columns were altered")
    if any(column not in PLUS_ELO_FEATURE_COLUMNS for column in RECENT_WIN_RATE_COLUMNS):
        raise RuntimeError("incumbent family must keep rolling recent-win-rate columns")
    if ELO_FEATURE_COLUMN not in PLUS_ELO_FEATURE_COLUMNS:
        raise RuntimeError("incumbent family must retain home_elo_minus_away_elo")
    if PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS != (
        PLUS_ELO_FEATURE_COLUMNS + XGOAL_FEATURE_COLUMNS
    ):
        raise RuntimeError("experimental family must equal incumbent plus the two xGoal-share columns")
    if XGOAL_RAW_FIELDS != ("xGoalsFor", "xGoalsAgainst"):
        raise RuntimeError("Issue #19 xGoal raw fields were altered")
    if XGOAL_FEATURE_COLUMNS != ("home_prior_xgoal_share", "away_prior_xgoal_share"):
        raise RuntimeError("Issue #19 xGoal feature columns were altered")
    forbidden = set(FORBIDDEN_EXPERIMENTAL_COLUMNS) & set(
        PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS
    )
    if forbidden:
        raise RuntimeError(
            "experimental family includes disallowed columns: "
            f"{sorted(forbidden)}"
        )
    if any(
        column not in PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS
        for column in RECENT_WIN_RATE_COLUMNS
    ):
        raise RuntimeError("experimental family must retain rolling recent-win-rate columns")
    if ELO_FEATURE_COLUMN not in PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS:
        raise RuntimeError("experimental family must retain home_elo_minus_away_elo")
