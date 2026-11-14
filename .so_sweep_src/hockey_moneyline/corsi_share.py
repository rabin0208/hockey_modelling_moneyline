"""Issue #25 prior Corsi share feature-family contract.

This experiment keeps the Issue #21 incumbent family unchanged and adds only
leakage-safe prior Corsi share columns constructed from ``shotAttemptsFor``
and ``shotAttemptsAgainst``. It reuses the Issue #19 inspect/join primitive
and does not create a second source pathway. Row-level ``corsiPercentage``
is never used.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import (
    CORSI_FEATURE_COLUMNS,
    CORSI_RAW_FIELDS,
    XGOAL_FEATURE_COLUMNS,
    XGOAL_RAW_FIELDS,
)
from hockey_moneyline.elo import ELO_FEATURE_COLUMN
from hockey_moneyline.errors import FeatureError
from hockey_moneyline.linear_recency import LINEAR_RECENCY_COLUMNS
from hockey_moneyline.ordered_lags import ORDERED_LAG_COLUMNS, RECENT_WIN_RATE_COLUMNS
from hockey_moneyline.xgoal_share import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
    assert_prior_xgoal_share_contract,
    xgoal_features_valid,
)

BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE = (
    "baseline_v1_plus_elo_plus_prior_xgoal_share_plus_prior_corsi_share"
)

PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS = (
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS + CORSI_FEATURE_COLUMNS
)

FORBIDDEN_EXPERIMENTAL_COLUMNS = (
    "corsiPercentage",
    "xGoalsPercentage",
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

# Published Issue #21 / PR #22 validation metrics for the carry-forward incumbent.
ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS = {
    "fold_a": {
        "accuracy": 0.619556,
        "roc_auc": 0.654698,
        "log_loss": 0.660324,
        "brier": 0.233328,
        "ece": 0.041597,
    },
    "fold_b": {
        "accuracy": 0.591057,
        "roc_auc": 0.628234,
        "log_loss": 0.669898,
        "brier": 0.238316,
        "ece": 0.054279,
    },
    "fold_c": {
        "accuracy": 0.591870,
        "roc_auc": 0.605962,
        "log_loss": 0.674706,
        "brier": 0.240804,
        "ece": 0.044352,
    },
    "macro": {
        "accuracy": 0.600828,
        "roc_auc": 0.629631,
        "log_loss": 0.668309,
        "brier": 0.237483,
        "ece": 0.046743,
    },
}

ISSUE21_ELIGIBLE_ROW_COUNTS = {
    "fold_a": {"train": 1121, "evaluation": 1217},
    "fold_b": {"train": 2338, "evaluation": 1230},
    "fold_c": {"train": 3568, "evaluation": 1230},
}

INCUMBENT_REPRODUCTION_ATOL = 1e-6

CORSI_SHARE_CONFIG = {
    "name": "prior_corsi_share",
    "raw_fields": list(CORSI_RAW_FIELDS),
    "unused_row_level_percentage": "corsiPercentage",
    "uses_corsiPercentage": False,
    "added_columns": list(CORSI_FEATURE_COLUMNS),
    "removed_from_incumbent": [],
    "incumbent_family": BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    "experimental_family": (
        BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE
    ),
    "aggregation": (
        "prior_corsi_share = sum(prior shotAttemptsFor) / "
        "(sum(prior shotAttemptsFor) + sum(prior shotAttemptsAgainst))"
    ),
    "timing": (
        "All baseline, Elo, xGoal-share, and Corsi-share snapshots for a "
        "calendar date are emitted before any same-day result updates team, "
        "Elo, or advanced state. Features continue to use only "
        "history.game_date < target.game_date."
    ),
    "minimum_history": (
        "Eligible train/evaluation rows still require at least five prior "
        "canonical games for both teams. Experimental rows also require finite "
        "prior xGoal and Corsi share features; missing values are not imputed."
    ),
    "reuses_issue19_primitive": True,
    "identities_independent": ("ARI", "UTA"),
}


def corsi_features_valid(row: Mapping[str, Any]) -> bool:
    """Return True when both approved Corsi-share columns are finite."""
    for column in CORSI_FEATURE_COLUMNS:
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


def xgoal_and_corsi_features_valid(row: Mapping[str, Any]) -> bool:
    """Return True when incumbent xGoal shares and Corsi shares are finite."""
    return xgoal_features_valid(row) and corsi_features_valid(row)


def attach_pregame_corsi_share_features(
    modeling_rows: Sequence[Mapping[str, Any]],
    corsi_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join Issue #25 Corsi snapshots onto incumbent modeling rows.

    Identity, date, and cardinality disagreements fail visibly. Missing Corsi
    shares remain ``None``; they are not imputed. ``corsiPercentage`` is not
    copied onto modeling rows.
    """
    if len(modeling_rows) != len(corsi_rows):
        raise FeatureError(
            "Issue #19 Corsi primitive produced "
            f"{len(corsi_rows)} rows but incumbent modeling has {len(modeling_rows)} rows"
        )

    corsi_by_id: dict[str, Mapping[str, Any]] = {}
    for corsi_row in corsi_rows:
        game_id = str(corsi_row["game_id"])
        if game_id in corsi_by_id:
            raise FeatureError(f"Duplicate Corsi feature row for game {game_id}")
        corsi_by_id[game_id] = corsi_row

    modeling_ids = [str(row["game_id"]) for row in modeling_rows]
    if len(set(modeling_ids)) != len(modeling_ids):
        raise FeatureError("Incumbent modeling rows contain duplicate game_id values")
    if set(corsi_by_id) != set(modeling_ids):
        missing = sorted(set(modeling_ids) - set(corsi_by_id))
        extra = sorted(set(corsi_by_id) - set(modeling_ids))
        raise FeatureError(
            "Issue #19 Corsi primitive cannot attach cleanly to incumbent "
            f"modeling rows; missing={missing[:10]} extra={extra[:10]}"
        )

    attached: list[dict[str, Any]] = []
    for row in modeling_rows:
        corsi_row = corsi_by_id[str(row["game_id"])]
        if (
            corsi_row["home_team_id"] != row["home_team_id"]
            or corsi_row["away_team_id"] != row["away_team_id"]
            or str(corsi_row["game_date"]) != str(row["game_date"])
            or int(corsi_row["season"]) != int(row["season"])
        ):
            raise FeatureError(
                "Issue #19 Corsi identity/date/season disagrees with modeling "
                f"row {row['game_id']}"
            )
        merged = dict(row)
        for column in CORSI_FEATURE_COLUMNS:
            merged[column] = corsi_row[column]
        attached.append(merged)
    return attached


def assert_prior_corsi_share_contract() -> None:
    """Guard incumbent Issue #21 columns and the Issue #25 experimental family."""
    assert_prior_xgoal_share_contract()
    if list(PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS) != [
        "home_prior_games",
        "away_prior_games",
        "home_win_rate",
        "away_win_rate",
        "home_goals_for_per_game",
        "away_goals_for_per_game",
        "home_goals_against_per_game",
        "away_goals_against_per_game",
        "home_goal_diff_per_game",
        "away_goal_diff_per_game",
        "home_home_win_rate",
        "away_away_win_rate",
        "home_season_games",
        "away_season_games",
        "home_season_win_rate",
        "away_season_win_rate",
        "home_recent_win_rate",
        "away_recent_win_rate",
        "home_elo_minus_away_elo",
        "home_prior_xgoal_share",
        "away_prior_xgoal_share",
    ]:
        raise RuntimeError("Issue #21 incumbent 21-column family was altered")
    if any(
        column not in PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS
        for column in RECENT_WIN_RATE_COLUMNS
    ):
        raise RuntimeError("incumbent family must keep rolling recent-win-rate columns")
    if ELO_FEATURE_COLUMN not in PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS:
        raise RuntimeError("incumbent family must retain home_elo_minus_away_elo")
    if XGOAL_FEATURE_COLUMNS != ("home_prior_xgoal_share", "away_prior_xgoal_share"):
        raise RuntimeError("Issue #21 xGoal feature columns were altered")
    if XGOAL_RAW_FIELDS != ("xGoalsFor", "xGoalsAgainst"):
        raise RuntimeError("Issue #19 xGoal raw fields were altered")
    if CORSI_RAW_FIELDS != ("shotAttemptsFor", "shotAttemptsAgainst"):
        raise RuntimeError("Issue #25 Corsi raw fields must be shotAttemptsFor/shotAttemptsAgainst")
    if CORSI_FEATURE_COLUMNS != ("home_prior_corsi_share", "away_prior_corsi_share"):
        raise RuntimeError("Issue #25 Corsi feature columns were altered")
    if PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS != (
        PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS + CORSI_FEATURE_COLUMNS
    ):
        raise RuntimeError(
            "experimental family must equal the Issue #21 incumbent plus the two Corsi-share columns"
        )
    forbidden = set(FORBIDDEN_EXPERIMENTAL_COLUMNS) & set(
        PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS
    )
    if forbidden:
        raise RuntimeError(
            "experimental family includes disallowed columns: "
            f"{sorted(forbidden)}"
        )
    if "corsiPercentage" in PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS:
        raise RuntimeError("experimental family must not include row-level corsiPercentage")
    if ELO_FEATURE_COLUMN not in PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS:
        raise RuntimeError("experimental family must retain home_elo_minus_away_elo")
    if any(
        column not in PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS
        for column in XGOAL_FEATURE_COLUMNS
    ):
        raise RuntimeError("experimental family must retain both prior xGoal share columns")
