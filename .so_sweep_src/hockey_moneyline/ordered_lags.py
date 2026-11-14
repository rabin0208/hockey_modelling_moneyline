"""Issue #15 ordered five-game recent-result lag feature contract.

This experiment replaces rolling recent-win-rate columns with explicit
ordered win/loss lags. Lag features encode result order only.
"""

from __future__ import annotations

from hockey_moneyline.elo import ELO_FEATURE_COLUMN, PLUS_ELO_FEATURE_COLUMNS
from hockey_moneyline.schema import FEATURE_COLUMNS

ORDERED_LAG_COUNT = 5
ORDERED_LAG_ENCODING = {"win": 1.0, "loss": 0.0}

HOME_RESULT_LAG_COLUMNS = tuple(
    f"home_result_lag_{index}" for index in range(1, ORDERED_LAG_COUNT + 1)
)
AWAY_RESULT_LAG_COLUMNS = tuple(
    f"away_result_lag_{index}" for index in range(1, ORDERED_LAG_COUNT + 1)
)
ORDERED_LAG_COLUMNS = HOME_RESULT_LAG_COLUMNS + AWAY_RESULT_LAG_COLUMNS

RECENT_WIN_RATE_COLUMNS = ("home_recent_win_rate", "away_recent_win_rate")

BASELINE_V1_PLUS_ELO_ORDERED5 = "baseline_v1_plus_elo_ordered5"

PLUS_ELO_ORDERED5_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in RECENT_WIN_RATE_COLUMNS
) + ORDERED_LAG_COLUMNS + (ELO_FEATURE_COLUMN,)

ORDERED_LAG_CONFIG = {
    "lag_count": ORDERED_LAG_COUNT,
    "encoding": dict(ORDERED_LAG_ENCODING),
    "removed_from_incumbent": list(RECENT_WIN_RATE_COLUMNS),
    "added_columns": list(ORDERED_LAG_COLUMNS),
    "incumbent_family": "baseline_v1_plus_elo",
    "experimental_family": BASELINE_V1_PLUS_ELO_ORDERED5,
    "semantics": (
        "lag_1 is the team's immediately preceding completed game before the "
        "target game date; lag_k is the k-th most recent completed game. "
        "Encoding is win=1.0, loss=0.0. Lags carry only the team's own result; "
        "no goals, margin, opponent, venue, Corsi, xG, or shots are encoded."
    ),
    "timing": (
        "All baseline, Elo, and lag features for a calendar date are "
        "snapshotted before any same-day result updates team history. "
        "Features continue to use only history.game_date < target.game_date."
    ),
    "minimum_history": (
        "Eligible train/evaluation rows still require at least five prior "
        "games for both teams. Selected ordered5 matrices assert that all "
        "ten lag values are available."
    ),
}


def assert_incumbent_columns_unchanged() -> None:
    """Guard against silently altering the Issue #13 incumbent feature list."""
    if PLUS_ELO_FEATURE_COLUMNS != FEATURE_COLUMNS + (ELO_FEATURE_COLUMN,):
        raise RuntimeError("baseline_v1_plus_elo feature columns were altered")
    if any(column in PLUS_ELO_ORDERED5_FEATURE_COLUMNS for column in RECENT_WIN_RATE_COLUMNS):
        raise RuntimeError("ordered5 must remove rolling recent-win-rate columns")
    if ELO_FEATURE_COLUMN not in PLUS_ELO_ORDERED5_FEATURE_COLUMNS:
        raise RuntimeError("ordered5 must retain home_elo_minus_away_elo")
    if set(ORDERED_LAG_COLUMNS) - set(PLUS_ELO_ORDERED5_FEATURE_COLUMNS):
        raise RuntimeError("ordered5 is missing required lag columns")
