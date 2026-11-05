"""Issue #17 fixed linear recency-weighted 10-game recent-form contract.

This experiment keeps the incumbent 10-game history window and replaces
uniform recent win rate with a pre-specified linear recency weighting.
It does not reuse the Issue #15 ordered-lag family as the experimental model.
"""

from __future__ import annotations

from typing import Sequence

from hockey_moneyline.elo import ELO_FEATURE_COLUMN, PLUS_ELO_FEATURE_COLUMNS
from hockey_moneyline.ordered_lags import ORDERED_LAG_COLUMNS, RECENT_WIN_RATE_COLUMNS
from hockey_moneyline.provenance import RECENT_FORM_WINDOW
from hockey_moneyline.schema import FEATURE_COLUMNS

LINEAR_RECENCY_WINDOW = RECENT_FORM_WINDOW
LINEAR_RECENCY_WEIGHTS = (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
LINEAR_RECENCY_COLUMNS = (
    "home_recent_win_rate_linear10",
    "away_recent_win_rate_linear10",
)
BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10 = "baseline_v1_plus_elo_linear_recency10"

PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in RECENT_WIN_RATE_COLUMNS
) + LINEAR_RECENCY_COLUMNS + (ELO_FEATURE_COLUMN,)

LINEAR_RECENCY_CONFIG = {
    "window": LINEAR_RECENCY_WINDOW,
    "weights_by_lag": list(LINEAR_RECENCY_WEIGHTS),
    "encoding": {"win": 1.0, "loss": 0.0},
    "formula": (
        "weighted_recent_win_rate = sum(result_lag_k * (11-k)) / "
        "sum(11-k for available k)"
    ),
    "removed_from_incumbent": list(RECENT_WIN_RATE_COLUMNS),
    "added_columns": list(LINEAR_RECENCY_COLUMNS),
    "incumbent_family": "baseline_v1_plus_elo",
    "experimental_family": BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10,
    "ordered5_not_incumbent": True,
    "semantics": (
        "Use the team's most recent up to 10 completed games before the target "
        "date. lag_1 is the immediately preceding completed game and receives "
        "weight 10; lag_k receives weight 11-k. Wins encode as 1.0 and losses "
        "as 0.0. Rows with n < 10 prior games, but still eligible under the "
        "five-game minimum, normalize by the sum of the available lag_1..lag_n "
        "weights only. Unavailable prior games are not imputed."
    ),
    "timing": (
        "All baseline, Elo, and linear-recency features for a calendar date "
        "are snapshotted before any same-day result updates team history. "
        "Features continue to use only history.game_date < target.game_date."
    ),
    "minimum_history": (
        "Eligible train/evaluation rows still require at least five prior "
        "games for both teams."
    ),
}


def linear_recency_win_rate(recent_results: Sequence[int]) -> float:
    """Return the fixed linear recency-weighted win rate.

    ``recent_results`` is chronological oldest→newest. ``lag_1`` is the last
    element. At most the most recent ``LINEAR_RECENCY_WINDOW`` games are used.
    An empty history returns 0.5 to match the incumbent empty-window snapshot
    and is never selected by the five-game eligibility rule.
    """
    n = min(len(recent_results), LINEAR_RECENCY_WINDOW)
    if n == 0:
        return 0.5
    available = tuple(recent_results)[-n:]
    weighted_sum = 0.0
    weight_sum = 0.0
    for lag in range(1, n + 1):
        weight = LINEAR_RECENCY_WEIGHTS[lag - 1]
        weighted_sum += float(available[-lag]) * weight
        weight_sum += weight
    return weighted_sum / weight_sum


def assert_linear_recency_contract() -> None:
    """Guard incumbent columns and the Issue #17 experimental family."""
    if PLUS_ELO_FEATURE_COLUMNS != FEATURE_COLUMNS + (ELO_FEATURE_COLUMN,):
        raise RuntimeError("baseline_v1_plus_elo feature columns were altered")
    if LINEAR_RECENCY_WEIGHTS != (10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        raise RuntimeError("linear-recency weights must be exactly 10..1")
    if LINEAR_RECENCY_WINDOW != 10:
        raise RuntimeError("linear-recency window must remain 10 games")
    if any(
        column in PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS
        for column in RECENT_WIN_RATE_COLUMNS
    ):
        raise RuntimeError("linear-recency10 must remove rolling recent-win-rate columns")
    if any(
        column in PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS
        for column in ORDERED_LAG_COLUMNS
    ):
        raise RuntimeError("linear-recency10 must not reintroduce ordered5 lag columns")
    if ELO_FEATURE_COLUMN not in PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS:
        raise RuntimeError("linear-recency10 must retain home_elo_minus_away_elo")
    if set(LINEAR_RECENCY_COLUMNS) - set(PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS):
        raise RuntimeError("linear-recency10 is missing required weighted columns")
    if any(column not in PLUS_ELO_FEATURE_COLUMNS for column in RECENT_WIN_RATE_COLUMNS):
        raise RuntimeError("incumbent family must keep rolling recent-win-rate columns")
