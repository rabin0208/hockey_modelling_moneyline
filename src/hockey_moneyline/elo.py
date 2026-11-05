"""Fixed Issue #13 opponent-adjusted Elo contract.

This is a single un-tuned feature-family experiment. Ratings are result-only,
have no home-ice or margin adjustment, and do not reset between seasons.
"""

from __future__ import annotations

from hockey_moneyline.schema import FEATURE_COLUMNS

ELO_FEATURE_COLUMN = "home_elo_minus_away_elo"
ELO_INITIAL = 1500.0
ELO_SCALE = 400.0
ELO_K = 20.0

PLUS_ELO_FEATURE_COLUMNS = FEATURE_COLUMNS + (ELO_FEATURE_COLUMN,)

ELO_CONFIG = {
    "initial_rating": ELO_INITIAL,
    "scale": ELO_SCALE,
    "k": ELO_K,
    "update": "result_only",
    "margin_of_victory_adjustment": False,
    "home_ice_adjustment": False,
    "season_reset": False,
    "season_regression": False,
    "new_team_rating": ELO_INITIAL,
    "identities_independent": ("ARI", "UTA"),
    "timing": (
        "All Elo features for a calendar date are snapshotted before any "
        "same-day result updates Elo state. Features continue to use only "
        "history.game_date < target.game_date."
    ),
}


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / ELO_SCALE))


def updated_ratings(
    home_rating: float,
    away_rating: float,
    home_won: bool,
) -> tuple[float, float]:
    home_expected = expected_score(home_rating, away_rating)
    home_score = 1.0 if home_won else 0.0
    new_home = home_rating + ELO_K * (home_score - home_expected)
    new_away = away_rating + ELO_K * ((1.0 - home_score) - (1.0 - home_expected))
    return new_home, new_away
