"""Fixed Issue #13 opponent-adjusted Elo contract.

This is a single un-tuned feature-family experiment. Ratings are result-only,
have no home-ice or margin adjustment. Career-prior windowing (see
``PRIOR_SEASON_WINDOW``) also limits which past seasons contribute to Elo.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from hockey_moneyline.provenance import seasons_in_prior_window
from hockey_moneyline.schema import FEATURE_COLUMNS

ELO_FEATURE_COLUMN = "home_elo_minus_away_elo"
ELO_INITIAL = 1500.0
ELO_SCALE = 400.0
ELO_K = 20.0
# Shootouts are noisier than regulation/OT; move Elo less far on those results.
ELO_K_SHOOTOUT = 5.0

PLUS_ELO_FEATURE_COLUMNS = FEATURE_COLUMNS + (ELO_FEATURE_COLUMN,)

ELO_CONFIG = {
    "initial_rating": ELO_INITIAL,
    "scale": ELO_SCALE,
    "k": ELO_K,
    "k_shootout": ELO_K_SHOOTOUT,
    "update": "result_only",
    "margin_of_victory_adjustment": False,
    "home_ice_adjustment": False,
    "season_reset": False,
    "season_regression": False,
    "prior_season_window": "uses provenance.PRIOR_SEASON_WINDOW",
    "new_team_rating": ELO_INITIAL,
    "identities_independent": ("ARI", "UTA"),
    "timing": (
        "All Elo features for a calendar date are snapshotted before any "
        "same-day result updates Elo state. Features continue to use only "
        "history.game_date < target.game_date. When PRIOR_SEASON_WINDOW is set, "
        "ratings are rebuilt from only that rolling season window."
    ),
}


@dataclass(frozen=True, slots=True)
class EloEvent:
    season: int
    home_team_id: str
    away_team_id: str
    home_won: bool
    decided_in_shootout: bool = False


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / ELO_SCALE))


def updated_ratings(
    home_rating: float,
    away_rating: float,
    home_won: bool,
    *,
    decided_in_shootout: bool = False,
    k: float | None = None,
) -> tuple[float, float]:
    if k is None:
        k = ELO_K_SHOOTOUT if decided_in_shootout else ELO_K
    home_expected = expected_score(home_rating, away_rating)
    home_score = 1.0 if home_won else 0.0
    new_home = home_rating + k * (home_score - home_expected)
    new_away = away_rating + k * ((1.0 - home_score) - (1.0 - home_expected))
    return new_home, new_away


def ratings_from_events(
    events: Iterable[EloEvent],
    current_season: int,
) -> dict[str, float]:
    """Replay Elo using only events inside the prior-season window."""

    allowed = seasons_in_prior_window(current_season)
    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    for event in events:
        if allowed is not None and event.season not in allowed:
            continue
        new_home, new_away = updated_ratings(
            ratings[event.home_team_id],
            ratings[event.away_team_id],
            event.home_won,
            decided_in_shootout=event.decided_in_shootout,
        )
        ratings[event.home_team_id] = new_home
        ratings[event.away_team_id] = new_away
    return ratings
