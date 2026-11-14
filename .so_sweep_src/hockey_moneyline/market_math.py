"""Deterministic two-sided decimal moneyline math for Issue #78.

This module implements the canonical formulas only. It does not read odds
files, call a provider, choose a betting threshold, or stake a bankroll.
Rabin repository code is not copied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hockey_moneyline.errors import MarketMathError

VIG_REMOVAL_METHOD = "proportional_no_vig_v1"


def _require_decimal_odds(value: object, *, side: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarketMathError(f"{side} decimal odds are malformed")
    odds = float(value)
    if not math.isfinite(odds) or odds <= 1.0:
        raise MarketMathError(f"{side} decimal odds must be finite and greater than 1")
    return odds


def _require_probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarketMathError(f"{label} is malformed")
    probability = float(value)
    if not math.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
        raise MarketMathError(f"{label} must be finite and strictly between 0 and 1")
    return probability


def implied_probability(decimal_odds: object, *, side: str) -> float:
    """Sportsbook implied probability ``q = 1 / d`` for one decimal price."""

    return 1.0 / _require_decimal_odds(decimal_odds, side=side)


def overround(home_implied: float, away_implied: float) -> float:
    """Two-sided overround ``q_home + q_away - 1``."""

    return float(home_implied) + float(away_implied) - 1.0


def proportional_fair_probabilities(
    home_implied: float,
    away_implied: float,
) -> tuple[float, float]:
    """Proportional no-vig probabilities ``p_fair = q / (q_home + q_away)``."""

    total = float(home_implied) + float(away_implied)
    if not math.isfinite(total) or total <= 0.0:
        raise MarketMathError("implied probabilities cannot be normalized")
    return float(home_implied) / total, float(away_implied) / total


def model_market_edge(model_probability: float, fair_probability: float) -> float:
    """Model-minus-fair edge. This is not expected value and not a stake."""

    return float(model_probability) - float(fair_probability)


def expected_value_per_unit(model_probability: float, decimal_odds: float) -> float:
    """Expected net return per 1 unit at the offered decimal price: ``p * d - 1``."""

    return float(model_probability) * float(decimal_odds) - 1.0


@dataclass(frozen=True, slots=True)
class TwoSidedMarketMath:
    """Computed market fields. Qualification and stake are intentionally absent."""

    vig_removal_method: str
    home_decimal_odds: float
    away_decimal_odds: float
    home_implied_probability: float
    away_implied_probability: float
    overround: float
    home_fair_market_probability: float
    away_fair_market_probability: float
    model_home_win_probability: float
    model_away_win_probability: float
    home_edge: float
    away_edge: float
    home_ev_per_unit: float
    away_ev_per_unit: float


def price_two_sided_moneyline(
    home_decimal_odds: object,
    away_decimal_odds: object,
    model_home_win_probability: object,
    model_away_win_probability: object,
    *,
    probability_sum_tolerance: float = 1e-6,
) -> TwoSidedMarketMath:
    """Price one two-sided decimal moneyline snapshot.

    Both offered prices are required. The missing side is never inferred.
    Model probabilities are not renormalized.
    """

    home_odds = _require_decimal_odds(home_decimal_odds, side="home")
    away_odds = _require_decimal_odds(away_decimal_odds, side="away")
    model_home = _require_probability(
        model_home_win_probability,
        label="model home win probability",
    )
    model_away = _require_probability(
        model_away_win_probability,
        label="model away win probability",
    )
    if abs((model_home + model_away) - 1.0) > probability_sum_tolerance:
        raise MarketMathError(
            "model home and away probabilities must sum to 1; they are not renormalized"
        )
    home_implied = 1.0 / home_odds
    away_implied = 1.0 / away_odds
    home_fair, away_fair = proportional_fair_probabilities(home_implied, away_implied)
    return TwoSidedMarketMath(
        vig_removal_method=VIG_REMOVAL_METHOD,
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
        home_implied_probability=home_implied,
        away_implied_probability=away_implied,
        overround=overround(home_implied, away_implied),
        home_fair_market_probability=home_fair,
        away_fair_market_probability=away_fair,
        model_home_win_probability=model_home,
        model_away_win_probability=model_away,
        home_edge=model_market_edge(model_home, home_fair),
        away_edge=model_market_edge(model_away, away_fair),
        home_ev_per_unit=expected_value_per_unit(model_home, home_odds),
        away_ev_per_unit=expected_value_per_unit(model_away, away_odds),
    )
