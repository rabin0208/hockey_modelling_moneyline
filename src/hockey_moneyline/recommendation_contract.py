"""Versioned odds-edge recommendation contract for Issue #78.

The recommendation record is a separate layer from protected prediction
records. This module does not score the frozen model, read 2026-27 outcomes,
contact an odds provider, or place a wager.

Schema v1 is moneyline-only. Any other market label fails closed.

No numeric BET cutoff is built in. With no explicitly supplied threshold
policy, a complete two-sided moneyline snapshot stays ``PASS`` under
``blocked-no-approved-threshold``. No maximum quote age is applied. A
quote-age cutoff is unapproved and stays under the odds/provider-policy
dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hockey_moneyline.errors import MarketMathError, RecommendationContractError
from hockey_moneyline.market_math import VIG_REMOVAL_METHOD, price_two_sided_moneyline

RECOMMENDATION_SCHEMA_VERSION = 1
RECOMMENDATION_KIND = "odds_edge_recommendation"
RECOMMENDATION_STATES = ("BET", "PASS", "NO_DATA")
QUALIFIED_SIDES = ("home", "away")
BLOCKED_THRESHOLD_POLICY_VERSION = "blocked-no-approved-threshold"
SCHEMA_V1_MARKET = "moneyline"
DEFAULT_MARKET = SCHEMA_V1_MARKET
PROBABILITY_SUM_TOLERANCE = 1e-6

_PROVENANCE_REQUIRED = ("source_name", "asset_identity", "terms_note")


def qualifying_side(
    home_edge: float,
    away_edge: float,
    minimum_edge: float,
) -> str | None:
    """Return the single side whose edge is strictly above ``minimum_edge``.

    Both sides qualifying is an error. Neither side qualifying returns None.
    """

    if not math.isfinite(minimum_edge) or minimum_edge < 0.0:
        raise RecommendationContractError("minimum edge must be finite and >= 0")
    home_qualifies = home_edge > minimum_edge
    away_qualifies = away_edge > minimum_edge
    if home_qualifies and away_qualifies:
        raise RecommendationContractError(
            "home and away edges both exceed the threshold"
        )
    if home_qualifies:
        return "home"
    if away_qualifies:
        return "away"
    return None


@dataclass(frozen=True, slots=True)
class EdgeThresholdPolicy:
    """Explicit research policy. Absence of this object blocks BET.

    Constructing a policy here does not approve it for Hockey. Product code
    must keep the default blocked policy until a separate decision records a
    version and cutoff from chronological out-of-sample evidence.
    """

    version: str
    minimum_edge: float


@dataclass(frozen=True, slots=True)
class OddsEdgeRecommendation:
    """One game recommendation. Stake is always null from the builder."""

    schema_version: int
    kind: str
    game_id: str | None
    home_team_id: str | None
    away_team_id: str | None
    book_source: str | None
    market: str | None
    odds_observed_at_utc: str | None
    scheduled_start_time_utc: str | None
    home_decimal_odds: float | None
    away_decimal_odds: float | None
    model_home_win_probability: float | None
    model_away_win_probability: float | None
    home_implied_probability: float | None
    away_implied_probability: float | None
    overround: float | None
    home_fair_market_probability: float | None
    away_fair_market_probability: float | None
    home_edge: float | None
    away_edge: float | None
    home_ev_per_unit: float | None
    away_ev_per_unit: float | None
    qualified_side: str | None
    recommendation_state: str
    threshold_policy_version: str
    stake_units: None
    vig_removal_method: str | None
    rationale: str
    fail_closed_reason: str | None
    source_provenance: dict[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "game_id": self.game_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "book_source": self.book_source,
            "market": self.market,
            "odds_observed_at_utc": self.odds_observed_at_utc,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "home_decimal_odds": self.home_decimal_odds,
            "away_decimal_odds": self.away_decimal_odds,
            "model_home_win_probability": self.model_home_win_probability,
            "model_away_win_probability": self.model_away_win_probability,
            "home_implied_probability": self.home_implied_probability,
            "away_implied_probability": self.away_implied_probability,
            "overround": self.overround,
            "home_fair_market_probability": self.home_fair_market_probability,
            "away_fair_market_probability": self.away_fair_market_probability,
            "home_edge": self.home_edge,
            "away_edge": self.away_edge,
            "home_ev_per_unit": self.home_ev_per_unit,
            "away_ev_per_unit": self.away_ev_per_unit,
            "qualified_side": self.qualified_side,
            "recommendation_state": self.recommendation_state,
            "threshold_policy_version": self.threshold_policy_version,
            "stake_units": self.stake_units,
            "vig_removal_method": self.vig_removal_method,
            "rationale": self.rationale,
            "fail_closed_reason": self.fail_closed_reason,
            "source_provenance": (
                None
                if self.source_provenance is None
                else dict(self.source_provenance)
            ),
        }


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _echo_decimal(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    odds = float(value)
    if not math.isfinite(odds) or odds <= 1.0:
        return None
    return odds


def _echo_probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probability = float(value)
    if not math.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
        return None
    return probability


def _parse_utc(value: object) -> datetime | None:
    text = _text(value)
    if text is None or text.casefold() in {"unknown", "none", "null", "closing"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.isoformat()


def _provenance(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    payload: dict[str, str] = {}
    for key in _PROVENANCE_REQUIRED:
        text = _text(value.get(key))
        if text is None:
            return None
        payload[key] = text
    retrieved = value.get("retrieved_at_utc")
    if retrieved is not None:
        text = _text(retrieved)
        if text is None:
            return None
        payload["retrieved_at_utc"] = text
    return payload


def _no_data(
    *,
    reason: str,
    rationale: str,
    game_id: str | None = None,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    book_source: str | None = None,
    market: str | None = None,
    odds_observed_at_utc: str | None = None,
    scheduled_start_time_utc: str | None = None,
    home_decimal_odds: float | None = None,
    away_decimal_odds: float | None = None,
    model_home_win_probability: float | None = None,
    model_away_win_probability: float | None = None,
    source_provenance: dict[str, str] | None = None,
) -> OddsEdgeRecommendation:
    record = OddsEdgeRecommendation(
        schema_version=RECOMMENDATION_SCHEMA_VERSION,
        kind=RECOMMENDATION_KIND,
        game_id=game_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        book_source=book_source,
        market=market,
        odds_observed_at_utc=odds_observed_at_utc,
        scheduled_start_time_utc=scheduled_start_time_utc,
        home_decimal_odds=home_decimal_odds,
        away_decimal_odds=away_decimal_odds,
        model_home_win_probability=model_home_win_probability,
        model_away_win_probability=model_away_win_probability,
        home_implied_probability=None,
        away_implied_probability=None,
        overround=None,
        home_fair_market_probability=None,
        away_fair_market_probability=None,
        home_edge=None,
        away_edge=None,
        home_ev_per_unit=None,
        away_ev_per_unit=None,
        qualified_side=None,
        recommendation_state="NO_DATA",
        threshold_policy_version=BLOCKED_THRESHOLD_POLICY_VERSION,
        stake_units=None,
        vig_removal_method=None,
        rationale=rationale,
        fail_closed_reason=reason,
        source_provenance=source_provenance,
    )
    validate_recommendation_payload(record.to_dict())
    return record


def _identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _has_extra_books(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    return True


def _has_aggregation_policy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _require_pass_bet_invariants(payload: Mapping[str, Any]) -> None:
    """Re-check priced-snapshot invariants on a PASS or BET payload.

    These checks do not assume the dict was produced by the builder. NO_DATA
    records are not passed here; they may echo a rejected market label.
    """

    game_id = _identifier(payload.get("game_id"))
    home_team_id = _identifier(payload.get("home_team_id"))
    away_team_id = _identifier(payload.get("away_team_id"))
    if (
        game_id is None
        or home_team_id is None
        or away_team_id is None
        or home_team_id == away_team_id
    ):
        raise RecommendationContractError(
            "PASS/BET requires a game id and two distinct team ids"
        )
    book = payload.get("book_source")
    if not isinstance(book, str) or not book.strip():
        raise RecommendationContractError("PASS/BET requires a single book source")
    if _has_extra_books(payload.get("additional_book_sources")) or _has_aggregation_policy(
        payload.get("aggregation_policy_version")
    ):
        raise RecommendationContractError(
            "PASS/BET is single-book and cannot include another book or an aggregation policy"
        )
    if payload.get("market") != SCHEMA_V1_MARKET:
        raise RecommendationContractError("schema v1 is moneyline-only")
    # No maximum quote age. That cutoff is unapproved and is not applied here.
    observed = _parse_utc(payload.get("odds_observed_at_utc"))
    scheduled = _parse_utc(payload.get("scheduled_start_time_utc"))
    if observed is None or scheduled is None or observed >= scheduled:
        raise RecommendationContractError(
            "PASS/BET requires a timezone-aware observation strictly before scheduled start"
        )
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise RecommendationContractError("source provenance is absent")
    for key in _PROVENANCE_REQUIRED:
        if not isinstance(provenance.get(key), str) or not str(provenance[key]).strip():
            raise RecommendationContractError(f"source provenance field {key} is absent")
    if "retrieved_at_utc" in provenance:
        retrieved = provenance.get("retrieved_at_utc")
        if not isinstance(retrieved, str) or not retrieved.strip():
            raise RecommendationContractError(
                "source provenance field retrieved_at_utc is absent"
            )


def validate_recommendation_payload(payload: Mapping[str, Any]) -> None:
    """Raise when a recommendation object breaks the versioned schema.

    PASS and BET payloads are checked on their own. A hand-built dict must
    still satisfy identity, single-book, moneyline, pregame timestamp, and
    provenance invariants. NO_DATA stays the fail-closed shape and may echo
    a rejected market label without derived prices.
    """

    if payload.get("schema_version") != RECOMMENDATION_SCHEMA_VERSION:
        raise RecommendationContractError("schema_version is not supported")
    if payload.get("kind") != RECOMMENDATION_KIND:
        raise RecommendationContractError("kind is not the odds-edge recommendation")
    state = payload.get("recommendation_state")
    if state not in RECOMMENDATION_STATES:
        raise RecommendationContractError("recommendation_state is not BET, PASS, or NO_DATA")
    policy = payload.get("threshold_policy_version")
    if not isinstance(policy, str) or not policy.strip():
        raise RecommendationContractError("threshold_policy_version is absent")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise RecommendationContractError("rationale is absent")
    if payload.get("stake_units") is not None:
        raise RecommendationContractError("stake_units must stay null in schema v1")
    qualified = payload.get("qualified_side")
    reason = payload.get("fail_closed_reason")
    if state == "BET":
        if qualified not in QUALIFIED_SIDES:
            raise RecommendationContractError("BET requires qualified_side home or away")
        if reason is not None:
            raise RecommendationContractError("BET cannot carry a fail-closed reason")
    else:
        if qualified is not None:
            raise RecommendationContractError("qualified_side must be null unless state is BET")
    if state == "NO_DATA":
        if not isinstance(reason, str) or not reason.strip():
            raise RecommendationContractError("NO_DATA requires a fail-closed reason")
        for field in (
            "home_implied_probability",
            "away_implied_probability",
            "overround",
            "home_fair_market_probability",
            "away_fair_market_probability",
            "home_edge",
            "away_edge",
            "home_ev_per_unit",
            "away_ev_per_unit",
            "vig_removal_method",
        ):
            if payload.get(field) is not None:
                raise RecommendationContractError(
                    f"NO_DATA must not invent derived market field {field}"
                )
        return
    if reason is not None:
        raise RecommendationContractError("fail_closed_reason must be null unless state is NO_DATA")
    _require_pass_bet_invariants(payload)
    if payload.get("vig_removal_method") != VIG_REMOVAL_METHOD:
        raise RecommendationContractError("vig removal method is not the canonical proportional method")
    for field in (
        "home_decimal_odds",
        "away_decimal_odds",
        "model_home_win_probability",
        "model_away_win_probability",
        "home_implied_probability",
        "away_implied_probability",
        "overround",
        "home_fair_market_probability",
        "away_fair_market_probability",
        "home_edge",
        "away_edge",
        "home_ev_per_unit",
        "away_ev_per_unit",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RecommendationContractError(f"{field} is missing")
        if not math.isfinite(float(value)):
            raise RecommendationContractError(f"{field} is not finite")


def build_odds_edge_recommendation(
    *,
    game_id: object,
    home_team_id: object,
    away_team_id: object,
    odds_game_id: object,
    odds_home_team_id: object,
    odds_away_team_id: object,
    book_source: object,
    odds_observed_at_utc: object,
    scheduled_start_time_utc: object,
    home_decimal_odds: object,
    away_decimal_odds: object,
    model_home_win_probability: object,
    model_away_win_probability: object,
    source_provenance: object,
    market: object = DEFAULT_MARKET,
    additional_book_sources: Sequence[object] | None = None,
    aggregation_policy_version: object = None,
    threshold_policy: EdgeThresholdPolicy | None = None,
) -> OddsEdgeRecommendation:
    """Build one recommendation, or a fail-closed ``NO_DATA`` record.

    Checks run in a fixed order so one reason is returned when several inputs
    are bad. Derived market fields stay null until every gate passes.
    """

    canonical_game_id = _text(game_id)
    canonical_home = _text(home_team_id)
    canonical_away = _text(away_team_id)
    if (
        canonical_game_id is None
        or canonical_home is None
        or canonical_away is None
        or canonical_home == canonical_away
    ):
        return _no_data(
            reason="absent_game_identity",
            rationale="Game identity is absent or the two teams are not distinct.",
        )

    odds_game = _text(odds_game_id)
    odds_home = _text(odds_home_team_id)
    odds_away = _text(odds_away_team_id)
    if (
        odds_game != canonical_game_id
        or odds_home != canonical_home
        or odds_away != canonical_away
    ):
        return _no_data(
            reason="identity_mismatch",
            rationale=(
                "Odds game or team identity does not match the canonical game. "
                "No side was inferred."
            ),
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
        )

    provenance = _provenance(source_provenance)
    if provenance is None:
        return _no_data(
            reason="absent_provenance",
            rationale="Price provenance is absent. source_name, asset_identity, and terms_note are required.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
        )

    extra_books = tuple(additional_book_sources or ())
    aggregation_text = _text(aggregation_policy_version)
    if extra_books or aggregation_text is not None:
        return _no_data(
            reason="multi_book_aggregation",
            rationale=(
                "Schema v1 does not aggregate books. A version string does not "
                "authorize averaging or choosing among materially different prices."
            ),
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            source_provenance=provenance,
        )

    book = _text(book_source)
    market_name = _text(market)
    if book is None:
        return _no_data(
            reason="absent_book_source",
            rationale="A single book or market source is required.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            source_provenance=provenance,
        )
    if market_name is None:
        return _no_data(
            reason="absent_market",
            rationale="Market name is required.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            source_provenance=provenance,
        )
    if market_name != SCHEMA_V1_MARKET:
        return _no_data(
            reason="unsupported_market",
            rationale=(
                "Schema v1 prices a two-sided moneyline only. "
                "Any other market label fails closed. The quote was not reinterpreted."
            ),
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            source_provenance=provenance,
        )

    observed = _parse_utc(odds_observed_at_utc)
    scheduled = _parse_utc(scheduled_start_time_utc)
    if observed is None:
        return _no_data(
            reason="absent_observation_timestamp",
            rationale=(
                "Odds observation timestamp is absent, unknown, or not timezone-aware. "
                "No staleness window is invented."
            ),
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            source_provenance=provenance,
        )
    if scheduled is None:
        return _no_data(
            reason="absent_scheduled_start",
            rationale="Scheduled start is absent, so pregame timing cannot be defended.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=_utc_text(observed),
            source_provenance=provenance,
        )
    observed_text = _utc_text(observed)
    scheduled_text = _utc_text(scheduled)
    if observed >= scheduled:
        return _no_data(
            reason="observation_not_before_puck_drop",
            rationale="Odds observation is not strictly before scheduled puck drop.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=observed_text,
            scheduled_start_time_utc=scheduled_text,
            source_provenance=provenance,
        )

    echoed_home_odds = _echo_decimal(home_decimal_odds)
    echoed_away_odds = _echo_decimal(away_decimal_odds)
    echoed_model_home = _echo_probability(model_home_win_probability)
    echoed_model_away = _echo_probability(model_away_win_probability)
    if home_decimal_odds is None or away_decimal_odds is None:
        return _no_data(
            reason="missing_side",
            rationale="One side's decimal price is missing. The other side was not invented.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=observed_text,
            scheduled_start_time_utc=scheduled_text,
            home_decimal_odds=echoed_home_odds,
            away_decimal_odds=echoed_away_odds,
            model_home_win_probability=echoed_model_home,
            model_away_win_probability=echoed_model_away,
            source_provenance=provenance,
        )
    if echoed_home_odds is None or echoed_away_odds is None:
        return _no_data(
            reason="malformed_odds",
            rationale="Both sides require finite decimal odds strictly greater than 1.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=observed_text,
            scheduled_start_time_utc=scheduled_text,
            home_decimal_odds=echoed_home_odds,
            away_decimal_odds=echoed_away_odds,
            model_home_win_probability=echoed_model_home,
            model_away_win_probability=echoed_model_away,
            source_provenance=provenance,
        )

    try:
        priced = price_two_sided_moneyline(
            echoed_home_odds,
            echoed_away_odds,
            model_home_win_probability,
            model_away_win_probability,
            probability_sum_tolerance=PROBABILITY_SUM_TOLERANCE,
        )
    except MarketMathError:
        return _no_data(
            reason="malformed_model_probability",
            rationale=(
                "Model probabilities must both be finite, strictly between 0 and 1, "
                "and sum to 1. They are not renormalized."
            ),
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=observed_text,
            scheduled_start_time_utc=scheduled_text,
            home_decimal_odds=echoed_home_odds,
            away_decimal_odds=echoed_away_odds,
            model_home_win_probability=echoed_model_home,
            model_away_win_probability=echoed_model_away,
            source_provenance=provenance,
        )

    common = dict(
        schema_version=RECOMMENDATION_SCHEMA_VERSION,
        kind=RECOMMENDATION_KIND,
        game_id=canonical_game_id,
        home_team_id=canonical_home,
        away_team_id=canonical_away,
        book_source=book,
        market=market_name,
        odds_observed_at_utc=observed_text,
        scheduled_start_time_utc=scheduled_text,
        home_decimal_odds=priced.home_decimal_odds,
        away_decimal_odds=priced.away_decimal_odds,
        model_home_win_probability=priced.model_home_win_probability,
        model_away_win_probability=priced.model_away_win_probability,
        home_implied_probability=priced.home_implied_probability,
        away_implied_probability=priced.away_implied_probability,
        overround=priced.overround,
        home_fair_market_probability=priced.home_fair_market_probability,
        away_fair_market_probability=priced.away_fair_market_probability,
        home_edge=priced.home_edge,
        away_edge=priced.away_edge,
        home_ev_per_unit=priced.home_ev_per_unit,
        away_ev_per_unit=priced.away_ev_per_unit,
        stake_units=None,
        vig_removal_method=priced.vig_removal_method,
        source_provenance=provenance,
    )

    if threshold_policy is None:
        record = OddsEdgeRecommendation(
            **common,
            qualified_side=None,
            recommendation_state="PASS",
            threshold_policy_version=BLOCKED_THRESHOLD_POLICY_VERSION,
            rationale=(
                "Market math is complete, but no approved edge-threshold policy is active. "
                "recommendation_state is PASS and qualified_side is null. "
                "This is not evidence that edge was below a studied cutoff."
            ),
            fail_closed_reason=None,
        )
        validate_recommendation_payload(record.to_dict())
        return record

    policy_version = _text(threshold_policy.version)
    minimum_edge = threshold_policy.minimum_edge
    if (
        policy_version is None
        or policy_version == BLOCKED_THRESHOLD_POLICY_VERSION
        or isinstance(minimum_edge, bool)
        or not isinstance(minimum_edge, (int, float))
        or not math.isfinite(float(minimum_edge))
        or float(minimum_edge) < 0.0
    ):
        return _no_data(
            reason="invalid_threshold_policy",
            rationale="The supplied threshold policy is not a usable explicit policy.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=observed_text,
            scheduled_start_time_utc=scheduled_text,
            home_decimal_odds=priced.home_decimal_odds,
            away_decimal_odds=priced.away_decimal_odds,
            model_home_win_probability=priced.model_home_win_probability,
            model_away_win_probability=priced.model_away_win_probability,
            source_provenance=provenance,
        )

    try:
        side = qualifying_side(
            priced.home_edge,
            priced.away_edge,
            float(minimum_edge),
        )
    except RecommendationContractError:
        return _no_data(
            reason="ambiguous_qualification",
            rationale="Both sides exceeded the supplied edge threshold, so no side was chosen.",
            game_id=canonical_game_id,
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            book_source=book,
            market=market_name,
            odds_observed_at_utc=observed_text,
            scheduled_start_time_utc=scheduled_text,
            home_decimal_odds=priced.home_decimal_odds,
            away_decimal_odds=priced.away_decimal_odds,
            model_home_win_probability=priced.model_home_win_probability,
            model_away_win_probability=priced.model_away_win_probability,
            source_provenance=provenance,
        )

    if side is None:
        record = OddsEdgeRecommendation(
            **common,
            qualified_side=None,
            recommendation_state="PASS",
            threshold_policy_version=policy_version,
            rationale=(
                f"Neither side's model-minus-fair edge is strictly above the "
                f"supplied minimum for policy {policy_version}. No stake is assigned."
            ),
            fail_closed_reason=None,
        )
    else:
        record = OddsEdgeRecommendation(
            **common,
            qualified_side=side,
            recommendation_state="BET",
            threshold_policy_version=policy_version,
            rationale=(
                f"{side} model-minus-fair edge is strictly above the supplied "
                f"minimum for policy {policy_version}. stake_units stays null."
            ),
            fail_closed_reason=None,
        )
    validate_recommendation_payload(record.to_dict())
    return record
