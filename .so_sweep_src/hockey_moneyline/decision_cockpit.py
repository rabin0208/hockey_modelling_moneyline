"""Presentation model for the Issue #85 daily Hockey decision cockpit.

This module joins an existing read-only protected prediction slate with a
market-snapshot reader and the merged Issue #78 recommendation contract. It
never scores a model, writes prediction/odds stores, chooses a canonical
threshold, contacts a provider, or places a wager.

Until Issue #78 has a historically validated policy, the canonical policy is
'blocked-no-approved-threshold'. A separately labelled temporary research
reference may apply Rabin's documented 'edge > 0' screen. That reference is
unoptimized, not historically validated, uses a flat 1-unit research baseline,
and never writes a stake into the recommendation schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.forward_odds_archive import (
    DEFAULT_STORE_RELATIVE_PATH as DEFAULT_FORWARD_ODDS_STORE_RELATIVE_PATH,
    decimal_odds_for_market_math,
    read_validated_snapshots,
)
from hockey_moneyline.recommendation_contract import (
    BLOCKED_THRESHOLD_POLICY_VERSION,
    EdgeThresholdPolicy,
    OddsEdgeRecommendation,
    build_odds_edge_recommendation,
)

TEMPORARY_RESEARCH_POLICY_VERSION = "research-rabin-edge-gt-zero-unvalidated-v1"
TEMPORARY_RESEARCH_POLICY_LABEL = (
    "Temporary research reference · edge > 0 · unoptimized / not historically validated"
)
CANONICAL_POLICY_LABEL = "Canonical policy blocked — no approved threshold"
FLAT_RESEARCH_BASELINE_UNITS = 1.0
MARKET_INPUT_FIXTURE_LABEL = "Synthetic/fixture market inputs"
MARKET_INPUT_ARCHIVE_LABEL = "Validated forward odds archive"
MARKET_INPUT_ABSENT_LABEL = "No market input reader connected"


class DecisionCockpitError(HockeyMoneylineError):
    """Cockpit presentation input is missing or malformed."""


@dataclass(frozen=True, slots=True)
class MarketSnapshotInput:
    """04 presentation seam for one already-identified two-sided moneyline.

    This is intentionally not an Issue #87 persistence schema. The reviewed
    #87 reader is adapted into this small read-only shape; archive storage,
    validation, and write semantics remain owned by 01.
    """

    game_id: str
    home_team_id: str
    away_team_id: str
    book_source: str
    odds_observed_at_utc: str
    scheduled_start_time_utc: str
    home_decimal_odds: float | None
    away_decimal_odds: float | None
    source_provenance: Mapping[str, str]


class MarketSnapshotReader(Protocol):
    """Read-only boundary consumed by the cockpit."""

    @property
    def input_label(self) -> str: ...

    def snapshots_for_games(
        self,
        game_ids: Sequence[str],
    ) -> Mapping[str, MarketSnapshotInput]: ...

    def no_data_for_game(self, game_id: str) -> tuple[str, str] | None: ...


class NullMarketSnapshotReader:
    """Reader used when no market source has been connected."""

    @property
    def input_label(self) -> str:
        return MARKET_INPUT_ABSENT_LABEL

    def snapshots_for_games(
        self,
        game_ids: Sequence[str],
    ) -> Mapping[str, MarketSnapshotInput]:
        return {}

    def no_data_for_game(self, game_id: str) -> tuple[str, str] | None:
        return (
            "market_snapshot_missing",
            "No two-sided pregame market snapshot is available for this game.",
        )


@dataclass(frozen=True, slots=True)
class FixtureMarketSnapshotReader:
    """Synthetic fixture reader for #85 UI/tests only.

    It is not a forward-odds store and has no write path. Contract validation
    still happens through build_odds_edge_recommendation.
    """

    snapshots: tuple[MarketSnapshotInput, ...]
    source_path: str

    @property
    def input_label(self) -> str:
        return f"{MARKET_INPUT_FIXTURE_LABEL} · {self.source_path}"

    @classmethod
    def from_path(cls, path: Path | str) -> "FixtureMarketSnapshotReader":
        source = Path(path)
        if not source.is_file():
            raise DecisionCockpitError(f"Market fixture is missing: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DecisionCockpitError(f"Market fixture is unreadable: {source}") from exc
        except json.JSONDecodeError as exc:
            raise DecisionCockpitError(f"Market fixture is not valid JSON: {source}") from exc
        if not isinstance(payload, Mapping):
            raise DecisionCockpitError("Market fixture must be a JSON object")
        if payload.get("schema_version") != 1 or payload.get("kind") != "issue85_market_fixture":
            raise DecisionCockpitError("Market fixture schema/version is not supported")
        rows = payload.get("snapshots")
        if not isinstance(rows, list):
            raise DecisionCockpitError("Market fixture snapshots must be a list")
        snapshots: list[MarketSnapshotInput] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise DecisionCockpitError(f"Market fixture row {index} must be an object")
            snapshot = _snapshot_from_mapping(row, index=index)
            if snapshot.game_id in seen:
                raise DecisionCockpitError(
                    f"Market fixture contains duplicate game_id {snapshot.game_id}"
                )
            seen.add(snapshot.game_id)
            snapshots.append(snapshot)
        return cls(tuple(snapshots), str(source))

    def snapshots_for_games(
        self,
        game_ids: Sequence[str],
    ) -> Mapping[str, MarketSnapshotInput]:
        wanted = set(game_ids)
        return {row.game_id: row for row in self.snapshots if row.game_id in wanted}

    def no_data_for_game(self, game_id: str) -> tuple[str, str] | None:
        return (
            "market_fixture_snapshot_missing",
            "The explicit synthetic fixture has no snapshot for this game.",
        )


class ForwardOddsMarketSnapshotReader:
    """Adapter over Issue #87's reviewed validated-reader surface.

    04 does not parse JSONL or reimplement archive validation. When multiple
    books exist for a game, a book must be selected explicitly; no sportsbook
    is chosen automatically. Within one selected book, the newest validated
    pregame observation is the current display snapshot.
    """

    def __init__(
        self,
        store_path: Path | str = DEFAULT_FORWARD_ODDS_STORE_RELATIVE_PATH,
        *,
        book_source: str | None = None,
    ) -> None:
        self.store_path = str(store_path)
        cleaned = book_source.strip() if isinstance(book_source, str) else None
        self.book_source = cleaned or None
        self._no_data: dict[str, tuple[str, str]] = {}

    @property
    def input_label(self) -> str:
        source = self.book_source or "auto only when one source exists"
        return f"{MARKET_INPUT_ARCHIVE_LABEL} · {source} · {self.store_path}"

    def snapshots_for_games(
        self,
        game_ids: Sequence[str],
    ) -> Mapping[str, MarketSnapshotInput]:
        self._no_data = {}
        result: dict[str, MarketSnapshotInput] = {}
        for game_id in game_ids:
            rows = read_validated_snapshots(
                self.store_path,
                nhl_game_id=game_id,
            )
            if self.book_source is None:
                books = sorted({row.book for row in rows})
                if len(books) > 1:
                    self._no_data[game_id] = (
                        "market_source_selection_required",
                        "Multiple validated sportsbook/source snapshots exist. "
                        "Choose one source explicitly; the cockpit will not choose "
                        "a sportsbook automatically.",
                    )
                    continue
                selected_rows = rows
            else:
                selected_rows = tuple(
                    row for row in rows if row.book == self.book_source
                )
                if not selected_rows and rows:
                    self._no_data[game_id] = (
                        "market_book_not_available",
                        f"No validated snapshot for source {self.book_source!r} "
                        "exists for this game.",
                    )
                    continue

            if not selected_rows:
                self._no_data[game_id] = (
                    "market_snapshot_missing",
                    "No validated two-sided pregame forward-odds snapshot is "
                    "available for this game.",
                )
                continue

            selected = selected_rows[-1]
            home_decimal, away_decimal = decimal_odds_for_market_math(selected)
            result[game_id] = MarketSnapshotInput(
                game_id=selected.nhl_game_id,
                home_team_id=selected.home_team,
                away_team_id=selected.away_team,
                book_source=selected.book,
                odds_observed_at_utc=selected.observation_utc,
                scheduled_start_time_utc=selected.scheduled_puck_drop_utc,
                home_decimal_odds=home_decimal,
                away_decimal_odds=away_decimal,
                source_provenance={
                    "source_name": selected.book,
                    "asset_identity": (
                        "forward_odds_snapshot_sha256:"
                        f"{selected.snapshot_sha256}"
                    ),
                    "terms_note": (
                        f"capture_method={selected.capture_method}; "
                        f"provenance={selected.provenance}"
                    ),
                },
            )
        return result

    def no_data_for_game(self, game_id: str) -> tuple[str, str] | None:
        return self._no_data.get(game_id)


def available_forward_odds_books(
    *,
    store_path: Path | str,
    game_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return validated book/source names represented on the requested slate."""

    wanted = set(game_ids)
    rows = read_validated_snapshots(store_path)
    return tuple(sorted({row.book for row in rows if row.nhl_game_id in wanted}))


@dataclass(frozen=True, slots=True)
class SideDecision:
    side: str
    team: str
    model_probability: float | None
    decimal_odds: float | None
    implied_probability: float | None
    fair_market_probability: float | None
    edge: float | None
    ev_per_unit: float | None


@dataclass(frozen=True, slots=True)
class CockpitGame:
    nhl_game_id: str
    away_team: str
    home_team: str
    scheduled_local_label: str
    scheduled_utc_label: str
    protected_prediction_status: str
    receipt_ready: bool
    decision_state: str
    decision_basis: str
    decision_side: str | None
    decision_team: str | None
    research_baseline_units: float | None
    canonical_policy_version: str
    canonical_policy_label: str
    temporary_policy_version: str
    temporary_policy_label: str
    book_source: str | None
    odds_observed_at_utc: str | None
    source_provenance: dict[str, str] | None
    home: SideDecision
    away: SideDecision
    overround: float | None
    no_data_reason: str | None
    recommendation_rationale: str
    audit: Any | None

    @property
    def is_priced(self) -> bool:
        return (
            self.home.fair_market_probability is not None
            and self.away.fair_market_probability is not None
        )


@dataclass(frozen=True, slots=True)
class DecisionCockpitView:
    selected_date: Any
    market_input_label: str
    games: tuple[CockpitGame, ...]
    canonical_policy_version: str
    canonical_policy_label: str
    temporary_policy_label: str
    flat_research_baseline_units: float
    disclaimer: str

    @property
    def priced_count(self) -> int:
        return sum(game.is_priced for game in self.games)

    @property
    def research_bet_count(self) -> int:
        return sum(game.decision_state == "BET" for game in self.games)

    @property
    def no_data_count(self) -> int:
        return sum(game.decision_state == "NO_DATA" for game in self.games)


_RESEARCH_POLICY = EdgeThresholdPolicy(
    version=TEMPORARY_RESEARCH_POLICY_VERSION,
    minimum_edge=0.0,
)


def build_decision_cockpit_view(
    *,
    dashboard_view: Any,
    market_reader: MarketSnapshotReader,
) -> DecisionCockpitView:
    """Join a protected dashboard view to read-only market inputs."""

    game_ids = tuple(str(game.nhl_game_id) for game in dashboard_view.games)
    snapshots = market_reader.snapshots_for_games(game_ids)
    unknown = set(snapshots) - set(game_ids)
    if unknown:
        raise DecisionCockpitError(
            "Market reader returned games outside the requested slate: "
            + ", ".join(sorted(unknown))
        )
    games = tuple(
        _build_game(
            game,
            snapshots.get(str(game.nhl_game_id)),
            market_reader=market_reader,
        )
        for game in dashboard_view.games
    )
    return DecisionCockpitView(
        selected_date=dashboard_view.selected_date,
        market_input_label=market_reader.input_label,
        games=games,
        canonical_policy_version=BLOCKED_THRESHOLD_POLICY_VERSION,
        canonical_policy_label=CANONICAL_POLICY_LABEL,
        temporary_policy_label=TEMPORARY_RESEARCH_POLICY_LABEL,
        flat_research_baseline_units=FLAT_RESEARCH_BASELINE_UNITS,
        disclaimer=(
            "Research-only decision surface. Temporary BET/PASS states use an "
            "unoptimized edge > 0 reference with a flat 1-unit baseline; they are "
            "not a validated canonical betting policy and never trigger wager placement."
        ),
    )


def _build_game(
    game: Any,
    snapshot: MarketSnapshotInput | None,
    *,
    market_reader: MarketSnapshotReader,
) -> CockpitGame:
    home_model = _probability_or_none(getattr(game, "home_model_win_probability", None))
    away_model = _probability_or_none(getattr(game, "away_model_win_probability", None))
    base = dict(
        nhl_game_id=str(game.nhl_game_id),
        away_team=str(game.away_team),
        home_team=str(game.home_team),
        scheduled_local_label=str(game.scheduled_local_label),
        scheduled_utc_label=str(game.scheduled_utc_label),
        protected_prediction_status=str(game.prediction_status),
        receipt_ready=(str(game.prediction_status) == "RECEIPT READY"),
        canonical_policy_version=BLOCKED_THRESHOLD_POLICY_VERSION,
        canonical_policy_label=CANONICAL_POLICY_LABEL,
        temporary_policy_version=TEMPORARY_RESEARCH_POLICY_VERSION,
        temporary_policy_label=TEMPORARY_RESEARCH_POLICY_LABEL,
        audit=getattr(game, "audit", None),
    )

    if home_model is None or away_model is None:
        return _no_data_game(
            **base,
            home_model=home_model,
            away_model=away_model,
            snapshot=snapshot,
            reason="protected_model_not_available",
            rationale=(
                "Protected model probability is not available; market values are not "
                "used to invent one."
            ),
        )
    if snapshot is None:
        missing = market_reader.no_data_for_game(str(game.nhl_game_id)) or (
            "market_snapshot_missing",
            "No two-sided pregame market snapshot is available for this game.",
        )
        return _no_data_game(
            **base,
            home_model=home_model,
            away_model=away_model,
            snapshot=None,
            reason=missing[0],
            rationale=missing[1],
        )
    if (
        snapshot.game_id != str(game.nhl_game_id)
        or snapshot.home_team_id != str(game.home_team)
        or snapshot.away_team_id != str(game.away_team)
    ):
        return _no_data_game(
            **base,
            home_model=home_model,
            away_model=away_model,
            snapshot=snapshot,
            reason="market_identity_mismatch",
            rationale=(
                "Market game/home/away identity does not match the protected slate; "
                "no side was inferred."
            ),
        )
    protected_start = getattr(game, "scheduled_start_time_utc", None)
    if not _same_utc_instant(snapshot.scheduled_start_time_utc, protected_start):
        return _no_data_game(
            **base,
            home_model=home_model,
            away_model=away_model,
            snapshot=snapshot,
            reason="market_schedule_mismatch",
            rationale=(
                "Market scheduled puck drop does not match the protected slate. "
                "The quote was not attached to this game."
            ),
        )

    canonical = _recommendation(snapshot, home_model, away_model, threshold_policy=None)
    if canonical.recommendation_state == "NO_DATA":
        return _game_from_recommendation(
            **base,
            canonical=canonical,
            research=None,
            decision_basis="canonical_fail_closed",
        )

    research = _recommendation(
        snapshot,
        home_model,
        away_model,
        threshold_policy=_RESEARCH_POLICY,
    )
    if research.recommendation_state == "NO_DATA":
        return _game_from_recommendation(
            **base,
            canonical=canonical,
            research=research,
            decision_basis="temporary_research_reference_fail_closed",
        )
    return _game_from_recommendation(
        **base,
        canonical=canonical,
        research=research,
        decision_basis="temporary_research_reference",
    )


def _recommendation(
    snapshot: MarketSnapshotInput,
    home_model: float,
    away_model: float,
    *,
    threshold_policy: EdgeThresholdPolicy | None,
) -> OddsEdgeRecommendation:
    return build_odds_edge_recommendation(
        game_id=snapshot.game_id,
        home_team_id=snapshot.home_team_id,
        away_team_id=snapshot.away_team_id,
        odds_game_id=snapshot.game_id,
        odds_home_team_id=snapshot.home_team_id,
        odds_away_team_id=snapshot.away_team_id,
        book_source=snapshot.book_source,
        odds_observed_at_utc=snapshot.odds_observed_at_utc,
        scheduled_start_time_utc=snapshot.scheduled_start_time_utc,
        home_decimal_odds=snapshot.home_decimal_odds,
        away_decimal_odds=snapshot.away_decimal_odds,
        model_home_win_probability=home_model,
        model_away_win_probability=away_model,
        source_provenance=dict(snapshot.source_provenance),
        threshold_policy=threshold_policy,
    )


def _game_from_recommendation(
    *,
    nhl_game_id: str,
    away_team: str,
    home_team: str,
    scheduled_local_label: str,
    scheduled_utc_label: str,
    protected_prediction_status: str,
    receipt_ready: bool,
    canonical_policy_version: str,
    canonical_policy_label: str,
    temporary_policy_version: str,
    temporary_policy_label: str,
    audit: Any | None,
    canonical: OddsEdgeRecommendation,
    research: OddsEdgeRecommendation | None,
    decision_basis: str,
) -> CockpitGame:
    display = research if research is not None else canonical
    state = display.recommendation_state
    qualified = display.qualified_side
    decision_team = (
        home_team
        if qualified == "home"
        else away_team
        if qualified == "away"
        else None
    )
    baseline = (
        FLAT_RESEARCH_BASELINE_UNITS
        if state == "BET" and research is not None
        else None
    )
    return CockpitGame(
        nhl_game_id=nhl_game_id,
        away_team=away_team,
        home_team=home_team,
        scheduled_local_label=scheduled_local_label,
        scheduled_utc_label=scheduled_utc_label,
        protected_prediction_status=protected_prediction_status,
        receipt_ready=receipt_ready,
        decision_state=state,
        decision_basis=decision_basis,
        decision_side=qualified,
        decision_team=decision_team,
        research_baseline_units=baseline,
        canonical_policy_version=canonical_policy_version,
        canonical_policy_label=canonical_policy_label,
        temporary_policy_version=temporary_policy_version,
        temporary_policy_label=temporary_policy_label,
        book_source=display.book_source,
        odds_observed_at_utc=display.odds_observed_at_utc,
        source_provenance=(
            None
            if display.source_provenance is None
            else dict(display.source_provenance)
        ),
        home=_side_from_recommendation("home", home_team, display),
        away=_side_from_recommendation("away", away_team, display),
        overround=display.overround,
        no_data_reason=display.fail_closed_reason,
        recommendation_rationale=display.rationale,
        audit=audit,
    )


def _no_data_game(
    *,
    nhl_game_id: str,
    away_team: str,
    home_team: str,
    scheduled_local_label: str,
    scheduled_utc_label: str,
    protected_prediction_status: str,
    receipt_ready: bool,
    canonical_policy_version: str,
    canonical_policy_label: str,
    temporary_policy_version: str,
    temporary_policy_label: str,
    audit: Any | None,
    home_model: float | None,
    away_model: float | None,
    snapshot: MarketSnapshotInput | None,
    reason: str,
    rationale: str,
) -> CockpitGame:
    return CockpitGame(
        nhl_game_id=nhl_game_id,
        away_team=away_team,
        home_team=home_team,
        scheduled_local_label=scheduled_local_label,
        scheduled_utc_label=scheduled_utc_label,
        protected_prediction_status=protected_prediction_status,
        receipt_ready=receipt_ready,
        decision_state="NO_DATA",
        decision_basis="fail_closed",
        decision_side=None,
        decision_team=None,
        research_baseline_units=None,
        canonical_policy_version=canonical_policy_version,
        canonical_policy_label=canonical_policy_label,
        temporary_policy_version=temporary_policy_version,
        temporary_policy_label=temporary_policy_label,
        book_source=None if snapshot is None else snapshot.book_source,
        odds_observed_at_utc=(
            None if snapshot is None else snapshot.odds_observed_at_utc
        ),
        source_provenance=(
            None if snapshot is None else dict(snapshot.source_provenance)
        ),
        home=SideDecision(
            side="home",
            team=home_team,
            model_probability=home_model,
            decimal_odds=None if snapshot is None else snapshot.home_decimal_odds,
            implied_probability=None,
            fair_market_probability=None,
            edge=None,
            ev_per_unit=None,
        ),
        away=SideDecision(
            side="away",
            team=away_team,
            model_probability=away_model,
            decimal_odds=None if snapshot is None else snapshot.away_decimal_odds,
            implied_probability=None,
            fair_market_probability=None,
            edge=None,
            ev_per_unit=None,
        ),
        overround=None,
        no_data_reason=reason,
        recommendation_rationale=rationale,
        audit=audit,
    )


def _side_from_recommendation(
    side: str,
    team: str,
    record: OddsEdgeRecommendation,
) -> SideDecision:
    return SideDecision(
        side=side,
        team=team,
        model_probability=getattr(record, f"model_{side}_win_probability"),
        decimal_odds=getattr(record, f"{side}_decimal_odds"),
        implied_probability=getattr(record, f"{side}_implied_probability"),
        fair_market_probability=getattr(
            record,
            f"{side}_fair_market_probability",
        ),
        edge=getattr(record, f"{side}_edge"),
        ev_per_unit=getattr(record, f"{side}_ev_per_unit"),
    )


def _snapshot_from_mapping(
    row: Mapping[str, Any],
    *,
    index: int,
) -> MarketSnapshotInput:
    required_text = (
        "game_id",
        "home_team_id",
        "away_team_id",
        "book_source",
        "odds_observed_at_utc",
        "scheduled_start_time_utc",
    )
    values: dict[str, str] = {}
    for key in required_text:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DecisionCockpitError(
                f"Market fixture row {index} field {key} is absent"
            )
        values[key] = value.strip()
    provenance = row.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise DecisionCockpitError(
            f"Market fixture row {index} source_provenance is absent"
        )
    clean_provenance: dict[str, str] = {}
    for key, value in provenance.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise DecisionCockpitError(
                f"Market fixture row {index} source_provenance must contain text fields"
            )
        clean_provenance[key] = value
    return MarketSnapshotInput(
        **values,
        home_decimal_odds=_fixture_decimal(
            row.get("home_decimal_odds"),
            field="home_decimal_odds",
            index=index,
        ),
        away_decimal_odds=_fixture_decimal(
            row.get("away_decimal_odds"),
            field="away_decimal_odds",
            index=index,
        ),
        source_provenance=clean_provenance,
    )


def _fixture_decimal(
    value: object,
    *,
    field: str,
    index: int,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionCockpitError(
            f"Market fixture row {index} field {field} must be numeric or null"
        )
    return float(value)


def _same_utc_instant(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        left_dt = datetime.fromisoformat(left.replace("Z", "+00:00"))
        right_dt = datetime.fromisoformat(right.replace("Z", "+00:00"))
    except ValueError:
        return False
    if left_dt.tzinfo is None or right_dt.tzinfo is None:
        return False
    return left_dt.astimezone(timezone.utc) == right_dt.astimezone(timezone.utc)


def _probability_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probability = float(value)
    if probability <= 0.0 or probability >= 1.0:
        return None
    return probability


def format_percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value * 100:+.1f} pp"
    return f"{value * 100:.1f}%"


def format_decimal(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def format_ev(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}u"


def render_cockpit_card_text(card: CockpitGame) -> str:
    """Accessible primary-view contract used by tests and non-visual review."""

    lines = [
        f"{card.away_team} at {card.home_team}",
        card.scheduled_local_label,
        f"Decision {card.decision_state}",
        f"Canonical policy {card.canonical_policy_version}",
        card.temporary_policy_label,
        f"Protected prediction {card.protected_prediction_status}",
    ]
    if card.decision_state == "NO_DATA":
        lines.extend(
            (
                f"NO DATA reason {card.no_data_reason}",
                card.recommendation_rationale,
            )
        )
        return "\n".join(lines)
    lines.extend(
        (
            f"Market source {card.book_source}",
            f"Observed {card.odds_observed_at_utc}",
            f"Home {card.home.team}: model {format_percent(card.home.model_probability)}, "
            f"odds {format_decimal(card.home.decimal_odds)}, fair "
            f"{format_percent(card.home.fair_market_probability)}, edge "
            f"{format_percent(card.home.edge, signed=True)}, EV "
            f"{format_ev(card.home.ev_per_unit)}",
            f"Away {card.away.team}: model {format_percent(card.away.model_probability)}, "
            f"odds {format_decimal(card.away.decimal_odds)}, fair "
            f"{format_percent(card.away.fair_market_probability)}, edge "
            f"{format_percent(card.away.edge, signed=True)}, EV "
            f"{format_ev(card.away.ev_per_unit)}",
        )
    )
    if card.decision_state == "BET":
        lines.append(
            f"Temporary research BET {card.decision_team} · flat "
            f"{card.research_baseline_units:.1f} unit baseline"
        )
    return "\n".join(lines)
