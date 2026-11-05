"""Issue #111 private Morning Board.

Composes the approved hosted seams for one manual Prepare Today action:

- #91 / #93 ``run_hosted_refresh`` for NHL + MoneyPuck and frozen in-memory scoring
- #98 / #107 PlayNow acquisition, bounded to at most one provider request
- #87 validated forward-odds archive as the only odds contract
- #85 / #78 market math and recommendation semantics

Page load and hydration read the last successful snapshot and the local #87
archive only. They do not call NHL, MoneyPuck, or PlayNow.

The canonical recommendation policy stays ``blocked-no-approved-threshold``.
The 10 percentage-point shortlist is a temporary unvalidated research rule.
A side is eligible only when model-minus-no-vig fair edge is at least +10.0
percentage points and offered-price EV/1u is strictly positive. That rule is
not a canonical BET/PASS threshold and does not unblock Issue #78. When a
future #78 policy version replaces the blocked version, the same adapter
boundary renders that policy and stops applying the 10 pp rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from hockey_moneyline.decision_cockpit import (
    CANONICAL_POLICY_LABEL,
    CockpitGame,
    DecisionCockpitView,
    ForwardOddsMarketSnapshotReader,
    SideDecision,
    build_decision_cockpit_view,
)
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.forward_odds_archive import read_validated_snapshots
from hockey_moneyline.hosted_cockpit import (
    DEFAULT_HOSTED_DISPLAY_TIMEZONE,
    CURRENT_CALCULATION_LABEL,
    PROTECTED_EVIDENCE_LABEL,
    HostedCockpitView,
    run_hosted_refresh,
    score_by_game_id,
)
from hockey_moneyline.hosted_odds import (
    build_playnow_market_view,
    hosted_playnow_schedule,
    refresh_playnow_odds,
    refresh_playnow_odds_for_schedule,
)
from hockey_moneyline.prediction_record_store import PredictionRecordStore
from hockey_moneyline.prospective_artifact import EXPECTED_FITTED_ARTIFACT_SHA256
from hockey_moneyline.recommendation_contract import BLOCKED_THRESHOLD_POLICY_VERSION
from hockey_moneyline.the_odds_api_playnow import PLAYNOW_BOOKMAKER_KEY

MORNING_BOARD_SCHEMA_VERSION = 1
MORNING_BOARD_KIND = "morning_board_last_successful_snapshot"
MORNING_ATTEMPT_KIND = "morning_board_last_attempt"
PLAYNOW_REUSE_MAX_AGE = timedelta(minutes=15)
TEN_PP_MINIMUM_EDGE = 0.10
TEN_PP_RULE_LABEL = "10 PP RESEARCH RULE · UNVALIDATED"
SHORTLIST_HEADING = "RESEARCH SHORTLIST — UNVALIDATED"
SHORTLIST_EMPTY_LABEL = "No 10 pp research recommendations"
RESEARCH_RANKING_LABEL = (
    "RESEARCH RANKING · positive EV, then edge magnitude · "
    "not a validated betting recommendation"
)
MARKET_CACHED = "CACHED"
MARKET_LIVE_REFRESHED = "LIVE REFRESHED"
HYDRATION_CURRENT = "CURRENT"
HYDRATION_STALE = "STALE"
HYDRATION_EMPTY = "EMPTY"
DECISION_NO_DATA = "NO DATA"
DECISION_NO_THRESHOLD = "NO CANONICAL BET THRESHOLD YET"
DECISION_BET = "BET"
DECISION_PASS = "PASS"
BLOCKED_DISPLAY_STATES = frozenset({DECISION_NO_DATA, DECISION_NO_THRESHOLD})

DEFAULT_MORNING_BOARD_DIR = Path("data/derived/morning_board")
DEFAULT_BOARD_PATH = DEFAULT_MORNING_BOARD_DIR / "last_successful_board.json"
DEFAULT_ATTEMPT_PATH = DEFAULT_MORNING_BOARD_DIR / "last_attempt.json"
DEFAULT_ODDS_STORE_PATH = DEFAULT_MORNING_BOARD_DIR / "playnow_forward_odds.jsonl"

_SCHEDULE_FIELDS = (
    "nhl_game_id",
    "home_team",
    "away_team",
    "scheduled_puck_drop_utc",
)


@dataclass(frozen=True, slots=True)
class MorningShortlistEntry:
    nhl_game_id: str
    matchup: str
    side: str
    team: str
    decimal_odds: float
    model_probability: float
    fair_market_probability: float
    edge: float
    ev_per_unit: float
    rule_label: str
    unvalidated: bool


@dataclass(frozen=True, slots=True)
class MorningBoardGame:
    game: CockpitGame
    decision_label: str
    rank_index: int
    protected_evidence: bool
    audit: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class MorningBoardView:
    requested_slate_date: date
    snapshot_slate_date: date | None
    hydration_status: str
    status_banner: str
    source_observed_at_utc: str | None
    source_observed_local: str | None
    playnow_observed_at_utc: str | None
    playnow_observed_local: str | None
    playnow_age_label: str | None
    market_mode: str | None
    playnow_request_count: int | None
    fut_count: int
    priced_count: int
    nhl_status: str
    moneypuck_status: str
    model_status: str
    model_artifact_sha256: str
    canonical_policy_version: str
    canonical_policy_label: str
    research_ranking_label: str
    shortlist_heading: str
    shortlist_rule_label: str
    shortlist_empty_label: str
    using_temporary_rule: bool
    shortlist: tuple[MorningShortlistEntry, ...]
    games: tuple[MorningBoardGame, ...]
    calculation_label: str
    protected_evidence_label: str
    protected_read_error: str | None
    blocker_code: str | None
    blocker_message: str | None

    @property
    def is_current(self) -> bool:
        return self.hydration_status == HYDRATION_CURRENT


@dataclass(frozen=True, slots=True)
class MorningPreparationResult:
    ok: bool
    blocker_code: str | None
    blocker_message: str | None
    playnow_request_count: int
    market_mode: str | None
    snapshot_replaced: bool
    view: MorningBoardView


@dataclass(frozen=True, slots=True)
class _PersistedGame:
    nhl_game_id: str
    away_team: str
    home_team: str
    scheduled_local_label: str
    scheduled_utc_label: str
    scheduled_start_time_utc: str
    prediction_status: str
    home_model_win_probability: float | None
    away_model_win_probability: float | None
    audit: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _Snapshot:
    slate_date: date
    prepared_at_utc: str
    source_observed_at_utc: str
    display_timezone: str
    nhl_status: str
    moneypuck_status: str
    model_status: str
    model_artifact_sha256: str
    refresh_readiness: str
    scoring_state: str
    calculation_label: str
    protected_evidence_label: str
    market_mode: str | None
    playnow_request_count: int
    odds_store_path: str
    fut_game_ids: tuple[str, ...]
    schedule: tuple[dict[str, str], ...]
    games: tuple[_PersistedGame, ...]


class _RecommendationAdapter:
    policy_version: str
    policy_label: str
    temporary: bool
    rule_label: str
    heading: str
    empty_label: str

    def display_state(self, game: CockpitGame) -> str:
        raise NotImplementedError

    def shortlist(self, games: Sequence[CockpitGame]) -> tuple[MorningShortlistEntry, ...]:
        raise NotImplementedError


class BlockedTenPpResearchAdapter(_RecommendationAdapter):
    """Temporary morning rule used only while #78 has no approved threshold."""

    policy_version = BLOCKED_THRESHOLD_POLICY_VERSION
    policy_label = CANONICAL_POLICY_LABEL
    temporary = True
    rule_label = TEN_PP_RULE_LABEL
    heading = SHORTLIST_HEADING
    empty_label = SHORTLIST_EMPTY_LABEL

    def display_state(self, game: CockpitGame) -> str:
        if not game.is_priced:
            return DECISION_NO_DATA
        return DECISION_NO_THRESHOLD

    def shortlist(self, games: Sequence[CockpitGame]) -> tuple[MorningShortlistEntry, ...]:
        entries = tuple(
            _shortlist_entry(game, side, rule_label=self.rule_label, unvalidated=True)
            for game in games
            for side in (_ten_pp_side(game),)
            if side is not None
        )
        return tuple(sorted(entries, key=_temporary_shortlist_sort_key))


class ApprovedCanonicalAdapter(_RecommendationAdapter):
    """Consumes a future non-blocked #78 policy without a second board design."""

    temporary = False
    heading = "RECOMMENDED"
    empty_label = "No recommendations under the approved policy"

    def __init__(self, policy_version: str, policy_label: str) -> None:
        self.policy_version = policy_version
        self.policy_label = policy_label
        self.rule_label = policy_label

    def display_state(self, game: CockpitGame) -> str:
        if not game.is_priced or game.decision_state == "NO_DATA":
            return DECISION_NO_DATA
        if game.decision_state == "BET":
            return DECISION_BET
        if game.decision_state == "PASS":
            return DECISION_PASS
        return DECISION_NO_DATA

    def shortlist(self, games: Sequence[CockpitGame]) -> tuple[MorningShortlistEntry, ...]:
        entries: list[MorningShortlistEntry] = []
        for game in games:
            if game.decision_state != "BET" or game.decision_side not in {"home", "away"}:
                continue
            side = game.home if game.decision_side == "home" else game.away
            if (
                side.decimal_odds is None
                or side.model_probability is None
                or side.fair_market_probability is None
                or side.edge is None
                or side.ev_per_unit is None
            ):
                continue
            entries.append(
                _shortlist_entry(
                    game,
                    side,
                    rule_label=self.rule_label,
                    unvalidated=False,
                )
            )
        return tuple(
            sorted(entries, key=lambda entry: (-entry.edge, -entry.ev_per_unit, entry.nhl_game_id))
        )


def adapter_for(policy_version: str, policy_label: str) -> _RecommendationAdapter:
    if policy_version == BLOCKED_THRESHOLD_POLICY_VERSION:
        return BlockedTenPpResearchAdapter()
    return ApprovedCanonicalAdapter(policy_version, policy_label)


def prepare_today(
    *,
    slate_date: date | str,
    observed_at_utc: datetime | str,
    board_path: Path | str = DEFAULT_BOARD_PATH,
    attempt_path: Path | str = DEFAULT_ATTEMPT_PATH,
    odds_store_path: Path | str = DEFAULT_ODDS_STORE_PATH,
    protected_store_path: Path | str | None = None,
    display_timezone: str = DEFAULT_HOSTED_DISPLAY_TIMEZONE,
    refresh_kwargs: Mapping[str, Any] | None = None,
    frozen_history: tuple[Any, Any, Any] | None = None,
    fetcher: Callable[..., Any] | None = None,
) -> MorningPreparationResult:
    """Refresh sources, reuse or fetch PlayNow once, and persist a same-day board."""

    slate = _parse_slate_date(slate_date)
    now = _parse_aware(observed_at_utc, label="observed_at_utc")
    board = Path(board_path)
    attempt = Path(attempt_path)
    odds_path = Path(odds_store_path)
    common = dict(
        board_path=board,
        attempt_path=attempt,
        protected_store_path=protected_store_path,
        display_timezone=display_timezone,
        now=now,
        slate_date=slate,
    )
    try:
        run = run_hosted_refresh(
            slate_date=slate,
            observed_at_utc=now,
            refresh_kwargs=refresh_kwargs,
            frozen_history=frozen_history,
        )
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        return _fail(
            code="source_refresh_failed",
            message=str(exc),
            playnow_request_count=0,
            **common,
        )
    if not run.is_successful:
        message = run.view.blocker_message or "Source refresh is blocked."
        if run.view.blocker_detail:
            message = f"{message}\n{run.view.blocker_detail}"
        return _fail(
            code=run.view.blocker_code or "source_refresh_blocked",
            message=message,
            playnow_request_count=0,
            **common,
        )

    schedule = tuple(dict(row) for row in hosted_playnow_schedule(run))
    try:
        reusable = _reusable_observation(
            odds_path,
            tuple(row["nhl_game_id"] for row in schedule),
            slate_date=slate,
            now=now,
            display_timezone=display_timezone,
        )
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        return _fail(
            code="playnow_archive_unreadable",
            message=str(exc),
            playnow_request_count=0,
            **common,
        )

    market_mode = MARKET_CACHED
    request_count = 0
    if reusable is None:
        market_mode = MARKET_LIVE_REFRESHED
        try:
            odds_kwargs: dict[str, Any] = {}
            if fetcher is not None:
                odds_kwargs["fetcher"] = fetcher
            odds = refresh_playnow_odds(run, store_path=odds_path, **odds_kwargs)
        except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
            attempted = 1 if schedule else 0
            return _fail(
                code="playnow_refresh_failed",
                message=str(exc),
                playnow_request_count=attempted,
                **common,
            )
        request_count = int(odds.request_count)
        if request_count > 1:
            return _fail(
                code="playnow_request_unbounded",
                message=f"PlayNow refresh reported {request_count} provider requests.",
                playnow_request_count=request_count,
                **common,
            )
    try:
        hosted_view = build_playnow_market_view(run, store_path=odds_path)
        snapshot = _snapshot_from_hosted_view(
            hosted_view,
            schedule=schedule,
            market_mode=market_mode,
            playnow_request_count=request_count,
            odds_store_path=odds_path,
            prepared_at_utc=now,
            display_timezone=display_timezone,
        )
        _write_json(board, _snapshot_payload(snapshot))
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        return _fail(
            code="morning_board_compose_failed",
            message=str(exc),
            playnow_request_count=request_count,
            **common,
        )
    _write_attempt(
        attempt,
        ok=True,
        slate_date=slate,
        attempted_at_utc=now,
        blocker_code=None,
        blocker_message=None,
        playnow_request_count=request_count,
        market_mode=market_mode,
        snapshot_preserved=False,
    )
    view = _view_from_snapshot(
        snapshot,
        requested_slate_date=slate,
        now=now,
        protected_store_path=protected_store_path,
        blocker_code=None,
        blocker_message=None,
    )
    return MorningPreparationResult(
        ok=True,
        blocker_code=None,
        blocker_message=None,
        playnow_request_count=request_count,
        market_mode=market_mode,
        snapshot_replaced=True,
        view=view,
    )


def force_live_odds_refresh(
    *,
    slate_date: date | str,
    observed_at_utc: datetime | str,
    board_path: Path | str = DEFAULT_BOARD_PATH,
    attempt_path: Path | str = DEFAULT_ATTEMPT_PATH,
    odds_store_path: Path | str = DEFAULT_ODDS_STORE_PATH,
    protected_store_path: Path | str | None = None,
    display_timezone: str = DEFAULT_HOSTED_DISPLAY_TIMEZONE,
    fetcher: Callable[..., Any] | None = None,
) -> MorningPreparationResult:
    """One explicit PlayNow pull. Does not refresh NHL or MoneyPuck."""

    slate = _parse_slate_date(slate_date)
    now = _parse_aware(observed_at_utc, label="observed_at_utc")
    board = Path(board_path)
    attempt = Path(attempt_path)
    common = dict(
        board_path=board,
        attempt_path=attempt,
        protected_store_path=protected_store_path,
        display_timezone=display_timezone,
        now=now,
        slate_date=slate,
    )
    try:
        snapshot = _read_snapshot(board)
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        return _fail(
            code="morning_board_unreadable",
            message=str(exc),
            playnow_request_count=0,
            **common,
        )
    if snapshot is None or snapshot.slate_date != slate:
        return _fail(
            code="force_refresh_requires_current_board",
            message=(
                "Prepare Today before forcing a live odds refresh. "
                "Prior-day state is not a current board."
            ),
            playnow_request_count=0,
            **common,
        )
    odds_path = Path(snapshot.odds_store_path or odds_store_path)
    try:
        odds_kwargs: dict[str, Any] = {}
        if fetcher is not None:
            odds_kwargs["fetcher"] = fetcher
        odds = refresh_playnow_odds_for_schedule(
            snapshot.schedule,
            store_path=odds_path,
            **odds_kwargs,
        )
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        attempted = 1 if snapshot.schedule else 0
        return _fail(
            code="playnow_refresh_failed",
            message=str(exc),
            playnow_request_count=attempted,
            **common,
        )
    request_count = int(odds.request_count)
    if request_count > 1:
        return _fail(
            code="playnow_request_unbounded",
            message=f"PlayNow refresh reported {request_count} provider requests.",
            playnow_request_count=request_count,
            **common,
        )
    try:
        updated = _replace_market_mode(
            snapshot,
            market_mode=MARKET_LIVE_REFRESHED,
            playnow_request_count=request_count,
            prepared_at_utc=now,
            odds_store_path=odds_path,
        )
        view = _view_from_snapshot(
            updated,
            requested_slate_date=slate,
            now=now,
            protected_store_path=protected_store_path,
            blocker_code=None,
            blocker_message=None,
        )
        _write_json(board, _snapshot_payload(updated))
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        return _fail(
            code="morning_board_compose_failed",
            message=str(exc),
            playnow_request_count=request_count,
            **common,
        )
    _write_attempt(
        attempt,
        ok=True,
        slate_date=slate,
        attempted_at_utc=now,
        blocker_code=None,
        blocker_message=None,
        playnow_request_count=request_count,
        market_mode=MARKET_LIVE_REFRESHED,
        snapshot_preserved=False,
    )
    return MorningPreparationResult(
        ok=True,
        blocker_code=None,
        blocker_message=None,
        playnow_request_count=request_count,
        market_mode=MARKET_LIVE_REFRESHED,
        snapshot_replaced=True,
        view=view,
    )


def load_morning_board(
    *,
    slate_date: date | str,
    observed_at_utc: datetime | str,
    board_path: Path | str = DEFAULT_BOARD_PATH,
    attempt_path: Path | str = DEFAULT_ATTEMPT_PATH,
    protected_store_path: Path | str | None = None,
    display_timezone: str = DEFAULT_HOSTED_DISPLAY_TIMEZONE,
) -> MorningBoardView:
    """Hydrate the last snapshot. This function has no provider seam."""

    slate = _parse_slate_date(slate_date)
    now = _parse_aware(observed_at_utc, label="observed_at_utc")
    snapshot, snapshot_error = _read_snapshot_safely(Path(board_path))
    attempt_blocker_code, attempt_blocker_message = _attempt_blocker(Path(attempt_path))
    blocker_code = snapshot_error[0] if snapshot_error is not None else attempt_blocker_code
    blocker_message = snapshot_error[1] if snapshot_error is not None else attempt_blocker_message
    if snapshot_error is not None and attempt_blocker_message:
        blocker_message = f"{snapshot_error[1]}\n{attempt_blocker_message}"
    if snapshot is None:
        return _empty_view(
            requested_slate_date=slate,
            now=now,
            display_timezone=display_timezone,
            blocker_code=blocker_code,
            blocker_message=blocker_message,
        )
    return _view_from_snapshot(
        snapshot,
        requested_slate_date=slate,
        now=now,
        protected_store_path=protected_store_path,
        blocker_code=blocker_code,
        blocker_message=blocker_message,
        display_timezone_override=display_timezone,
    )


def render_morning_board_text(view: MorningBoardView) -> str:
    """Stable text contract for tests and non-visual review."""

    lines = [
        f"Requested slate {view.requested_slate_date.isoformat()}",
        f"Hydration {view.hydration_status}",
        view.status_banner,
        f"Source refresh {view.source_observed_at_utc or 'unavailable'}",
        f"Source local {view.source_observed_local or 'unavailable'}",
        f"PlayNow observation {view.playnow_observed_at_utc or 'unavailable'}",
        f"PlayNow local {view.playnow_observed_local or 'unavailable'}",
        f"PlayNow age {view.playnow_age_label or 'unavailable'}",
        f"Market mode {view.market_mode or 'unavailable'}",
        f"FUT count {view.fut_count}",
        f"Priced count {view.priced_count}",
        f"Model {view.model_status}",
        f"Frozen artifact {view.model_artifact_sha256}",
        f"Canonical policy {view.canonical_policy_version}",
        view.canonical_policy_label,
        "SHOW RECOMMENDED",
        view.shortlist_heading,
        view.shortlist_rule_label,
    ]
    if view.blocker_message:
        lines.append(f"BLOCKER {view.blocker_message}")
    if not view.shortlist:
        lines.append(view.shortlist_empty_label)
    for entry in view.shortlist:
        lines.extend(
            (
                f"Shortlist {entry.matchup}",
                f"Side {entry.side} {entry.team}",
                f"Decimal {entry.decimal_odds:.2f}",
                f"Model {entry.model_probability:.6f}",
                f"Fair {entry.fair_market_probability:.6f}",
                f"Edge {entry.edge * 100:.1f} pp",
                f"EV {entry.ev_per_unit:+.3f}u",
                entry.rule_label,
            )
        )
    lines.append(view.research_ranking_label)
    for row in view.games:
        game = row.game
        lines.extend(
            (
                f"Rank {row.rank_index}",
                f"{game.away_team} at {game.home_team}",
                game.scheduled_local_label,
                f"Decision {row.decision_label}",
                f"Protected evidence {'yes' if row.protected_evidence else 'no'}",
                f"Away model {game.away.model_probability}",
                f"Home model {game.home.model_probability}",
                f"Away decimal {game.away.decimal_odds}",
                f"Home decimal {game.home.decimal_odds}",
                f"Away fair {game.away.fair_market_probability}",
                f"Home fair {game.home.fair_market_probability}",
                f"Away edge {game.away.edge}",
                f"Home edge {game.home.edge}",
                f"Away ev {game.away.ev_per_unit}",
                f"Home ev {game.home.ev_per_unit}",
                f"Observed {game.odds_observed_at_utc}",
            )
        )
    lines.append("AUDIT")
    for row in view.games:
        if row.audit:
            lines.append(f"Audit {row.game.nhl_game_id}")
    return "\n".join(lines)


def _fail(
    *,
    code: str,
    message: str,
    playnow_request_count: int,
    board_path: Path,
    attempt_path: Path,
    protected_store_path: Path | str | None,
    display_timezone: str,
    now: datetime,
    slate_date: date,
) -> MorningPreparationResult:
    _write_attempt(
        attempt_path,
        ok=False,
        slate_date=slate_date,
        attempted_at_utc=now,
        blocker_code=code,
        blocker_message=message,
        playnow_request_count=playnow_request_count,
        market_mode=None,
        snapshot_preserved=True,
    )
    view = load_morning_board(
        slate_date=slate_date,
        observed_at_utc=now,
        board_path=board_path,
        attempt_path=attempt_path,
        protected_store_path=protected_store_path,
        display_timezone=display_timezone,
    )
    return MorningPreparationResult(
        ok=False,
        blocker_code=code,
        blocker_message=message,
        playnow_request_count=playnow_request_count,
        market_mode=view.market_mode,
        snapshot_replaced=False,
        view=view,
    )


def _reusable_observation(
    store_path: Path,
    game_ids: Sequence[str],
    *,
    slate_date: date,
    now: datetime,
    display_timezone: str,
) -> datetime | None:
    if not game_ids or not store_path.is_file():
        return None
    zone = ZoneInfo(display_timezone)
    wanted = set(game_ids)
    observations = tuple(
        _parse_aware(row.observation_utc, label="observation_utc")
        for row in read_validated_snapshots(store_path)
        if row.book == PLAYNOW_BOOKMAKER_KEY and row.nhl_game_id in wanted
    )
    same_day = tuple(
        observed
        for observed in observations
        if observed.astimezone(zone).date() == slate_date and observed <= now
    )
    if not same_day:
        return None
    latest = max(same_day)
    if now - latest >= PLAYNOW_REUSE_MAX_AGE:
        return None
    return latest


def _ten_pp_eligible(side: SideDecision) -> bool:
    """Temporary Show Recommended gate: +10 pp edge and strictly positive EV/1u."""

    return (
        side.edge is not None
        and side.edge >= TEN_PP_MINIMUM_EDGE
        and side.ev_per_unit is not None
        and side.ev_per_unit > 0.0
    )


def _ten_pp_side(game: CockpitGame) -> SideDecision | None:
    if not game.is_priced:
        return None
    qualified = tuple(side for side in (game.home, game.away) if _ten_pp_eligible(side))
    if len(qualified) != 1:
        return None
    side = qualified[0]
    if (
        side.decimal_odds is None
        or side.model_probability is None
        or side.fair_market_probability is None
    ):
        return None
    return side


def _temporary_shortlist_sort_key(
    entry: MorningShortlistEntry,
) -> tuple[float, float, str, str]:
    return (-entry.ev_per_unit, -entry.edge, entry.nhl_game_id, entry.team)


def _shortlist_entry(
    game: CockpitGame,
    side: SideDecision,
    *,
    rule_label: str,
    unvalidated: bool,
) -> MorningShortlistEntry:
    assert side.decimal_odds is not None
    assert side.model_probability is not None
    assert side.fair_market_probability is not None
    assert side.edge is not None
    assert side.ev_per_unit is not None
    return MorningShortlistEntry(
        nhl_game_id=game.nhl_game_id,
        matchup=f"{game.away_team} @ {game.home_team}",
        side=side.side,
        team=side.team,
        decimal_odds=side.decimal_odds,
        model_probability=side.model_probability,
        fair_market_probability=side.fair_market_probability,
        edge=side.edge,
        ev_per_unit=side.ev_per_unit,
        rule_label=rule_label,
        unvalidated=unvalidated,
    )


def research_rank_key(game: CockpitGame) -> tuple[Any, ...]:
    start = game.scheduled_utc_label
    game_id = game.nhl_game_id
    if not game.is_priced:
        return (2, 0.0, 0.0, start, game_id)
    positive = max(
        (
            side.ev_per_unit
            for side in (game.home, game.away)
            if side.ev_per_unit is not None and side.ev_per_unit > 0.0
        ),
        default=None,
    )
    magnitude = max(
        (
            abs(side.edge)
            for side in (game.home, game.away)
            if side.edge is not None
        ),
        default=0.0,
    )
    if positive is not None:
        return (0, -positive, -magnitude, start, game_id)
    return (1, 0.0, -magnitude, start, game_id)


def rank_games(
    games: Sequence[CockpitGame],
    adapter: _RecommendationAdapter,
) -> tuple[CockpitGame, ...]:
    def _key(game: CockpitGame) -> tuple[Any, ...]:
        base = research_rank_key(game)
        if adapter.temporary:
            return base
        canonical_first = 0 if game.decision_state == "BET" else 1
        return (canonical_first, *base)

    return tuple(sorted(games, key=_key))


def _snapshot_from_hosted_view(
    view: HostedCockpitView,
    *,
    schedule: tuple[dict[str, str], ...],
    market_mode: str,
    playnow_request_count: int,
    odds_store_path: Path,
    prepared_at_utc: datetime,
    display_timezone: str,
) -> _Snapshot:
    scores = score_by_game_id(view)
    games = tuple(
        _persisted_game(game, scores.get(game.nhl_game_id))
        for game in view.decision_view.games
    )
    fut_ids = tuple(row["nhl_game_id"] for row in schedule)
    return _Snapshot(
        slate_date=view.selected_date,
        prepared_at_utc=prepared_at_utc.astimezone(timezone.utc).isoformat(),
        source_observed_at_utc=view.observed_at_utc,
        display_timezone=display_timezone,
        nhl_status=view.nhl_status,
        moneypuck_status=view.moneypuck_status,
        model_status=view.model_status,
        model_artifact_sha256=EXPECTED_FITTED_ARTIFACT_SHA256,
        refresh_readiness=view.refresh_readiness,
        scoring_state=view.scoring_state,
        calculation_label=view.calculation_label,
        protected_evidence_label=view.protected_evidence_label,
        market_mode=market_mode,
        playnow_request_count=playnow_request_count,
        odds_store_path=str(odds_store_path),
        fut_game_ids=fut_ids,
        schedule=tuple(_schedule_row(row) for row in schedule),
        games=games,
    )


def _persisted_game(game: CockpitGame, score: Any) -> _PersistedGame:
    scheduled = ""
    audit = None
    if score is not None:
        scheduled = str(score.scheduled_start_time_utc)
        audit = score.to_dict()
    return _PersistedGame(
        nhl_game_id=game.nhl_game_id,
        away_team=game.away_team,
        home_team=game.home_team,
        scheduled_local_label=game.scheduled_local_label,
        scheduled_utc_label=game.scheduled_utc_label,
        scheduled_start_time_utc=scheduled,
        prediction_status=game.protected_prediction_status,
        home_model_win_probability=game.home.model_probability,
        away_model_win_probability=game.away.model_probability,
        audit=audit,
    )


def _schedule_row(row: Mapping[str, str]) -> dict[str, str]:
    missing = [field for field in _SCHEDULE_FIELDS if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"PlayNow schedule row is missing {missing}")
    return {field: str(row[field]) for field in _SCHEDULE_FIELDS}


def _replace_market_mode(
    snapshot: _Snapshot,
    *,
    market_mode: str,
    playnow_request_count: int,
    prepared_at_utc: datetime,
    odds_store_path: Path,
) -> _Snapshot:
    return _Snapshot(
        slate_date=snapshot.slate_date,
        prepared_at_utc=prepared_at_utc.astimezone(timezone.utc).isoformat(),
        source_observed_at_utc=snapshot.source_observed_at_utc,
        display_timezone=snapshot.display_timezone,
        nhl_status=snapshot.nhl_status,
        moneypuck_status=snapshot.moneypuck_status,
        model_status=snapshot.model_status,
        model_artifact_sha256=snapshot.model_artifact_sha256,
        refresh_readiness=snapshot.refresh_readiness,
        scoring_state=snapshot.scoring_state,
        calculation_label=snapshot.calculation_label,
        protected_evidence_label=snapshot.protected_evidence_label,
        market_mode=market_mode,
        playnow_request_count=playnow_request_count,
        odds_store_path=str(odds_store_path),
        fut_game_ids=snapshot.fut_game_ids,
        schedule=snapshot.schedule,
        games=snapshot.games,
    )


def _view_from_snapshot(
    snapshot: _Snapshot,
    *,
    requested_slate_date: date,
    now: datetime,
    protected_store_path: Path | str | None,
    blocker_code: str | None,
    blocker_message: str | None,
    display_timezone_override: str | None = None,
) -> MorningBoardView:
    zone_name = display_timezone_override or snapshot.display_timezone
    dashboard_games = snapshot.games
    dashboard = _Dashboard(
        selected_date=snapshot.slate_date,
        games=dashboard_games,
    )
    reader = ForwardOddsMarketSnapshotReader(
        store_path=snapshot.odds_store_path,
        book_source=PLAYNOW_BOOKMAKER_KEY,
    )
    decision = build_decision_cockpit_view(
        dashboard_view=dashboard,
        market_reader=reader,
    )
    protected_ids, protected_error = _protected_game_ids(protected_store_path)
    return _assemble_view(
        decision=decision,
        requested_slate_date=requested_slate_date,
        snapshot=snapshot,
        now=now,
        display_timezone=zone_name,
        protected_ids=protected_ids,
        protected_error=protected_error,
        blocker_code=blocker_code,
        blocker_message=blocker_message,
        audits={game.nhl_game_id: game.audit for game in snapshot.games},
    )


def _assemble_view(
    *,
    decision: DecisionCockpitView,
    requested_slate_date: date,
    snapshot: _Snapshot,
    now: datetime,
    display_timezone: str,
    protected_ids: frozenset[str],
    protected_error: str | None,
    blocker_code: str | None,
    blocker_message: str | None,
    audits: Mapping[str, dict[str, Any] | None],
) -> MorningBoardView:
    adapter = adapter_for(
        decision.canonical_policy_version,
        decision.canonical_policy_label,
    )
    ranked = rank_games(decision.games, adapter)
    fut_ids = set(snapshot.fut_game_ids)
    priced_ids = {game.nhl_game_id for game in ranked if game.is_priced and game.nhl_game_id in fut_ids}
    rows = tuple(
        MorningBoardGame(
            game=game,
            decision_label=adapter.display_state(game),
            rank_index=index + 1,
            protected_evidence=game.nhl_game_id in protected_ids,
            audit=audits.get(game.nhl_game_id),
        )
        for index, game in enumerate(ranked)
    )
    observation = _latest_observation(ranked)
    source_dt = _parse_aware(snapshot.source_observed_at_utc, label="source_observed_at_utc")
    current = snapshot.slate_date == requested_slate_date
    if current:
        status = HYDRATION_CURRENT
        banner = f"Today's board · {snapshot.slate_date.isoformat()}"
    else:
        status = HYDRATION_STALE
        banner = (
            f"STALE · last successful slate {snapshot.slate_date.isoformat()} "
            f"is not {requested_slate_date.isoformat()}. "
            "Prepare Today before treating this as today's board."
        )
    return MorningBoardView(
        requested_slate_date=requested_slate_date,
        snapshot_slate_date=snapshot.slate_date,
        hydration_status=status,
        status_banner=banner,
        source_observed_at_utc=source_dt.astimezone(timezone.utc).isoformat(),
        source_observed_local=_format_local(source_dt, display_timezone),
        playnow_observed_at_utc=(
            None if observation is None else observation.astimezone(timezone.utc).isoformat()
        ),
        playnow_observed_local=(
            None if observation is None else _format_local(observation, display_timezone)
        ),
        playnow_age_label=_format_age(now, observation),
        market_mode=snapshot.market_mode,
        playnow_request_count=snapshot.playnow_request_count,
        fut_count=len(snapshot.fut_game_ids),
        priced_count=len(priced_ids),
        nhl_status=snapshot.nhl_status,
        moneypuck_status=snapshot.moneypuck_status,
        model_status=snapshot.model_status,
        model_artifact_sha256=snapshot.model_artifact_sha256,
        canonical_policy_version=decision.canonical_policy_version,
        canonical_policy_label=decision.canonical_policy_label,
        research_ranking_label=RESEARCH_RANKING_LABEL,
        shortlist_heading=adapter.heading,
        shortlist_rule_label=adapter.rule_label,
        shortlist_empty_label=adapter.empty_label,
        using_temporary_rule=adapter.temporary,
        shortlist=adapter.shortlist(ranked),
        games=rows,
        calculation_label=snapshot.calculation_label,
        protected_evidence_label=snapshot.protected_evidence_label,
        protected_read_error=protected_error,
        blocker_code=blocker_code,
        blocker_message=blocker_message,
    )


def _empty_view(
    *,
    requested_slate_date: date,
    now: datetime,
    display_timezone: str,
    blocker_code: str | None,
    blocker_message: str | None,
) -> MorningBoardView:
    del now, display_timezone
    adapter = BlockedTenPpResearchAdapter()
    banner = "No morning board yet. Prepare Today to score the slate and read PlayNow."
    if blocker_message and blocker_code != "morning_board_unreadable":
        banner = "No successful morning board is available."
    return MorningBoardView(
        requested_slate_date=requested_slate_date,
        snapshot_slate_date=None,
        hydration_status=HYDRATION_EMPTY,
        status_banner=banner,
        source_observed_at_utc=None,
        source_observed_local=None,
        playnow_observed_at_utc=None,
        playnow_observed_local=None,
        playnow_age_label=None,
        market_mode=None,
        playnow_request_count=None,
        fut_count=0,
        priced_count=0,
        nhl_status="Not refreshed",
        moneypuck_status="Not refreshed",
        model_status=(
            f"FROZEN · SHA-256 {EXPECTED_FITTED_ARTIFACT_SHA256[:12]}… · "
            "not scored until Prepare Today"
        ),
        model_artifact_sha256=EXPECTED_FITTED_ARTIFACT_SHA256,
        canonical_policy_version=BLOCKED_THRESHOLD_POLICY_VERSION,
        canonical_policy_label=CANONICAL_POLICY_LABEL,
        research_ranking_label=RESEARCH_RANKING_LABEL,
        shortlist_heading=adapter.heading,
        shortlist_rule_label=adapter.rule_label,
        shortlist_empty_label=adapter.empty_label,
        using_temporary_rule=True,
        shortlist=(),
        games=(),
        calculation_label=CURRENT_CALCULATION_LABEL,
        protected_evidence_label=PROTECTED_EVIDENCE_LABEL,
        protected_read_error=None,
        blocker_code=blocker_code,
        blocker_message=blocker_message,
    )


@dataclass(frozen=True, slots=True)
class _Dashboard:
    selected_date: date
    games: tuple[_PersistedGame, ...]


def _protected_game_ids(
    path: Path | str | None,
) -> tuple[frozenset[str], str | None]:
    if path is None:
        return frozenset(), None
    store_path = Path(path)
    if not store_path.is_file():
        return frozenset(), None
    try:
        envelopes = tuple(PredictionRecordStore(store_path).iter_envelopes())
    except (HockeyMoneylineError, OSError, ValueError, TypeError) as exc:
        return frozenset(), f"Protected prediction records could not be read: {exc}"
    return frozenset(str(envelope["record"]["game_id"]) for envelope in envelopes), None


def _latest_observation(games: Sequence[CockpitGame]) -> datetime | None:
    stamps = tuple(
        _parse_aware(game.odds_observed_at_utc, label="odds_observed_at_utc")
        for game in games
        if game.is_priced and isinstance(game.odds_observed_at_utc, str)
    )
    if not stamps:
        return None
    return max(stamps)


def _format_local(value: datetime, display_timezone: str) -> str:
    local = value.astimezone(ZoneInfo(display_timezone))
    hour = local.strftime("%I").lstrip("0") or "0"
    return f"{local.strftime('%Y-%m-%d')} {hour}:{local.strftime('%M %p %Z')}"


def _format_age(now: datetime, observed: datetime | None) -> str | None:
    if observed is None:
        return None
    seconds = int((now - observed).total_seconds())
    if seconds < 0:
        return "clock skew"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _snapshot_payload(snapshot: _Snapshot) -> dict[str, Any]:
    return {
        "schema_version": MORNING_BOARD_SCHEMA_VERSION,
        "kind": MORNING_BOARD_KIND,
        "slate_date": snapshot.slate_date.isoformat(),
        "prepared_at_utc": snapshot.prepared_at_utc,
        "source_observed_at_utc": snapshot.source_observed_at_utc,
        "display_timezone": snapshot.display_timezone,
        "nhl_status": snapshot.nhl_status,
        "moneypuck_status": snapshot.moneypuck_status,
        "model_status": snapshot.model_status,
        "model_artifact_sha256": snapshot.model_artifact_sha256,
        "refresh_readiness": snapshot.refresh_readiness,
        "scoring_state": snapshot.scoring_state,
        "calculation_label": snapshot.calculation_label,
        "protected_evidence_label": snapshot.protected_evidence_label,
        "market_mode": snapshot.market_mode,
        "playnow_request_count": snapshot.playnow_request_count,
        "odds_store_path": snapshot.odds_store_path,
        "fut_game_ids": list(snapshot.fut_game_ids),
        "schedule": [dict(row) for row in snapshot.schedule],
        "games": [_game_payload(game) for game in snapshot.games],
        "odds_contract": "issue87_forward_odds_snapshot",
        "canonical_recommendation_threshold": BLOCKED_THRESHOLD_POLICY_VERSION,
    }


def _game_payload(game: _PersistedGame) -> dict[str, Any]:
    return {
        "nhl_game_id": game.nhl_game_id,
        "away_team": game.away_team,
        "home_team": game.home_team,
        "scheduled_local_label": game.scheduled_local_label,
        "scheduled_utc_label": game.scheduled_utc_label,
        "scheduled_start_time_utc": game.scheduled_start_time_utc,
        "prediction_status": game.prediction_status,
        "home_model_win_probability": game.home_model_win_probability,
        "away_model_win_probability": game.away_model_win_probability,
        "audit": game.audit,
    }


def _read_snapshot(path: Path) -> _Snapshot | None:
    snapshot, error = _read_snapshot_safely(path)
    if error is not None:
        raise ValueError(error[1])
    return snapshot


def _read_snapshot_safely(
    path: Path,
) -> tuple[_Snapshot | None, tuple[str, str] | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Morning board snapshot is not a JSON object")
        if payload.get("schema_version") != MORNING_BOARD_SCHEMA_VERSION:
            raise ValueError("Morning board snapshot schema is not supported")
        if payload.get("kind") != MORNING_BOARD_KIND:
            raise ValueError("Morning board snapshot kind is not supported")
        if payload.get("odds_contract") != "issue87_forward_odds_snapshot":
            raise ValueError("Morning board snapshot is not tied to the Issue #87 odds contract")
        return _snapshot_from_payload(payload), None
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return None, ("morning_board_unreadable", str(exc))


def _snapshot_from_payload(payload: Mapping[str, Any]) -> _Snapshot:
    games_payload = payload.get("games")
    schedule_payload = payload.get("schedule")
    fut_ids = payload.get("fut_game_ids")
    if not isinstance(games_payload, list) or not isinstance(schedule_payload, list):
        raise ValueError("Morning board snapshot games/schedule are malformed")
    if not isinstance(fut_ids, list):
        raise ValueError("Morning board snapshot FUT ids are malformed")
    return _Snapshot(
        slate_date=_parse_slate_date(payload.get("slate_date")),
        prepared_at_utc=_text(payload.get("prepared_at_utc"), label="prepared_at_utc"),
        source_observed_at_utc=_text(
            payload.get("source_observed_at_utc"),
            label="source_observed_at_utc",
        ),
        display_timezone=_text(payload.get("display_timezone"), label="display_timezone"),
        nhl_status=_text(payload.get("nhl_status"), label="nhl_status"),
        moneypuck_status=_text(payload.get("moneypuck_status"), label="moneypuck_status"),
        model_status=_text(payload.get("model_status"), label="model_status"),
        model_artifact_sha256=_text(
            payload.get("model_artifact_sha256"),
            label="model_artifact_sha256",
        ),
        refresh_readiness=_text(payload.get("refresh_readiness"), label="refresh_readiness"),
        scoring_state=_text(payload.get("scoring_state"), label="scoring_state"),
        calculation_label=_text(payload.get("calculation_label"), label="calculation_label"),
        protected_evidence_label=_text(
            payload.get("protected_evidence_label"),
            label="protected_evidence_label",
        ),
        market_mode=_optional_text(payload.get("market_mode")),
        playnow_request_count=_nonnegative_int(payload.get("playnow_request_count")),
        odds_store_path=_text(payload.get("odds_store_path"), label="odds_store_path"),
        fut_game_ids=tuple(_text(game_id, label="fut_game_id") for game_id in fut_ids),
        schedule=tuple(_schedule_row(row) for row in schedule_payload if isinstance(row, Mapping)),
        games=tuple(_persisted_from_payload(row) for row in games_payload),
    )


def _persisted_from_payload(payload: Any) -> _PersistedGame:
    if not isinstance(payload, Mapping):
        raise ValueError("Morning board game row is malformed")
    audit = payload.get("audit")
    if audit is not None and not isinstance(audit, dict):
        raise ValueError("Morning board game audit is malformed")
    return _PersistedGame(
        nhl_game_id=_text(payload.get("nhl_game_id"), label="nhl_game_id"),
        away_team=_text(payload.get("away_team"), label="away_team"),
        home_team=_text(payload.get("home_team"), label="home_team"),
        scheduled_local_label=_text(
            payload.get("scheduled_local_label"),
            label="scheduled_local_label",
        ),
        scheduled_utc_label=_text(payload.get("scheduled_utc_label"), label="scheduled_utc_label"),
        scheduled_start_time_utc=_text(
            payload.get("scheduled_start_time_utc"),
            label="scheduled_start_time_utc",
        ),
        prediction_status=_text(payload.get("prediction_status"), label="prediction_status"),
        home_model_win_probability=_optional_probability(payload.get("home_model_win_probability")),
        away_model_win_probability=_optional_probability(payload.get("away_model_win_probability")),
        audit=audit,
    )


def _attempt_blocker(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "morning_attempt_unreadable", "The last Prepare Today attempt log is unreadable."
    if not isinstance(payload, dict) or payload.get("kind") != MORNING_ATTEMPT_KIND:
        return "morning_attempt_unreadable", "The last Prepare Today attempt log is not a morning attempt."
    if payload.get("ok") is True:
        return None, None
    code = payload.get("blocker_code")
    message = payload.get("blocker_message")
    return (
        str(code) if isinstance(code, str) and code else "morning_attempt_failed",
        str(message) if isinstance(message, str) and message else "Prepare Today failed.",
    )


def _write_attempt(
    path: Path,
    *,
    ok: bool,
    slate_date: date,
    attempted_at_utc: datetime,
    blocker_code: str | None,
    blocker_message: str | None,
    playnow_request_count: int,
    market_mode: str | None,
    snapshot_preserved: bool,
) -> None:
    _write_json(
        path,
        {
            "schema_version": MORNING_BOARD_SCHEMA_VERSION,
            "kind": MORNING_ATTEMPT_KIND,
            "ok": ok,
            "slate_date": slate_date.isoformat(),
            "attempted_at_utc": attempted_at_utc.astimezone(timezone.utc).isoformat(),
            "blocker_code": blocker_code,
            "blocker_message": blocker_message,
            "playnow_request_count": playnow_request_count,
            "market_mode": market_mode,
            "snapshot_preserved": snapshot_preserved,
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_slate_date(value: object) -> date:
    if isinstance(value, datetime):
        raise ValueError("slate date must be a date, not a timestamp")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("slate date is not ISO-8601") from exc
    raise ValueError("slate date is missing")


def _parse_aware(value: object, *, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{label} is not a valid timestamp") from exc
    else:
        raise ValueError(f"{label} is missing")
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional text field is malformed")
    return value


def _optional_probability(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("model probability is malformed")
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise ValueError("model probability is outside (0, 1)")
    return probability


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("playnow_request_count is malformed")
    if value < 0:
        raise ValueError("playnow_request_count cannot be negative")
    return value
