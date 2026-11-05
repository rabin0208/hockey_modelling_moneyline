"""Issue #91 bounded on-demand NHL + MoneyPuck refresh for the hosted cockpit.

This is the 01-owned readiness primitive. 03 scores only from
``scoring_inputs()``. 04 displays ``blocker.message`` and does not parse NHL
or MoneyPuck responses.

The refresh reuses the approved NHL Web API schedule contract and MoneyPuck's
published ``all_teams.csv``. It applies Issue #54/#59 current-season prior
validation. It does not fit, score, tune, read the historical cache, or write
protected prediction records.

Each call fetches both sources into memory under one caller-supplied
observation timestamp. A failed or incomplete fetch is a blocked bundle.
Stale files are not substituted.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hockey_moneyline.current_season_prior import (
    CURRENT_SEASON,
    CURRENT_SEASON_REGULAR_WINDOW_START,
    MONEYPUCK_CURRENT_SEASON_COLUMNS,
    assemble_current_season_nhl_coverage,
    build_current_season_prior_state,
    current_season_coverage_start,
    parse_current_season_moneypuck_rows,
    required_coverage_dates,
)
from hockey_moneyline.errors import HostedRefreshError, HockeyMoneylineError
from hockey_moneyline.nhl_api import (
    FINAL_GAME_STATES,
    NHL_WEB_BASE_URL,
    NHLScheduleGame,
    NHLScheduleSnapshot,
    SCHEDULE_DATE_PATH,
)
from hockey_moneyline.normalize import TARGET_ERA_TEAM_CODES, canonical_moneypuck_team_code
from hockey_moneyline.provenance import (
    ACCESS_METHOD,
    MONEYPUCK_SITUATION,
    SOURCE_FILE_URL,
    SOURCE_ID,
    SOURCE_NAME,
    SOURCE_PAGE,
    USER_AGENT,
)
from hockey_moneyline.unscored_target import UnscoredTarget

HOSTED_REFRESH_SCHEMA_VERSION = 1
HOSTED_REFRESH_KIND = "hosted_refresh_bundle"
READY = "READY"
BLOCKED = "BLOCKED"
FUT_GAME_STATE = "FUT"
HOSTED_TARGET_SOURCE = "hosted_on_demand_refresh"
MAX_MONEYPUCK_RESPONSE_BYTES = 256 * 1024 * 1024
ON_DEMAND_REFRESH_CONSUMPTION_OUTCOME = (
    "READY FOR 03/04 ON-DEMAND REFRESH CONSUMPTION"
)
PUBLICATION_LAG_MESSAGE = (
    "MoneyPuck has not yet published one or more NHL-completed games required "
    "for today's predictions. No prediction was generated from incomplete state."
)
FETCH_FAILURE_MESSAGE = (
    "Approved NHL or MoneyPuck source refresh failed. Stale state was not used. "
    "No prediction was generated."
)
IDENTITY_CONFLICT_MESSAGE = (
    "NHL and MoneyPuck game identity or timing disagree. "
    "No prediction was generated from inconsistent state."
)
STALE_STATE_MESSAGE = (
    "Provider material is stale relative to the requested observation. "
    "Stale state was not used. No prediction was generated."
)
INCOMPLETE_COVERAGE_MESSAGE = (
    "NHL schedule/status coverage is incomplete for this slate. "
    "No prediction was generated from partial schedule evidence."
)
CALENDAR_BOUND_MESSAGE = (
    "NHL schedule shows a 2026-27 regular-season game before the documented "
    "window start. No prediction was generated."
)
UNREPRODUCIBLE_CUTOFF_MESSAGE = (
    "The slate date is before the documented current-season coverage start. "
    "No prediction was generated."
)
SOURCE_CONTRACT_MESSAGE = (
    "Current-season source validation failed. "
    "No prediction was generated from incomplete or inconsistent state."
)
_BLOCKER_MESSAGES = {
    "moneypuck_publication_lag": PUBLICATION_LAG_MESSAGE,
    "source_fetch_failure": FETCH_FAILURE_MESSAGE,
    "source_identity_conflict": IDENTITY_CONFLICT_MESSAGE,
    "stale_provider_material": STALE_STATE_MESSAGE,
    "incomplete_nhl_coverage": INCOMPLETE_COVERAGE_MESSAGE,
    "calendar_bound_conflict": CALENDAR_BOUND_MESSAGE,
    "unreproducible_cutoff": UNREPRODUCIBLE_CUTOFF_MESSAGE,
    "source_contract_violation": SOURCE_CONTRACT_MESSAGE,
}
_GAME_ID_RE = re.compile(r"\b\d{10}\b")
_MODULE_FILENAMES = ("hosted_refresh.py", "issue91.py")
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "sklearn",
        "streamlit",
        "threading",
        "subprocess",
        "schedule",
        "requests",
    }
)
_FORBIDDEN_MODULES = frozenset(
    {
        "hockey_moneyline.prospective_scoring",
        "hockey_moneyline.prospective_artifact",
        "hockey_moneyline.issue52",
        "hockey_moneyline.issue55",
        "hockey_moneyline.issue68",
        "hockey_moneyline.prediction_record_store",
        "hockey_moneyline.elo",
        "hockey_moneyline.evaluate",
        "hockey_moneyline.forward_odds_archive",
        "hockey_moneyline.recommendation_contract",
        "hockey_moneyline.market_math",
        "hockey_moneyline.odds_gap_assist",
        "hockey_moneyline.acquire",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "score_pregame_prediction",
        "execute_manual_pregame_prediction",
        "execute_manual_opening_day_batch",
        "LogisticRegression",
        "fit",
        "predict",
        "predict_proba",
        "acquire_source_csv",
    }
)
_FORBIDDEN_SOURCE_TOKENS = (
    "data/" "cache",
    "pregame_" "prediction_records",
    "frozen_" "prospective_fitted_artifact",
)


def parse_observation_timestamp(value: datetime | str) -> datetime:
    """Return one UTC timestamp. Naive and malformed values are rejected."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise HostedRefreshError(
                "observation timestamp is malformed",
                code="malformed_timestamp",
            ) from exc
    else:
        raise HostedRefreshError(
            "observation timestamp is malformed",
            code="malformed_timestamp",
        )
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise HostedRefreshError(
            "observation timestamp must be timezone-aware; naive timestamps are rejected",
            code="naive_timestamp",
        )
    return parsed.astimezone(timezone.utc)


def parse_slate_date(value: date | str) -> date:
    """Return the NHL slate date. A timestamp is not accepted as a date."""

    if isinstance(value, datetime) or not isinstance(value, (date, str)):
        raise HostedRefreshError("slate date is malformed", code="malformed_slate_date")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HostedRefreshError("slate date is malformed", code="malformed_slate_date") from exc


@dataclass(frozen=True, slots=True)
class RefreshBlocker:
    """Calm failure 04 can show without reading provider payloads."""

    code: str
    message: str
    detail: str
    missing_game_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.code not in _BLOCKER_MESSAGES:
            raise HostedRefreshError(f"unknown hosted refresh blocker code {self.code}")
        if self.message != _BLOCKER_MESSAGES[self.code]:
            raise HostedRefreshError("hosted refresh blocker message does not match its code")
        if not self.detail:
            raise HostedRefreshError("hosted refresh blocker detail is empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "missing_game_ids": list(self.missing_game_ids),
        }


@dataclass(frozen=True, slots=True)
class NhlCoverageGame:
    game_id: str
    season: int
    game_type: str
    game_date: date
    scheduled_start_time_utc: datetime
    home_team: str
    away_team: str
    home_nhl_team_id: int
    away_nhl_team_id: int
    game_state: str
    game_schedule_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "season": self.season,
            "game_type": self.game_type,
            "game_date": self.game_date.isoformat(),
            "scheduled_start_time_utc": self.scheduled_start_time_utc.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_nhl_team_id": self.home_nhl_team_id,
            "away_nhl_team_id": self.away_nhl_team_id,
            "game_state": self.game_state,
            "game_schedule_state": self.game_schedule_state,
        }


@dataclass(frozen=True, slots=True)
class NhlSourceCoverage:
    """Parsed schedule/status coverage. This is not an NHL response body."""

    base_url: str
    schedule_date_path: str
    endpoint_path: str
    observed_at_utc: datetime
    covered_dates: tuple[date, ...]
    games: tuple[NhlCoverageGame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "schedule_date_path": self.schedule_date_path,
            "endpoint_path": self.endpoint_path,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "covered_dates": [day.isoformat() for day in self.covered_dates],
            "covered_date_count": len(self.covered_dates),
            "games": [game.to_dict() for game in self.games],
        }


@dataclass(frozen=True, slots=True)
class MoneyPuckSourceProvenance:
    """Hash and identity of the published CSV bytes fetched for this refresh."""

    source_id: str
    source_name: str
    source_page: str
    source_file_url: str
    access_method: str
    sha256: str
    byte_count: int
    retrieved_at_utc: datetime
    situation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_page": self.source_page,
            "source_file_url": self.source_file_url,
            "access_method": self.access_method,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "retrieved_at_utc": self.retrieved_at_utc.isoformat(),
            "situation": self.situation,
        }


@dataclass(frozen=True, slots=True)
class HostedSlateTarget:
    """One regular-season game on the requested NHL slate date."""

    game_id: str
    season: int
    game_type: str
    game_date: date
    scheduled_start_time_utc: datetime
    observed_at_utc: datetime
    home_team_id: str
    away_team_id: str
    home_nhl_team_id: int
    away_nhl_team_id: int
    game_state: str
    game_schedule_state: str
    scoring_eligible: bool
    ineligibility_reason: str | None
    prior_state: Any | None

    def to_dict(self) -> dict[str, Any]:
        prior = None
        if self.prior_state is not None:
            prior = self.prior_state.to_audit_dict()
        return {
            "game_id": self.game_id,
            "season": self.season,
            "game_type": self.game_type,
            "game_date": self.game_date.isoformat(),
            "scheduled_start_time_utc": self.scheduled_start_time_utc.isoformat(),
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "home_nhl_team_id": self.home_nhl_team_id,
            "away_nhl_team_id": self.away_nhl_team_id,
            "game_state": self.game_state,
            "game_schedule_state": self.game_schedule_state,
            "scoring_eligible": self.scoring_eligible,
            "ineligibility_reason": self.ineligibility_reason,
            "prior_state": prior,
        }

    def as_unscored_target(self) -> UnscoredTarget:
        require_hosted_pregame_scoring_eligibility(self)
        return UnscoredTarget(
            game_id=self.game_id,
            game_date=self.game_date,
            season=self.season,
            game_type=self.game_type,
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
            scheduled_start_time_utc=self.scheduled_start_time_utc,
            observed_at_utc=self.observed_at_utc,
            nhl_game_id=self.game_id,
            home_nhl_team_id=self.home_nhl_team_id,
            away_nhl_team_id=self.away_nhl_team_id,
            source=HOSTED_TARGET_SOURCE,
        )


@dataclass(frozen=True, slots=True)
class HostedScoringInput:
    """Validated pregame target and Issue #54 package for one eligible game.

    03 consumes this object. It does not fetch providers or repeat validation.
    """

    target: UnscoredTarget
    prior_state: Any

    def issue52_scoring_kwargs(self) -> dict[str, Any]:
        return self.prior_state.as_issue52_scoring_kwargs()


@dataclass(frozen=True, slots=True)
class HostedRefreshBundle:
    """Deterministic refresh result. Blocked bundles expose no scoring inputs."""

    schema_version: int
    kind: str
    readiness: str
    blocker: RefreshBlocker | None
    observed_at_utc: datetime
    slate_date: date
    nhl_coverage: NhlSourceCoverage | None
    moneypuck_provenance: MoneyPuckSourceProvenance | None
    required_completed_game_ids: tuple[str, ...]
    excluded_preseason_game_ids: tuple[str, ...]
    slate_targets: tuple[HostedSlateTarget, ...]

    def __post_init__(self) -> None:
        if self.schema_version != HOSTED_REFRESH_SCHEMA_VERSION or self.kind != HOSTED_REFRESH_KIND:
            raise HostedRefreshError("unsupported hosted refresh bundle")
        if self.readiness not in {READY, BLOCKED}:
            raise HostedRefreshError("unsupported hosted refresh readiness")
        if self.observed_at_utc.tzinfo is None:
            raise HostedRefreshError("observation timestamp must be timezone-aware")
        if (self.readiness == READY) != (self.blocker is None):
            raise HostedRefreshError("readiness and blocker must agree")
        for target in self.slate_targets:
            if target.scoring_eligible:
                if self.readiness != READY or target.prior_state is None:
                    raise HostedRefreshError("eligible target is missing validated prior state")
                if target.ineligibility_reason is not None:
                    raise HostedRefreshError("eligible target cannot carry an ineligibility reason")
                if target.game_state != FUT_GAME_STATE:
                    raise HostedRefreshError("a target that is no longer FUT cannot be scoring eligible")
                if target.prior_state.cutoff_game_id != target.game_id:
                    raise HostedRefreshError("prior-state cutoff does not match the slate target")
                if target.prior_state.observed_at_utc != target.observed_at_utc:
                    raise HostedRefreshError("prior-state observation timestamp does not match the refresh")
                if (
                    target.prior_state.eligible_prior_completed_game_ids
                    != self.required_completed_game_ids
                ):
                    raise HostedRefreshError(
                        "prior package disagrees with required completed games"
                    )
            elif not target.ineligibility_reason or target.prior_state is not None:
                raise HostedRefreshError(
                    "ineligible target must explain why and must not carry prior state"
                )
        if self.readiness == BLOCKED:
            for target in self.slate_targets:
                if target.scoring_eligible or target.prior_state is not None:
                    raise HostedRefreshError(
                        "BLOCKED refresh cannot expose hosted pregame scoring inputs"
                    )

    @property
    def prior_states(self) -> tuple[Any, ...]:
        return tuple(
            target.prior_state
            for target in self.slate_targets
            if target.prior_state is not None
        )

    def scoring_inputs(self) -> tuple[HostedScoringInput, ...]:
        """Return eligible pregame inputs. A blocked refresh raises."""

        if self.readiness != READY or self.blocker is not None:
            message = self.blocker.message if self.blocker is not None else "refresh is not ready"
            code = self.blocker.code if self.blocker is not None else "blocked"
            raise HostedRefreshError(message, code=code)
        built: list[HostedScoringInput] = []
        for target in self.slate_targets:
            if not target.scoring_eligible:
                continue
            built.append(
                HostedScoringInput(
                    target=target.as_unscored_target(),
                    prior_state=target.prior_state,
                )
            )
        return tuple(built)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "readiness": self.readiness,
            "blocker": None if self.blocker is None else self.blocker.to_dict(),
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "slate_date": self.slate_date.isoformat(),
            "stale_state_used": False,
            "protected_record_written": False,
            "model_scored": False,
            "nhl_coverage": None if self.nhl_coverage is None else self.nhl_coverage.to_dict(),
            "moneypuck_provenance": (
                None
                if self.moneypuck_provenance is None
                else self.moneypuck_provenance.to_dict()
            ),
            "required_completed_game_ids": list(self.required_completed_game_ids),
            "excluded_preseason_game_ids": list(self.excluded_preseason_game_ids),
            "slate_targets": [target.to_dict() for target in self.slate_targets],
        }


def require_hosted_pregame_scoring_eligibility(target: HostedSlateTarget) -> None:
    """Reject any slate row 03 must not score."""

    if not isinstance(target, HostedSlateTarget):
        raise HostedRefreshError(
            "hosted pregame eligibility requires a HostedSlateTarget from the refresh bundle"
        )
    if target.game_state != FUT_GAME_STATE:
        raise HostedRefreshError(
            _no_longer_fut_message(target.game_id, target.game_state),
            code="target_not_fut",
        )
    if target.observed_at_utc >= target.scheduled_start_time_utc:
        raise HostedRefreshError(
            _puck_drop_message(target.game_id),
            code="observation_at_or_after_puck_drop",
        )
    if not target.scoring_eligible or target.prior_state is None:
        raise HostedRefreshError(
            target.ineligibility_reason
            or f"Target {target.game_id} is not eligible for hosted pregame scoring.",
            code="target_not_eligible",
        )
    if target.prior_state.cutoff_game_id != target.game_id:
        raise HostedRefreshError(
            "prior-state cutoff does not match the slate target",
            code="source_identity_conflict",
        )
    if target.prior_state.observed_at_utc != target.observed_at_utc:
        raise HostedRefreshError(
            "prior-state observation timestamp does not match the refresh",
            code="stale_provider_material",
        )


def refresh_hosted_slate(
    slate_date: date | str,
    observed_at_utc: datetime | str,
    *,
    nhl_opener: Callable[..., Any] = urlopen,
    moneypuck_opener: Callable[..., Any] = urlopen,
    timeout: int = 30,
    moneypuck_timeout: int = 120,
) -> HostedRefreshBundle:
    """Fetch, validate, and return one in-memory slate refresh bundle.

    ``slate_date`` is the NHL game date. ``observed_at_utc`` is the single
    timezone-aware timestamp that binds both fetches. Pass a new timestamp
    when the operator refreshes. This function does not read a previous file.
    """

    observed = parse_observation_timestamp(observed_at_utc)
    slate = parse_slate_date(slate_date)
    if slate < current_season_coverage_start():
        return _bundle(
            observed=observed,
            slate=slate,
            readiness=BLOCKED,
            blocker=_blocker(
                "unreproducible_cutoff",
                detail=(
                    "slate date "
                    f"{slate.isoformat()} is before coverage start "
                    f"{current_season_coverage_start().isoformat()}"
                ),
            ),
        )

    try:
        snapshot = assemble_current_season_nhl_coverage(
            slate,
            observed_at_utc=observed,
            timeout=timeout,
            opener=nhl_opener,
        )
    except HockeyMoneylineError as exc:
        return _bundle(
            observed=observed,
            slate=slate,
            readiness=BLOCKED,
            blocker=_blocker("source_fetch_failure", detail=str(exc)),
        )
    except (OSError, TimeoutError) as exc:
        return _bundle(
            observed=observed,
            slate=slate,
            readiness=BLOCKED,
            blocker=_blocker("source_fetch_failure", detail=f"NHL refresh failed: {exc}"),
        )

    nhl_coverage = _nhl_coverage(snapshot)
    slate_games = _slate_games(snapshot, slate)
    preseason_ids = _preseason_ids(snapshot)
    required_games = _required_completed_games(snapshot, slate)
    required_ids = tuple(game.nhl_game_id for game in required_games)

    try:
        payload = _fetch_moneypuck_published_csv(
            timeout=moneypuck_timeout,
            opener=moneypuck_opener,
        )
    except HostedRefreshError as exc:
        return _blocked_with_inventory(
            observed=observed,
            slate=slate,
            nhl_coverage=nhl_coverage,
            slate_games=slate_games,
            required_ids=required_ids,
            excluded_preseason_game_ids=preseason_ids,
            blocker=_blocker(
                exc.code if exc.code in _BLOCKER_MESSAGES else "source_fetch_failure",
                detail=str(exc),
            ),
        )

    provenance = _provenance(payload, observed)
    try:
        rows = _rows_from_csv_bytes(payload)
        parsed = parse_current_season_moneypuck_rows(rows)
    except HostedRefreshError as exc:
        return _blocked_with_inventory(
            observed=observed,
            slate=slate,
            nhl_coverage=nhl_coverage,
            moneypuck=provenance,
            slate_games=slate_games,
            required_ids=required_ids,
            excluded_preseason_game_ids=preseason_ids,
            blocker=_blocker(
                exc.code if exc.code in _BLOCKER_MESSAGES else "source_contract_violation",
                detail=str(exc),
            ),
        )
    except HockeyMoneylineError as exc:
        return _blocked_with_inventory(
            observed=observed,
            slate=slate,
            nhl_coverage=nhl_coverage,
            moneypuck=provenance,
            slate_games=slate_games,
            required_ids=required_ids,
            excluded_preseason_game_ids=preseason_ids,
            blocker=_blocker_from_provider_error(exc),
        )

    excluded = tuple(
        sorted(set(preseason_ids) | set(parsed.excluded_preseason_game_ids))
    )
    provider_blocker = _provider_blocker(
        snapshot=snapshot,
        parsed=parsed,
        slate=slate,
        observed=observed,
        required_games=required_games,
    )
    if provider_blocker is not None:
        return _blocked_with_inventory(
            observed=observed,
            slate=slate,
            nhl_coverage=nhl_coverage,
            moneypuck=provenance,
            slate_games=slate_games,
            required_ids=required_ids,
            excluded_preseason_game_ids=excluded,
            blocker=provider_blocker,
        )

    packages: dict[str, Any] = {}
    prepared: list[tuple[NHLScheduleGame, UnscoredTarget | None, str | None]] = []
    try:
        for game in slate_games:
            eligible, reason = _eligibility(game, observed)
            if not eligible:
                prepared.append((game, None, reason))
                continue
            prepared.append((game, _unscored_target(game, observed), None))
    except HostedRefreshError as exc:
        return _blocked_with_inventory(
            observed=observed,
            slate=slate,
            nhl_coverage=nhl_coverage,
            moneypuck=provenance,
            slate_games=slate_games,
            required_ids=required_ids,
            excluded_preseason_game_ids=excluded,
            blocker=_blocker(
                exc.code if exc.code in _BLOCKER_MESSAGES else "source_identity_conflict",
                detail=str(exc),
            ),
        )

    for game, unscored, reason in prepared:
        if unscored is None:
            continue
        try:
            packages[game.nhl_game_id] = build_current_season_prior_state(
                unscored,
                snapshot,
                rows,
                observed_at_utc=observed,
            )
        except HockeyMoneylineError as exc:
            return _blocked_with_inventory(
                observed=observed,
                slate=slate,
                nhl_coverage=nhl_coverage,
                moneypuck=provenance,
                slate_games=slate_games,
                required_ids=required_ids,
                excluded_preseason_game_ids=excluded,
                blocker=_blocker_from_provider_error(exc),
            )

    targets: list[HostedSlateTarget] = []
    for game, unscored, reason in prepared:
        if unscored is None:
            if not reason:
                return _blocked_with_inventory(
                    observed=observed,
                    slate=slate,
                    nhl_coverage=nhl_coverage,
                    moneypuck=provenance,
                    slate_games=slate_games,
                    required_ids=required_ids,
                    excluded_preseason_game_ids=excluded,
                    blocker=_blocker(
                        "source_contract_violation",
                        detail=f"slate game {game.nhl_game_id} has no eligibility decision",
                    ),
                )
            targets.append(
                _slate_target(
                    game,
                    observed=observed,
                    scoring_eligible=False,
                    ineligibility_reason=reason,
                    prior_state=None,
                    strict_teams=False,
                )
            )
            continue
        package = packages.get(game.nhl_game_id)
        if package is None:
            return _blocked_with_inventory(
                observed=observed,
                slate=slate,
                nhl_coverage=nhl_coverage,
                moneypuck=provenance,
                slate_games=slate_games,
                required_ids=required_ids,
                excluded_preseason_game_ids=excluded,
                blocker=_blocker(
                    "source_contract_violation",
                    detail=f"validated prior state missing for eligible target {game.nhl_game_id}",
                ),
            )
        targets.append(
            _slate_target(
                game,
                observed=observed,
                scoring_eligible=True,
                ineligibility_reason=None,
                prior_state=package,
                home_team_id=unscored.home_team_id,
                away_team_id=unscored.away_team_id,
            )
        )
    return _bundle(
        observed=observed,
        slate=slate,
        readiness=READY,
        blocker=None,
        nhl_coverage=nhl_coverage,
        moneypuck=provenance,
        required_ids=required_ids,
        excluded_preseason_game_ids=excluded,
        slate_targets=tuple(targets),
    )


def assert_hosted_refresh_does_not_score() -> None:
    """Fail if this lane imports or calls model fit/score, odds, or record writes."""

    module_dir = Path(__file__).resolve().parent
    for name in _MODULE_FILENAMES:
        path = module_dir / name
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_SOURCE_TOKENS:
            if token in source:
                raise HostedRefreshError(f"{name} references durable store token {token}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            for module_name in modules:
                root = module_name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORT_ROOTS or module_name in _FORBIDDEN_MODULES:
                    raise HostedRefreshError(
                        f"{name} imports a scoring, odds, or store module: {module_name}"
                    )
            called = _called_name(node)
            if called in _FORBIDDEN_CALLS:
                raise HostedRefreshError(f"{name} calls {called}")


def _called_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _blocker(
    code: str,
    *,
    detail: str,
    missing_game_ids: Sequence[str] = (),
) -> RefreshBlocker:
    return RefreshBlocker(
        code=code,
        message=_BLOCKER_MESSAGES[code],
        detail=detail,
        missing_game_ids=tuple(missing_game_ids),
    )


def _blocker_from_provider_error(exc: HockeyMoneylineError) -> RefreshBlocker:
    text = str(exc)
    code = _provider_error_code(text)
    missing = tuple(dict.fromkeys(_GAME_ID_RE.findall(text)))
    if code != "moneypuck_publication_lag":
        missing = ()
    return _blocker(code, detail=text, missing_game_ids=missing)


def _provider_error_code(text: str) -> str:
    if "before the documented window start" in text:
        return "calendar_bound_conflict"
    if "stale relative" in text:
        return "stale_provider_material"
    if "coverage is incomplete" in text or "cannot be reproduced" in text:
        return "incomplete_nhl_coverage"
    if "Source identities conflict" in text or (
        "Unknown " in text and "team identity" in text
    ):
        return "source_identity_conflict"
    if _is_publication_lag(text):
        return "moneypuck_publication_lag"
    return "source_contract_violation"


def _is_publication_lag(text: str) -> bool:
    markers = (
        "required MoneyPuck completed-state records are missing",
        "Required winner-labeled completed result is missing",
        "Prior xGoal data required by the frozen state is unavailable",
        "is unavailable for game",
        "is non-numeric",
        "is non-finite",
        "missing columns",
        "missing its AWAY row",
        "have no HOME pair",
        "is missing goalsFor",
        "is missing goalsAgainst",
        "is missing ",
        "no header",
    )
    return any(marker in text for marker in markers)


def _no_longer_fut_message(game_id: str, game_state: str) -> str:
    return (
        f"Target {game_id} is no longer FUT; NHL game_state is {game_state!r}. "
        "Hosted pregame scoring is rejected."
    )


def _puck_drop_message(game_id: str) -> str:
    return (
        f"Target {game_id} observation is at or after puck drop. "
        "Hosted pregame scoring is rejected."
    )


def _bundle(
    *,
    observed: datetime,
    slate: date,
    readiness: str,
    blocker: RefreshBlocker | None,
    nhl_coverage: NhlSourceCoverage | None = None,
    moneypuck: MoneyPuckSourceProvenance | None = None,
    required_ids: tuple[str, ...] = (),
    excluded_preseason_game_ids: tuple[str, ...] = (),
    slate_targets: tuple[HostedSlateTarget, ...] = (),
) -> HostedRefreshBundle:
    return HostedRefreshBundle(
        schema_version=HOSTED_REFRESH_SCHEMA_VERSION,
        kind=HOSTED_REFRESH_KIND,
        readiness=readiness,
        blocker=blocker,
        observed_at_utc=observed,
        slate_date=slate,
        nhl_coverage=nhl_coverage,
        moneypuck_provenance=moneypuck,
        required_completed_game_ids=required_ids,
        excluded_preseason_game_ids=excluded_preseason_game_ids,
        slate_targets=slate_targets,
    )


def _blocked_with_inventory(
    *,
    observed: datetime,
    slate: date,
    nhl_coverage: NhlSourceCoverage,
    slate_games: Sequence[NHLScheduleGame],
    required_ids: tuple[str, ...],
    excluded_preseason_game_ids: tuple[str, ...],
    blocker: RefreshBlocker,
    moneypuck: MoneyPuckSourceProvenance | None = None,
) -> HostedRefreshBundle:
    targets = tuple(
        _slate_target(
            game,
            observed=observed,
            scoring_eligible=False,
            ineligibility_reason=blocker.message,
            prior_state=None,
            strict_teams=False,
        )
        for game in slate_games
    )
    return _bundle(
        observed=observed,
        slate=slate,
        readiness=BLOCKED,
        blocker=blocker,
        nhl_coverage=nhl_coverage,
        moneypuck=moneypuck,
        required_ids=required_ids,
        excluded_preseason_game_ids=excluded_preseason_game_ids,
        slate_targets=targets,
    )


def _nhl_coverage(snapshot: NHLScheduleSnapshot) -> NhlSourceCoverage:
    return NhlSourceCoverage(
        base_url=NHL_WEB_BASE_URL,
        schedule_date_path=SCHEDULE_DATE_PATH,
        endpoint_path=snapshot.endpoint_path,
        observed_at_utc=snapshot.observed_at_utc,
        covered_dates=tuple(sorted(snapshot.covered_dates)),
        games=tuple(
            NhlCoverageGame(
                game_id=game.nhl_game_id,
                season=game.season,
                game_type=game.game_type,
                game_date=game.game_date,
                scheduled_start_time_utc=game.scheduled_start_time_utc,
                home_team=game.home_team_abbrev,
                away_team=game.away_team_abbrev,
                home_nhl_team_id=game.home_nhl_team_id,
                away_nhl_team_id=game.away_nhl_team_id,
                game_state=game.game_state,
                game_schedule_state=game.game_schedule_state,
            )
            for game in snapshot.games
        ),
    )


def _provenance(payload: bytes, observed: datetime) -> MoneyPuckSourceProvenance:
    return MoneyPuckSourceProvenance(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        source_page=SOURCE_PAGE,
        source_file_url=SOURCE_FILE_URL,
        access_method=ACCESS_METHOD,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        retrieved_at_utc=observed,
        situation=MONEYPUCK_SITUATION,
    )


def _fetch_moneypuck_published_csv(
    *,
    timeout: int,
    opener: Callable[..., Any],
) -> bytes:
    request = Request(
        SOURCE_FILE_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise HostedRefreshError(
                    f"Unexpected HTTP {status} from MoneyPuck",
                    code="source_fetch_failure",
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    raise HostedRefreshError(
                        "MoneyPuck response body was not bytes",
                        code="source_fetch_failure",
                    )
                total += len(chunk)
                if total > MAX_MONEYPUCK_RESPONSE_BYTES:
                    raise HostedRefreshError(
                        "MoneyPuck response exceeded the bounded in-memory refresh limit",
                        code="source_fetch_failure",
                    )
                chunks.append(bytes(chunk))
    except HostedRefreshError:
        raise
    except HTTPError as exc:
        raise HostedRefreshError(
            f"HTTP {exc.code} fetching MoneyPuck",
            code="source_fetch_failure",
        ) from exc
    except URLError as exc:
        raise HostedRefreshError(
            f"Network failure fetching MoneyPuck: {exc.reason}",
            code="source_fetch_failure",
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise HostedRefreshError(
            f"MoneyPuck refresh failed: {exc}",
            code="source_fetch_failure",
        ) from exc
    payload = b"".join(chunks)
    if not payload:
        raise HostedRefreshError(
            "MoneyPuck published file was empty",
            code="source_fetch_failure",
        )
    return payload


def _rows_from_csv_bytes(payload: bytes) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HostedRefreshError(
            "MoneyPuck payload is not valid UTF-8",
            code="source_fetch_failure",
        ) from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HostedRefreshError(
            "MoneyPuck file has no header",
            code="moneypuck_publication_lag",
        )
    missing = [
        column for column in MONEYPUCK_CURRENT_SEASON_COLUMNS if column not in reader.fieldnames
    ]
    if missing:
        raise HostedRefreshError(
            f"MoneyPuck file is missing columns {missing}",
            code="moneypuck_publication_lag",
        )
    return [{key: value if value is not None else "" for key, value in row.items()} for row in reader]


def _slate_games(snapshot: NHLScheduleSnapshot, slate: date) -> tuple[NHLScheduleGame, ...]:
    selected = [
        game
        for game in snapshot.games
        if game.game_date == slate
        and game.season == CURRENT_SEASON
        and game.game_type == "regular"
    ]
    selected.sort(key=lambda game: (game.scheduled_start_time_utc, game.nhl_game_id))
    return tuple(selected)


def _required_completed_games(
    snapshot: NHLScheduleSnapshot,
    slate: date,
) -> tuple[NHLScheduleGame, ...]:
    selected = [
        game
        for game in snapshot.games
        if game.season == CURRENT_SEASON
        and game.game_type == "regular"
        and game.game_state in FINAL_GAME_STATES
        and game.game_date < slate
    ]
    selected.sort(key=lambda game: (game.game_date, game.nhl_game_id))
    return tuple(selected)


def _preseason_ids(snapshot: NHLScheduleSnapshot) -> tuple[str, ...]:
    return tuple(
        sorted(game.nhl_game_id for game in snapshot.games if game.game_type == "preseason")
    )


def _provider_blocker(
    *,
    snapshot: NHLScheduleSnapshot,
    parsed: Any,
    slate: date,
    observed: datetime,
    required_games: Sequence[NHLScheduleGame],
) -> RefreshBlocker | None:
    if snapshot.observed_at_utc.astimezone(timezone.utc) != observed:
        return _blocker(
            "stale_provider_material",
            detail=(
                "NHL snapshot observation "
                f"{snapshot.observed_at_utc.isoformat()} disagrees with "
                f"{observed.isoformat()}"
            ),
        )
    missing_dates = [
        day for day in required_coverage_dates(slate) if day not in snapshot.covered_dates
    ]
    if missing_dates:
        return _blocker(
            "incomplete_nhl_coverage",
            detail=f"missing NHL coverage date {missing_dates[0].isoformat()}",
        )
    early = [
        game
        for game in snapshot.games
        if game.season == CURRENT_SEASON
        and game.game_type == "regular"
        and game.game_date < CURRENT_SEASON_REGULAR_WINDOW_START
    ]
    if early:
        game = early[0]
        return _blocker(
            "calendar_bound_conflict",
            detail=(
                f"{game.nhl_game_id} on {game.game_date.isoformat()} is before "
                f"{CURRENT_SEASON_REGULAR_WINDOW_START.isoformat()}"
            ),
        )

    history = {game.game_id: game for game in parsed.schedule_history}
    winners = {game.game_id: game for game in parsed.winner_labeled}
    missing: list[str] = []
    for nhl_game in required_games:
        history_game = history.get(nhl_game.nhl_game_id)
        if history_game is None:
            missing.append(nhl_game.nhl_game_id)
            continue
        expected = (
            nhl_game.game_date,
            nhl_game.home_team_abbrev,
            nhl_game.away_team_abbrev,
            nhl_game.season,
        )
        observed_identity = (
            history_game.game_date,
            history_game.home_team_id,
            history_game.away_team_id,
            history_game.season,
        )
        if expected != observed_identity:
            return _blocker(
                "source_identity_conflict",
                detail=(
                    f"NHL {expected} != MoneyPuck {observed_identity} "
                    f"for {nhl_game.nhl_game_id}"
                ),
            )
        start = nhl_game.scheduled_start_time_utc.astimezone(timezone.utc)
        if start > observed:
            return _blocker(
                "stale_provider_material",
                detail=(
                    f"observation {observed.isoformat()} is before completed game "
                    f"{nhl_game.nhl_game_id} start {start.isoformat()}"
                ),
            )
        winner = winners.get(nhl_game.nhl_game_id)
        if winner is None:
            if nhl_game.nhl_game_id not in parsed.history_only_tied_game_ids:
                return _blocker(
                    "moneypuck_publication_lag",
                    detail=(
                        "required winner-labeled result or tied schedule-history "
                        f"row is missing for {nhl_game.nhl_game_id}"
                    ),
                    missing_game_ids=(nhl_game.nhl_game_id,),
                )
            continue
        for team_id in (winner.home_team_id, winner.away_team_id):
            if (winner.game_id, team_id) not in parsed.xgoal_records:
                return _blocker(
                    "moneypuck_publication_lag",
                    detail=(
                        f"required xGoal material is missing for {winner.game_id} team {team_id}"
                    ),
                    missing_game_ids=(winner.game_id,),
                )
    if missing:
        return _blocker(
            "moneypuck_publication_lag",
            detail=(
                "NHL final games absent from MoneyPuck current-season "
                f"regular-season material: {', '.join(missing)}"
            ),
            missing_game_ids=tuple(missing),
        )

    nhl_by_id = {game.nhl_game_id: game for game in snapshot.games}
    for history_game in parsed.schedule_history:
        if history_game.game_date >= slate:
            continue
        nhl_game = nhl_by_id.get(history_game.game_id)
        if nhl_game is None:
            return _blocker(
                "source_identity_conflict",
                detail=(
                    "MoneyPuck current-season game "
                    f"{history_game.game_id} is absent from NHL schedule/status evidence"
                ),
            )
        if nhl_game.game_type != "regular" or nhl_game.season != CURRENT_SEASON:
            return _blocker(
                "source_identity_conflict",
                detail=(
                    f"MoneyPuck regular-season row {history_game.game_id} disagrees "
                    "with NHL game type or season"
                ),
            )
        if nhl_game.game_state not in FINAL_GAME_STATES:
            return _blocker(
                "source_identity_conflict",
                detail=(
                    f"MoneyPuck has a completed row for {history_game.game_id} while "
                    f"NHL state is {nhl_game.game_state!r}"
                ),
            )
    for game in _slate_games(snapshot, slate):
        if game.game_state != FUT_GAME_STATE:
            continue
        if game.scheduled_start_time_utc.astimezone(timezone.utc) <= observed:
            continue
        if game.nhl_game_id in history or game.nhl_game_id in winners:
            return _blocker(
                "source_identity_conflict",
                detail=(
                    "MoneyPuck current-season material includes unscored target "
                    f"{game.nhl_game_id} as completed state"
                ),
            )
    return None


def _eligibility(game: NHLScheduleGame, observed: datetime) -> tuple[bool, str | None]:
    if game.game_state != FUT_GAME_STATE:
        return False, _no_longer_fut_message(game.nhl_game_id, game.game_state)
    start = game.scheduled_start_time_utc.astimezone(timezone.utc)
    if observed >= start:
        return False, _puck_drop_message(game.nhl_game_id)
    if game.season != CURRENT_SEASON or game.game_type != "regular":
        return False, (
            f"Target {game.nhl_game_id} is not a 2026-27 regular-season game. "
            "Hosted pregame scoring is rejected."
        )
    return True, None


def _canonical_team(code: str, season: int) -> str:
    try:
        canonical = canonical_moneypuck_team_code(code, season)
    except HockeyMoneylineError as exc:
        raise HostedRefreshError(str(exc), code="source_identity_conflict") from exc
    if canonical not in TARGET_ERA_TEAM_CODES:
        raise HostedRefreshError(
            f"Unknown team identity {code!r} in current season {season}",
            code="source_identity_conflict",
        )
    return canonical


def _team_label(code: str, season: int) -> str:
    try:
        return _canonical_team(code, season)
    except HostedRefreshError:
        return str(code or "").strip().upper()


def _unscored_target(game: NHLScheduleGame, observed: datetime) -> UnscoredTarget:
    home = _canonical_team(game.home_team_abbrev, game.season)
    away = _canonical_team(game.away_team_abbrev, game.season)
    try:
        return UnscoredTarget(
            game_id=game.nhl_game_id,
            game_date=game.game_date,
            season=game.season,
            game_type=game.game_type,
            home_team_id=home,
            away_team_id=away,
            scheduled_start_time_utc=game.scheduled_start_time_utc.astimezone(timezone.utc),
            observed_at_utc=observed,
            nhl_game_id=game.nhl_game_id,
            home_nhl_team_id=game.home_nhl_team_id,
            away_nhl_team_id=game.away_nhl_team_id,
            source=HOSTED_TARGET_SOURCE,
        )
    except HockeyMoneylineError as exc:
        raise HostedRefreshError(str(exc), code="source_identity_conflict") from exc


def _slate_target(
    game: NHLScheduleGame,
    *,
    observed: datetime,
    scoring_eligible: bool,
    ineligibility_reason: str | None,
    prior_state: Any | None,
    strict_teams: bool = True,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
) -> HostedSlateTarget:
    if home_team_id is None or away_team_id is None:
        label = _canonical_team if strict_teams else _team_label
        home_team_id = label(game.home_team_abbrev, game.season)
        away_team_id = label(game.away_team_abbrev, game.season)
    return HostedSlateTarget(
        game_id=game.nhl_game_id,
        season=game.season,
        game_type=game.game_type,
        game_date=game.game_date,
        scheduled_start_time_utc=game.scheduled_start_time_utc.astimezone(timezone.utc),
        observed_at_utc=observed,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_nhl_team_id=game.home_nhl_team_id,
        away_nhl_team_id=game.away_nhl_team_id,
        game_state=game.game_state,
        game_schedule_state=game.game_schedule_state,
        scoring_eligible=scoring_eligible,
        ineligibility_reason=ineligibility_reason,
        prior_state=prior_state,
    )
