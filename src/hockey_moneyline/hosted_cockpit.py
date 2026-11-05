"""Issue #93 hosted product composition for the private Hockey cockpit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from hockey_moneyline.decision_cockpit import (
    DecisionCockpitView,
    MarketSnapshotReader,
    NullMarketSnapshotReader,
    build_decision_cockpit_view,
)
from hockey_moneyline.hosted_refresh import READY, HostedRefreshBundle, refresh_hosted_slate
from hockey_moneyline.hosted_scoring import (
    SCORING_BLOCKED,
    SCORED,
    HostedGameScore,
    HostedScoringBundle,
    score_hosted_refresh_bundle,
)
from hockey_moneyline.issue55 import load_frozen_development_history
from hockey_moneyline.issue97 import (
    _CapturingMoneyPuckOpener,
    ensure_frozen_historical_cache,
)
from hockey_moneyline.prospective_artifact import EXPECTED_FITTED_ARTIFACT_SHA256

DEFAULT_HOSTED_DISPLAY_TIMEZONE = "America/Vancouver"
CURRENT_CALCULATION_LABEL = "Current hosted web calculation"
PROTECTED_EVIDENCE_LABEL = (
    "Protected receipt evidence is separate. Refresh Data does not create a protected "
    "prediction record or post a GitHub receipt."
)


@dataclass(frozen=True, slots=True)
class _HostedDashboardGame:
    nhl_game_id: str
    away_team: str
    home_team: str
    scheduled_local_label: str
    scheduled_utc_label: str
    scheduled_start_time_utc: str
    prediction_status: str
    home_model_win_probability: float | None
    away_model_win_probability: float | None
    audit: None = None


@dataclass(frozen=True, slots=True)
class _HostedDashboardView:
    selected_date: date
    games: tuple[_HostedDashboardGame, ...]


@dataclass(frozen=True, slots=True)
class HostedCockpitView:
    selected_date: date
    observed_at_utc: str
    refresh_readiness: str
    scoring_state: str
    nhl_status: str
    moneypuck_status: str
    model_status: str
    calculation_label: str
    protected_evidence_label: str
    blocker_code: str | None
    blocker_message: str | None
    blocker_detail: str | None
    decision_view: DecisionCockpitView
    scores: tuple[HostedGameScore, ...]

    @property
    def has_scored_games(self) -> bool:
        return any(score.status == SCORED for score in self.scores)


@dataclass(frozen=True, slots=True)
class HostedCockpitRun:
    refresh: HostedRefreshBundle
    scoring: HostedScoringBundle
    view: HostedCockpitView

    @property
    def is_successful(self) -> bool:
        return self.refresh.readiness == READY and self.scoring.scoring_state != SCORING_BLOCKED


def run_hosted_refresh(
    *,
    slate_date: date | str,
    observed_at_utc: datetime | str,
    market_reader: MarketSnapshotReader | None = None,
    refresh_kwargs: Mapping[str, Any] | None = None,
    frozen_history: tuple[Any, Any, Any] | None = None,
) -> HostedCockpitRun:
    kwargs = dict(refresh_kwargs or {})
    capture: _CapturingMoneyPuckOpener | None = None
    if frozen_history is None:
        base_moneypuck_opener = kwargs.pop("moneypuck_opener", urlopen)
        capture = _CapturingMoneyPuckOpener(base_moneypuck_opener)
        kwargs["moneypuck_opener"] = capture

    refresh = refresh_hosted_slate(
        slate_date,
        observed_at_utc,
        **kwargs,
    )

    scoring_inputs = refresh.scoring_inputs() if refresh.readiness == READY else ()
    if scoring_inputs and frozen_history is None:
        if capture is None or not capture.payload:
            raise ValueError(
                "READY hosted refresh did not retain the MoneyPuck bytes required "
                "to reconstruct frozen feature history"
            )
        with TemporaryDirectory(prefix="hockey-hosted-frozen-") as temp_dir:
            cache_dir = Path(temp_dir)
            ensure_frozen_historical_cache(cache_dir, capture.payload)
            frozen_history = load_frozen_development_history(cache_dir)

    if frozen_history is None:
        scoring = score_hosted_refresh_bundle(refresh)
    else:
        frozen_games, frozen_records, frozen_schedule = frozen_history
        scoring = score_hosted_refresh_bundle(
            refresh,
            frozen_completed_history=frozen_games,
            frozen_advanced_records=frozen_records,
            frozen_schedule_history=frozen_schedule,
        )

    view = build_hosted_cockpit_view(
        refresh_bundle=refresh,
        scoring_bundle=scoring,
        market_reader=market_reader,
    )
    return HostedCockpitRun(refresh=refresh, scoring=scoring, view=view)


def choose_display_run(
    *,
    latest_attempt: HostedCockpitRun | None,
    last_successful: HostedCockpitRun | None,
) -> HostedCockpitRun | None:
    if latest_attempt is not None and latest_attempt.is_successful:
        return latest_attempt
    if last_successful is not None and last_successful.is_successful:
        return last_successful
    return None


def build_hosted_cockpit_view(
    *,
    refresh_bundle: HostedRefreshBundle,
    scoring_bundle: HostedScoringBundle,
    market_reader: MarketSnapshotReader | None = None,
    display_timezone: str = DEFAULT_HOSTED_DISPLAY_TIMEZONE,
) -> HostedCockpitView:
    if scoring_bundle.slate_date != refresh_bundle.slate_date.isoformat():
        raise ValueError("hosted scoring slate date does not match refresh bundle")
    if scoring_bundle.observed_at_utc != refresh_bundle.observed_at_utc.isoformat():
        raise ValueError("hosted scoring observation does not match refresh bundle")

    dashboard = _HostedDashboardView(
        selected_date=refresh_bundle.slate_date,
        games=tuple(
            _dashboard_game_from_score(score, display_timezone=display_timezone)
            for score in scoring_bundle.games
        ),
    )
    decision = build_decision_cockpit_view(
        dashboard_view=dashboard,
        market_reader=market_reader or NullMarketSnapshotReader(),
    )

    blocker_code = None
    blocker_message = None
    blocker_detail = None
    if refresh_bundle.blocker is not None:
        blocker_code = refresh_bundle.blocker.code
        blocker_message = refresh_bundle.blocker.message
        blocker_detail = refresh_bundle.blocker.detail
    elif scoring_bundle.scoring_state == SCORING_BLOCKED:
        blocker_code = scoring_bundle.blocker_code
        blocker_message = scoring_bundle.blocker_message

    return HostedCockpitView(
        selected_date=refresh_bundle.slate_date,
        observed_at_utc=refresh_bundle.observed_at_utc.isoformat(),
        refresh_readiness=refresh_bundle.readiness,
        scoring_state=scoring_bundle.scoring_state,
        nhl_status=_nhl_status(refresh_bundle),
        moneypuck_status=_moneypuck_status(refresh_bundle),
        model_status=_model_status(scoring_bundle),
        calculation_label=CURRENT_CALCULATION_LABEL,
        protected_evidence_label=PROTECTED_EVIDENCE_LABEL,
        blocker_code=blocker_code,
        blocker_message=blocker_message,
        blocker_detail=blocker_detail,
        decision_view=decision,
        scores=tuple(scoring_bundle.games),
    )


def score_by_game_id(view: HostedCockpitView) -> dict[str, HostedGameScore]:
    return {score.game_id: score for score in view.scores}


def _dashboard_game_from_score(
    score: HostedGameScore,
    *,
    display_timezone: str,
) -> _HostedDashboardGame:
    scheduled = _parse_aware_iso(score.scheduled_start_time_utc)
    local = scheduled.astimezone(ZoneInfo(display_timezone))
    hour = local.strftime("%I").lstrip("0") or "0"
    local_label = f"{local.strftime('%a')} {hour}:{local.strftime('%M %p %Z')}"
    utc_label = scheduled.strftime("%Y-%m-%d %H:%M UTC")
    return _HostedDashboardGame(
        nhl_game_id=score.game_id,
        away_team=score.away_team_id,
        home_team=score.home_team_id,
        scheduled_local_label=local_label,
        scheduled_utc_label=utc_label,
        scheduled_start_time_utc=score.scheduled_start_time_utc,
        prediction_status=(
            "CURRENT WEB CALCULATION"
            if score.status == SCORED
            else f"NOT SCORED · {score.reason_code or 'unknown'}"
        ),
        home_model_win_probability=score.model_home_win_probability,
        away_model_win_probability=score.model_away_win_probability,
    )


def _parse_aware_iso(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError("hosted scheduled start must be timezone-aware")
    return parsed


def _nhl_status(bundle: HostedRefreshBundle) -> str:
    if bundle.nhl_coverage is None:
        return "BLOCKED · validated NHL schedule/status coverage unavailable"
    return (
        "READY · "
        f"{len(bundle.nhl_coverage.games)} schedule/status rows · "
        f"{len(bundle.nhl_coverage.covered_dates)} covered dates"
    )


def _moneypuck_status(bundle: HostedRefreshBundle) -> str:
    if bundle.moneypuck_provenance is None:
        return "BLOCKED · validated MoneyPuck source unavailable"
    return f"READY · SHA-256 {bundle.moneypuck_provenance.sha256[:12]}…"


def _model_status(bundle: HostedScoringBundle) -> str:
    if bundle.scoring_state == SCORING_BLOCKED:
        return "BLOCKED · no model probability generated"
    return f"FROZEN · SHA-256 {EXPECTED_FITTED_ARTIFACT_SHA256[:12]}…"
