"""Issue #107 PlayNow odds composition for the private hosted cockpit.

This module is intentionally thin. Provider acquisition and quote validation stay
owned by Issue #98 / PR #106, persistence validation stays owned by Issue #87,
and market math/recommendation semantics stay owned by Issue #78/#85.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Mapping

from hockey_moneyline.decision_cockpit import ForwardOddsMarketSnapshotReader
from hockey_moneyline.errors import ForwardOddsArchiveError
from hockey_moneyline.hosted_cockpit import (
    HostedCockpitRun,
    HostedCockpitView,
    build_hosted_cockpit_view,
)
from hockey_moneyline.hosted_refresh import FUT_GAME_STATE, READY as REFRESH_READY
from hockey_moneyline.the_odds_api_playnow import (
    NO_DATA,
    PLAYNOW_BOOKMAKER_KEY,
    READY,
    OddsApiUsage,
    PlayNowAcquisition,
    commit_playnow_quotes,
    fetch_playnow_current_odds,
    prepare_playnow_snapshots,
)

PLAYNOW_DISPLAY_NAME = "PlayNow"
RUNTIME_STORE_FILENAME = "forward_odds_snapshots.jsonl"


@dataclass(frozen=True, slots=True)
class HostedOddsRefresh:
    """One explicit operator-triggered current PlayNow acquisition."""

    status: str
    acquisition: PlayNowAcquisition
    store_path: str
    written: tuple[dict[str, Any], ...]
    requested_game_ids: tuple[str, ...]

    @property
    def ready_count(self) -> int:
        return len(self.acquisition.ready)

    @property
    def request_count(self) -> int:
        return self.acquisition.request_count

    @property
    def missing_game_ids(self) -> tuple[str, ...]:
        return self.acquisition.schedule_game_ids_without_ready_quote

    @property
    def latest_observation_utc(self) -> str | None:
        observations = sorted(
            quote.observation_utc
            for quote in self.acquisition.ready
            if quote.observation_utc is not None
        )
        return observations[-1] if observations else None


def create_runtime_forward_odds_store() -> Path:
    """Create one session/runtime-local Issue #87 archive path.

    Streamlit Community Cloud filesystem state is runtime-local and must not be
    represented as durable protected evidence. A fresh path per Refresh Odds
    action also prevents a failed current acquisition from silently displaying
    an older session snapshot as if it came from the latest request.
    """

    directory = Path(mkdtemp(prefix="hockey-playnow-current-"))
    return directory / RUNTIME_STORE_FILENAME


def hosted_playnow_schedule(run: HostedCockpitRun) -> tuple[dict[str, str], ...]:
    """Project validated #91 FUT slate identity into the #98 schedule contract."""

    if not isinstance(run, HostedCockpitRun):
        raise ForwardOddsArchiveError(
            "Refresh Odds requires a hosted cockpit run from Refresh Data"
        )
    if run.refresh.readiness != REFRESH_READY or not run.is_successful:
        raise ForwardOddsArchiveError(
            "Refresh Data must be READY before PlayNow odds can be refreshed"
        )

    rows: list[dict[str, str]] = []
    for target in run.refresh.slate_targets:
        if target.game_state != FUT_GAME_STATE:
            continue
        rows.append(
            {
                "nhl_game_id": target.game_id,
                "home_team": target.home_team_id,
                "away_team": target.away_team_id,
                "scheduled_puck_drop_utc": target.scheduled_start_time_utc.isoformat(),
            }
        )
    return tuple(rows)


def refresh_playnow_odds_for_schedule(
    schedule: tuple[Mapping[str, str], ...] | tuple[dict[str, str], ...],
    *,
    store_path: Path | str,
    fetcher: Callable[..., tuple[Any, OddsApiUsage]] = fetch_playnow_current_odds,
    fetch_kwargs: Mapping[str, Any] | None = None,
) -> HostedOddsRefresh:
    """Fetch once for an already-built #98 schedule, then write via #87 only.

    An empty schedule makes zero provider requests. A non-empty schedule makes
    exactly one ``fetcher`` call.
    """

    rows = tuple(dict(row) for row in schedule)
    requested_game_ids = tuple(str(row["nhl_game_id"]) for row in rows)
    if not rows:
        acquisition = PlayNowAcquisition(
            quotes=(),
            schedule_game_ids_without_ready_quote=(),
            usage=None,
            request_count=0,
        )
        return HostedOddsRefresh(
            status=NO_DATA,
            acquisition=acquisition,
            store_path=str(store_path),
            written=(),
            requested_game_ids=(),
        )

    payload, usage = fetcher(**dict(fetch_kwargs or {}))
    acquisition = prepare_playnow_snapshots(payload, rows, usage=usage)
    written = tuple(
        commit_playnow_quotes(
            acquisition,
            store=store_path,
            confirm=True,
        )
    )
    return HostedOddsRefresh(
        status=READY if acquisition.ready else NO_DATA,
        acquisition=acquisition,
        store_path=str(store_path),
        written=written,
        requested_game_ids=requested_game_ids,
    )


def refresh_playnow_odds(
    run: HostedCockpitRun,
    *,
    store_path: Path | str,
    fetcher: Callable[..., tuple[Any, OddsApiUsage]] = fetch_playnow_current_odds,
    fetch_kwargs: Mapping[str, Any] | None = None,
) -> HostedOddsRefresh:
    """Fetch once, validate via #98/#87, and write READY quotes via #87 only."""

    schedule = hosted_playnow_schedule(run)
    return refresh_playnow_odds_for_schedule(
        schedule,
        store_path=store_path,
        fetcher=fetcher,
        fetch_kwargs=fetch_kwargs,
    )


def build_playnow_market_view(
    run: HostedCockpitRun,
    *,
    store_path: Path | str,
) -> HostedCockpitView:
    """Read current attempt through #87/#85 and let #78 build market math."""

    reader = ForwardOddsMarketSnapshotReader(
        store_path=store_path,
        book_source=PLAYNOW_BOOKMAKER_KEY,
    )
    return build_hosted_cockpit_view(
        refresh_bundle=run.refresh,
        scoring_bundle=run.scoring,
        market_reader=reader,
    )
