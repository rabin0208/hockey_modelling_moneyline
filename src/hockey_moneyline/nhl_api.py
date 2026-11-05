"""Bounded read-only NHL Web API schedule access for Issues #6/#9.

Only public unauthenticated schedule endpoints are used. Raw NHL payloads are
parsed in memory and are not persisted by this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hockey_moneyline.errors import NHLApiError, SchemaError
from hockey_moneyline.schema import parse_game_date, parse_start_time

NHL_WEB_BASE_URL = "https://api-web.nhle.com"
SCHEDULE_DATE_PATH = "/v1/schedule/{date}"
SCHEDULE_NOW_PATH = "/v1/schedule/now"
NHL_USER_AGENT = (
    "hockey-moneyline-issue-9-private-dev/0.4 "
    "(+https://github.com/EduardoRasanmar/hockey_moneyline)"
)
GAME_TYPE_NAMES = {1: "preseason", 2: "regular", 3: "playoff"}
FINAL_GAME_STATES = frozenset({"OFF", "FINAL"})
DEFAULT_CONTEXT_LOOKBACK_DAYS = 7
DEFAULT_CONTEXT_LOOKAHEAD_DAYS = 7
DEFAULT_MAX_PREVIOUS_LOOKBACK_DAYS = 45


@dataclass(frozen=True, slots=True)
class NHLScheduleGame:
    nhl_game_id: str
    season: int
    game_type: str
    game_date: date
    scheduled_start_time_utc: datetime
    home_nhl_team_id: int
    away_nhl_team_id: int
    home_team_abbrev: str
    away_team_abbrev: str
    game_state: str
    game_schedule_state: str
    observed_at_utc: datetime

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["game_date"] = self.game_date.isoformat()
        payload["scheduled_start_time_utc"] = self.scheduled_start_time_utc.isoformat()
        payload["observed_at_utc"] = self.observed_at_utc.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class NHLScheduleSnapshot:
    endpoint_path: str
    observed_at_utc: datetime
    games: tuple[NHLScheduleGame, ...]
    covered_dates: frozenset[date] = frozenset()

    def to_summary(self) -> dict[str, Any]:
        covered = sorted(self.covered_dates)
        return {
            "endpoint_path": self.endpoint_path,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "game_count": len(self.games),
            "game_ids": [game.nhl_game_id for game in self.games],
            "coverage_start": covered[0].isoformat() if covered else None,
            "coverage_end": covered[-1].isoformat() if covered else None,
            "covered_date_count": len(covered),
        }


def _aware_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        raise NHLApiError("observed_at_utc must be timezone-aware")
    return result.astimezone(timezone.utc)


def _required(mapping: Mapping[str, Any], field: str, *, game_id: Any) -> Any:
    value = mapping.get(field)
    if value in (None, ""):
        raise NHLApiError(f"NHL schedule game {game_id!r} missing {field}")
    return value


def _parse_game(
    raw: Mapping[str, Any],
    *,
    fallback_date: str | None,
    observed_at_utc: datetime,
) -> NHLScheduleGame:
    game_id = _required(raw, "id", game_id="unknown")
    home = _required(raw, "homeTeam", game_id=game_id)
    away = _required(raw, "awayTeam", game_id=game_id)
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        raise NHLApiError(f"NHL schedule game {game_id} has invalid team payload")

    try:
        game_type_number = int(_required(raw, "gameType", game_id=game_id))
        parsed_date = parse_game_date(raw.get("gameDate") or fallback_date or "")
        parsed_start = parse_start_time(_required(raw, "startTimeUTC", game_id=game_id))
        if parsed_start is None:
            raise NHLApiError(f"NHL schedule game {game_id} missing startTimeUTC")
        season = int(_required(raw, "season", game_id=game_id))
        home_team_id = int(_required(home, "id", game_id=game_id))
        away_team_id = int(_required(away, "id", game_id=game_id))
    except (SchemaError, TypeError, ValueError) as exc:
        raise NHLApiError(f"NHL schedule game {game_id} has invalid identity/timing fields") from exc

    return NHLScheduleGame(
        nhl_game_id=str(game_id),
        season=season,
        game_type=GAME_TYPE_NAMES.get(game_type_number, f"other:{game_type_number}"),
        game_date=parsed_date,
        scheduled_start_time_utc=parsed_start,
        home_nhl_team_id=home_team_id,
        away_nhl_team_id=away_team_id,
        home_team_abbrev=str(_required(home, "abbrev", game_id=game_id)).upper(),
        away_team_abbrev=str(_required(away, "abbrev", game_id=game_id)).upper(),
        game_state=str(_required(raw, "gameState", game_id=game_id)),
        game_schedule_state=str(_required(raw, "gameScheduleState", game_id=game_id)),
        observed_at_utc=observed_at_utc,
    )


def _requested_schedule_date(endpoint_path: str) -> date | None:
    prefix = "/v1/schedule/"
    if not endpoint_path.startswith(prefix):
        return None
    token = endpoint_path[len(prefix) :].split("?", 1)[0].strip("/")
    if token in ("", "now"):
        return None
    try:
        return parse_game_date(token)
    except SchemaError:
        return None


def _merge_game(
    by_id: dict[str, NHLScheduleGame],
    game: NHLScheduleGame,
) -> None:
    existing = by_id.get(game.nhl_game_id)
    if existing is None:
        by_id[game.nhl_game_id] = game
        return
    if existing == game:
        return
    raise NHLApiError(
        f"Conflicting duplicate NHL game id {game.nhl_game_id}: "
        f"{existing.to_dict()} != {game.to_dict()}"
    )


def parse_schedule_payload(
    payload: Mapping[str, Any],
    *,
    endpoint_path: str,
    observed_at_utc: datetime | None = None,
) -> NHLScheduleSnapshot:
    observed = _aware_utc(observed_at_utc)
    by_id: dict[str, NHLScheduleGame] = {}
    covered_dates: set[date] = set()
    requested_date = _requested_schedule_date(endpoint_path)
    if requested_date is not None:
        # A direct schedule/{date} request is authoritative evidence that this
        # calendar date was queried, even when there are zero games that day.
        covered_dates.add(requested_date)

    weeks = payload.get("gameWeek")
    if isinstance(weeks, list):
        groups: list[tuple[str | None, Any]] = []
        for week in weeks:
            if not isinstance(week, Mapping):
                continue
            raw_date = week.get("date")
            if raw_date not in (None, ""):
                try:
                    covered_dates.add(parse_game_date(raw_date))
                except SchemaError as exc:
                    raise NHLApiError(f"NHL schedule group has invalid date {raw_date!r}") from exc
            groups.append((str(raw_date) if raw_date not in (None, "") else None, week.get("games", [])))
    elif isinstance(payload.get("games"), list):
        groups = [(requested_date.isoformat() if requested_date else None, payload["games"])]
    else:
        raise NHLApiError("NHL schedule payload has neither gameWeek nor games")

    for fallback_date, games in groups:
        if not isinstance(games, list):
            raise NHLApiError("NHL schedule games must be a list")
        for raw in games:
            if not isinstance(raw, Mapping):
                raise NHLApiError("NHL schedule game entry must be an object")
            game = _parse_game(raw, fallback_date=fallback_date, observed_at_utc=observed)
            _merge_game(by_id, game)

    parsed = sorted(
        by_id.values(),
        key=lambda game: (game.scheduled_start_time_utc, game.nhl_game_id),
    )
    return NHLScheduleSnapshot(
        endpoint_path=endpoint_path,
        observed_at_utc=observed,
        games=tuple(parsed),
        covered_dates=frozenset(covered_dates),
    )


def merge_schedule_snapshots(
    snapshots: Iterable[NHLScheduleSnapshot],
) -> NHLScheduleSnapshot:
    items = list(snapshots)
    if not items:
        raise NHLApiError("Cannot assemble an empty NHL schedule snapshot set")
    observed = items[0].observed_at_utc
    by_id: dict[str, NHLScheduleGame] = {}
    covered_dates: set[date] = set()
    for snapshot in items:
        if snapshot.observed_at_utc != observed:
            raise NHLApiError("Schedule snapshots must share one observation timestamp")
        covered_dates.update(snapshot.covered_dates)
        for game in snapshot.games:
            _merge_game(by_id, game)
    games = tuple(
        sorted(
            by_id.values(),
            key=lambda game: (game.scheduled_start_time_utc, game.nhl_game_id),
        )
    )
    return NHLScheduleSnapshot(
        endpoint_path="assembled:/v1/schedule/{date}",
        observed_at_utc=observed,
        games=games,
        covered_dates=frozenset(covered_dates),
    )


def _read_json(
    url: str,
    *,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> Mapping[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": NHL_USER_AGENT, "Accept": "application/json"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise NHLApiError(f"Unexpected HTTP {status} from {url}")
            payload = json.load(response)
    except HTTPError as exc:
        raise NHLApiError(f"HTTP {exc.code} fetching {url}") from exc
    except URLError as exc:
        raise NHLApiError(f"Network failure fetching {url}: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise NHLApiError(f"Invalid JSON from {url}") from exc
    if not isinstance(payload, Mapping):
        raise NHLApiError(f"NHL API payload from {url} is not an object")
    return payload


def fetch_schedule(
    target_date: str | date | None = None,
    *,
    observed_at_utc: datetime | None = None,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> NHLScheduleSnapshot:
    if target_date is None:
        endpoint = SCHEDULE_NOW_PATH
    else:
        endpoint = SCHEDULE_DATE_PATH.format(date=parse_game_date(target_date).isoformat())
    payload = _read_json(NHL_WEB_BASE_URL + endpoint, timeout=timeout, opener=opener)
    return parse_schedule_payload(
        payload,
        endpoint_path=endpoint,
        observed_at_utc=observed_at_utc,
    )


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _target_game(snapshot: NHLScheduleSnapshot, target_nhl_game_id: str) -> NHLScheduleGame:
    matches = [game for game in snapshot.games if game.nhl_game_id == str(target_nhl_game_id)]
    if len(matches) != 1:
        raise NHLApiError(
            f"Target NHL game {target_nhl_game_id!r} was not uniquely present in schedule data"
        )
    return matches[0]


def _has_prior_final(snapshot: NHLScheduleSnapshot, target: NHLScheduleGame, team_id: int) -> bool:
    return any(
        game.scheduled_start_time_utc < target.scheduled_start_time_utc
        and game.game_state in FINAL_GAME_STATES
        and team_id in (game.home_nhl_team_id, game.away_nhl_team_id)
        for game in snapshot.games
    )


def assemble_schedule_window(
    target_date: str | date,
    target_nhl_game_id: str,
    *,
    observed_at_utc: datetime | None = None,
    lookback_days: int = DEFAULT_CONTEXT_LOOKBACK_DAYS,
    lookahead_days: int = DEFAULT_CONTEXT_LOOKAHEAD_DAYS,
    max_previous_lookback_days: int = DEFAULT_MAX_PREVIOUS_LOOKBACK_DAYS,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> NHLScheduleSnapshot:
    """Assemble explicit calendar coverage for leakage-safe schedule context.

    Every calendar date required by the ±7-day context is explicitly covered by
    a schedule/{date} response (directly or as an explicit gameWeek date). If no
    previous final game for either target team is present in that base window,
    the assembler extends backward one calendar date at a time, bounded by
    ``max_previous_lookback_days``. Failure to establish a previous game raises
    NHLApiError instead of allowing downstream rest/context to masquerade as 0.
    """

    if lookback_days < 0 or lookahead_days < 0 or max_previous_lookback_days < lookback_days:
        raise NHLApiError("Invalid schedule-window bounds")

    observed = _aware_utc(observed_at_utc)
    requested_target_date = parse_game_date(target_date)
    snapshots: list[NHLScheduleSnapshot] = []

    def assembled() -> NHLScheduleSnapshot:
        return merge_schedule_snapshots(snapshots)

    def fetch_day(day: date) -> None:
        current = assembled() if snapshots else None
        if current is not None and day in current.covered_dates:
            return
        snapshots.append(
            fetch_schedule(
                day,
                observed_at_utc=observed,
                timeout=timeout,
                opener=opener,
            )
        )

    fetch_day(requested_target_date)
    current = assembled()
    target = _target_game(current, target_nhl_game_id)
    if target.game_date != requested_target_date:
        raise NHLApiError(
            f"Target NHL game {target_nhl_game_id} date {target.game_date} "
            f"does not match requested date {requested_target_date}"
        )
    if observed >= target.scheduled_start_time_utc:
        raise NHLApiError("Schedule window must be observed before target start")

    required_start = target.game_date - timedelta(days=lookback_days)
    required_end = target.game_date + timedelta(days=lookahead_days)
    for day in _date_range(required_start, required_end):
        fetch_day(day)

    current = assembled()
    target = _target_game(current, target_nhl_game_id)
    target_teams = (target.home_nhl_team_id, target.away_nhl_team_id)

    cursor = required_start - timedelta(days=1)
    earliest = target.game_date - timedelta(days=max_previous_lookback_days)
    while not all(_has_prior_final(current, target, team_id) for team_id in target_teams):
        if cursor < earliest:
            missing_teams = [
                team_id
                for team_id in target_teams
                if not _has_prior_final(current, target, team_id)
            ]
            raise NHLApiError(
                "Could not establish a previous final game within bounded schedule "
                f"lookback for NHL team ids {missing_teams}"
            )
        fetch_day(cursor)
        current = assembled()
        target = _target_game(current, target_nhl_game_id)
        cursor -= timedelta(days=1)

    return current
