"""Resolve MoneyPuck tied regular-season scores via NHL final shootout winners.

MoneyPuck ``situation=all`` keeps regulation+OT goals only, so shootouts appear
as ties and have no winner column. NHL ``/v1/score/{date}`` exposes the final
score and ``gameOutcome.lastPeriodType == "SO"``. Results are cached so live
scoring only fetches newly missing dates.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import urlopen

from hockey_moneyline.errors import NHLApiError, SchemaError
from hockey_moneyline.nhl_api import NHL_WEB_BASE_URL, _read_json
from hockey_moneyline.normalize import (
    _game_type_from_row,
    _int_score,
    _row_season,
    canonical_moneypuck_team_code,
)
from hockey_moneyline.provenance import MONEYPUCK_SITUATION, REGULAR_SEASON_GAME_TYPE
from hockey_moneyline.schema import parse_game_date

DEFAULT_SHOOTOUT_CACHE = Path("data/cache/nhl/shootout_winners.json")
SCORE_DATE_PATH = "/v1/score/{date}"
FINAL_STATES = frozenset({"OFF", "FINAL"})


@dataclass(frozen=True, slots=True)
class TiedMoneyPuckGame:
    game_id: str
    game_date: date
    season: int
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int


@dataclass(frozen=True, slots=True)
class ShootoutWinner:
    game_id: str
    game_date: date
    home_team_abbrev: str
    away_team_abbrev: str
    home_win: int
    nhl_home_score: int
    nhl_away_score: int
    last_period_type: str = "SO"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["game_date"] = self.game_date.isoformat()
        return payload

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ShootoutWinner":
        home_win = int(row["home_win"])
        if home_win not in (0, 1):
            raise SchemaError(f"Shootout home_win must be 0 or 1, got {home_win!r}")
        last_period = str(row.get("last_period_type") or "SO").upper()
        if last_period != "SO":
            raise SchemaError(
                f"Cached shootout resolution for {row.get('game_id')} has "
                f"last_period_type {last_period!r}, expected SO"
            )
        return cls(
            game_id=str(row["game_id"]),
            game_date=parse_game_date(row["game_date"]),
            home_team_abbrev=str(row["home_team_abbrev"]).upper(),
            away_team_abbrev=str(row["away_team_abbrev"]).upper(),
            home_win=home_win,
            nhl_home_score=int(row["nhl_home_score"]),
            nhl_away_score=int(row["nhl_away_score"]),
            last_period_type=last_period,
        )


def list_tied_moneypuck_games(
    path: Path,
    seasons: Iterable[int] | None = None,
) -> list[TiedMoneyPuckGame]:
    """Return regular-season HOME rows whose MoneyPuck situation=all scores are tied."""

    allowed = set(seasons) if seasons is not None else None
    tied: dict[str, TiedMoneyPuckGame] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SchemaError(f"MoneyPuck file has no header: {path}")
            for row in reader:
                if str(row.get("situation") or "").strip() != MONEYPUCK_SITUATION:
                    continue
                if _game_type_from_row(row) != REGULAR_SEASON_GAME_TYPE:
                    continue
                if str(row.get("home_or_away") or "").strip().upper() != "HOME":
                    continue
                season = _row_season(row)
                if allowed is not None and season not in allowed:
                    continue
                game_id = str(row.get("gameId") or "")
                if not game_id:
                    raise SchemaError("MoneyPuck tied-row scan found a row without gameId")
                home_score = _int_score(row.get("goalsFor"), game_id=game_id, field="goalsFor")
                away_score = _int_score(
                    row.get("goalsAgainst"), game_id=game_id, field="goalsAgainst"
                )
                if home_score != away_score:
                    continue
                game = TiedMoneyPuckGame(
                    game_id=game_id,
                    game_date=parse_game_date(row["gameDate"]),
                    season=season,
                    home_team_id=canonical_moneypuck_team_code(row.get("team"), season),
                    away_team_id=canonical_moneypuck_team_code(
                        row.get("opposingTeam"), season
                    ),
                    home_score=home_score,
                    away_score=away_score,
                )
                existing = tied.get(game_id)
                if existing is not None and existing != game:
                    raise SchemaError(f"Conflicting tied MoneyPuck payloads for {game_id}")
                tied[game_id] = game
    except OSError as exc:
        raise SchemaError(f"Unreadable MoneyPuck file {path}") from exc
    return sorted(tied.values(), key=lambda item: (item.game_date, item.game_id))


def load_shootout_winner_cache(path: Path = DEFAULT_SHOOTOUT_CACHE) -> dict[str, ShootoutWinner]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"Unreadable shootout winner cache {path}") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"Shootout winner cache must be a JSON object: {path}")
    winners: dict[str, ShootoutWinner] = {}
    for game_id, row in payload.items():
        if not isinstance(row, Mapping):
            raise SchemaError(f"Shootout cache entry {game_id!r} must be an object")
        winner = ShootoutWinner.from_mapping({**row, "game_id": row.get("game_id", game_id)})
        if winner.game_id != str(game_id):
            raise SchemaError(
                f"Shootout cache key {game_id!r} disagrees with payload game_id "
                f"{winner.game_id!r}"
            )
        winners[winner.game_id] = winner
    return winners


def save_shootout_winner_cache(
    winners: Mapping[str, ShootoutWinner],
    path: Path = DEFAULT_SHOOTOUT_CACHE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        game_id: winners[game_id].to_dict()
        for game_id in sorted(winners)
    }
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_score_game(raw: Mapping[str, Any]) -> ShootoutWinner | None:
    game_id = str(raw.get("id") or raw.get("gamePk") or "")
    if not game_id:
        return None
    state = str(raw.get("gameState") or "").upper()
    if state not in FINAL_STATES:
        return None
    outcome = raw.get("gameOutcome")
    if not isinstance(outcome, Mapping):
        return None
    last_period = str(outcome.get("lastPeriodType") or "").upper()
    if last_period != "SO":
        return None
    home = raw.get("homeTeam")
    away = raw.get("awayTeam")
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        raise NHLApiError(f"NHL score game {game_id} has invalid team payload")
    try:
        home_score = int(home["score"])
        away_score = int(away["score"])
        game_date = parse_game_date(raw.get("gameDate") or "")
    except (KeyError, TypeError, ValueError, SchemaError) as exc:
        raise NHLApiError(f"NHL score game {game_id} has invalid score/date fields") from exc
    if home_score == away_score:
        raise NHLApiError(
            f"NHL shootout game {game_id} still has tied final scores {home_score}-{away_score}"
        )
    return ShootoutWinner(
        game_id=game_id,
        game_date=game_date,
        home_team_abbrev=str(home.get("abbrev") or "").upper(),
        away_team_abbrev=str(away.get("abbrev") or "").upper(),
        home_win=1 if home_score > away_score else 0,
        nhl_home_score=home_score,
        nhl_away_score=away_score,
        last_period_type=last_period,
    )


def fetch_score_board(
    target_date: str | date,
    *,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, ShootoutWinner]:
    day = parse_game_date(target_date)
    endpoint = SCORE_DATE_PATH.format(date=day.isoformat())
    payload = _read_json(NHL_WEB_BASE_URL + endpoint, timeout=timeout, opener=opener)
    games = payload.get("games")
    if not isinstance(games, list):
        raise NHLApiError(f"NHL score payload for {day.isoformat()} is missing games[]")
    winners: dict[str, ShootoutWinner] = {}
    for raw in games:
        if not isinstance(raw, Mapping):
            raise NHLApiError(f"NHL score game entry for {day.isoformat()} must be an object")
        parsed = _parse_score_game(raw)
        if parsed is None:
            continue
        if parsed.game_date != day:
            # Score boards can include adjacent-date spill; keep the API date.
            parsed = ShootoutWinner(
                game_id=parsed.game_id,
                game_date=day,
                home_team_abbrev=parsed.home_team_abbrev,
                away_team_abbrev=parsed.away_team_abbrev,
                home_win=parsed.home_win,
                nhl_home_score=parsed.nhl_home_score,
                nhl_away_score=parsed.nhl_away_score,
                last_period_type=parsed.last_period_type,
            )
        winners[parsed.game_id] = parsed
    return winners


def _validate_resolution(candidate: TiedMoneyPuckGame, winner: ShootoutWinner) -> None:
    if winner.game_id != candidate.game_id:
        raise SchemaError(
            f"Shootout resolution game_id {winner.game_id} != tied game {candidate.game_id}"
        )
    if winner.home_team_abbrev != candidate.home_team_id:
        raise SchemaError(
            f"Shootout resolution home team {winner.home_team_abbrev} disagrees with "
            f"MoneyPuck {candidate.home_team_id} for {candidate.game_id}"
        )
    if winner.away_team_abbrev != candidate.away_team_id:
        raise SchemaError(
            f"Shootout resolution away team {winner.away_team_abbrev} disagrees with "
            f"MoneyPuck {candidate.away_team_id} for {candidate.game_id}"
        )
    if winner.last_period_type != "SO":
        raise SchemaError(
            f"Tied MoneyPuck game {candidate.game_id} resolved with "
            f"{winner.last_period_type!r}, not SO"
        )


def ensure_shootout_winners(
    candidates: Sequence[TiedMoneyPuckGame],
    *,
    cache_path: Path = DEFAULT_SHOOTOUT_CACHE,
    fetch_missing: bool = True,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
    pause_seconds: float = 0.05,
) -> dict[str, ShootoutWinner]:
    """Return NHL shootout winners for tied MoneyPuck games, fetching into cache."""

    winners = load_shootout_winner_cache(cache_path)
    needed = [game for game in candidates if game.game_id not in winners]
    if needed and fetch_missing:
        by_date: dict[date, list[TiedMoneyPuckGame]] = {}
        for game in needed:
            by_date.setdefault(game.game_date, []).append(game)
        for index, day in enumerate(sorted(by_date)):
            board = fetch_score_board(day, timeout=timeout, opener=opener)
            day_updated = False
            for game in by_date[day]:
                winner = board.get(game.game_id)
                if winner is None:
                    continue
                _validate_resolution(game, winner)
                winners[game.game_id] = winner
                day_updated = True
            if day_updated:
                save_shootout_winner_cache(winners, cache_path)
            if pause_seconds > 0 and index + 1 < len(by_date):
                time.sleep(pause_seconds)

    resolved: dict[str, ShootoutWinner] = {}
    for game in candidates:
        winner = winners.get(game.game_id)
        if winner is None:
            continue
        _validate_resolution(game, winner)
        resolved[game.game_id] = winner
    return resolved


def home_win_by_game_id(winners: Mapping[str, ShootoutWinner]) -> dict[str, int]:
    return {game_id: winner.home_win for game_id, winner in winners.items()}
