"""NHL lastPeriodType cache for regulation / overtime / shootout labels.

MoneyPuck ``situation=all`` includes OT goals but has no OT flag. NHL
``/v1/score/{date}`` exposes ``gameOutcome.lastPeriodType`` in {REG, OT, SO}.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import urlopen

from hockey_moneyline.errors import NHLApiError, SchemaError
from hockey_moneyline.nhl_api import NHL_WEB_BASE_URL, _read_json
from hockey_moneyline.schema import NormalizedGame, parse_game_date

DEFAULT_PERIOD_CACHE = Path("data/cache/nhl/last_period_types.json")
SCORE_DATE_PATH = "/v1/score/{date}"
FINAL_STATES = frozenset({"OFF", "FINAL"})
VALID_PERIOD_TYPES = frozenset({"REG", "OT", "SO"})


@dataclass(frozen=True, slots=True)
class PeriodOutcome:
    game_id: str
    game_date: date
    last_period_type: str
    home_team_abbrev: str
    away_team_abbrev: str
    home_win: int
    nhl_home_score: int
    nhl_away_score: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["game_date"] = self.game_date.isoformat()
        return payload

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PeriodOutcome":
        last_period = str(row.get("last_period_type") or "").upper()
        if last_period not in VALID_PERIOD_TYPES:
            raise SchemaError(
                f"Period cache entry {row.get('game_id')!r} has invalid "
                f"last_period_type {last_period!r}"
            )
        home_win = int(row["home_win"])
        if home_win not in (0, 1):
            raise SchemaError(f"Period home_win must be 0 or 1, got {home_win!r}")
        return cls(
            game_id=str(row["game_id"]),
            game_date=parse_game_date(row["game_date"]),
            last_period_type=last_period,
            home_team_abbrev=str(row["home_team_abbrev"]).upper(),
            away_team_abbrev=str(row["away_team_abbrev"]).upper(),
            home_win=home_win,
            nhl_home_score=int(row["nhl_home_score"]),
            nhl_away_score=int(row["nhl_away_score"]),
        )


def load_period_outcome_cache(path: Path = DEFAULT_PERIOD_CACHE) -> dict[str, PeriodOutcome]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"Unreadable period outcome cache {path}") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"Period outcome cache must be a JSON object: {path}")
    outcomes: dict[str, PeriodOutcome] = {}
    for game_id, row in payload.items():
        if not isinstance(row, Mapping):
            raise SchemaError(f"Period cache entry {game_id!r} must be an object")
        outcome = PeriodOutcome.from_mapping({**row, "game_id": row.get("game_id", game_id)})
        if outcome.game_id != str(game_id):
            raise SchemaError(
                f"Period cache key {game_id!r} disagrees with payload game_id "
                f"{outcome.game_id!r}"
            )
        outcomes[outcome.game_id] = outcome
    return outcomes


def save_period_outcome_cache(
    outcomes: Mapping[str, PeriodOutcome],
    path: Path = DEFAULT_PERIOD_CACHE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {game_id: outcomes[game_id].to_dict() for game_id in sorted(outcomes)}
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_score_game(raw: Mapping[str, Any], board_date: date) -> PeriodOutcome | None:
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
    if last_period not in VALID_PERIOD_TYPES:
        return None
    home = raw.get("homeTeam")
    away = raw.get("awayTeam")
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        raise NHLApiError(f"NHL score game {game_id} has invalid team payload")
    try:
        home_score = int(home["score"])
        away_score = int(away["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NHLApiError(f"NHL score game {game_id} has invalid scores") from exc
    if home_score == away_score:
        # Final NHL boards should not stay tied; skip incomplete payloads.
        return None
    return PeriodOutcome(
        game_id=game_id,
        game_date=board_date,
        last_period_type=last_period,
        home_team_abbrev=str(home.get("abbrev") or "").upper(),
        away_team_abbrev=str(away.get("abbrev") or "").upper(),
        home_win=1 if home_score > away_score else 0,
        nhl_home_score=home_score,
        nhl_away_score=away_score,
    )


def fetch_score_board_periods(
    target_date: str | date,
    *,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, PeriodOutcome]:
    day = parse_game_date(target_date)
    endpoint = SCORE_DATE_PATH.format(date=day.isoformat())
    payload = _read_json(NHL_WEB_BASE_URL + endpoint, timeout=timeout, opener=opener)
    games = payload.get("games")
    if not isinstance(games, list):
        raise NHLApiError(f"NHL score payload for {day.isoformat()} is missing games[]")
    outcomes: dict[str, PeriodOutcome] = {}
    for raw in games:
        if not isinstance(raw, Mapping):
            raise NHLApiError(f"NHL score game entry for {day.isoformat()} must be an object")
        parsed = _parse_score_game(raw, day)
        if parsed is None:
            continue
        outcomes[parsed.game_id] = parsed
    return outcomes


def ensure_period_outcomes(
    games: Sequence[NormalizedGame],
    *,
    cache_path: Path = DEFAULT_PERIOD_CACHE,
    fetch_missing: bool = True,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
    pause_seconds: float = 0.05,
) -> dict[str, PeriodOutcome]:
    """Ensure lastPeriodType coverage for the given games; fetch missing dates."""

    outcomes = load_period_outcome_cache(cache_path)
    missing_dates = sorted(
        {game.game_date for game in games if game.game_id not in outcomes}
    )
    if missing_dates and fetch_missing:
        dirty = False
        for index, day in enumerate(missing_dates):
            board = fetch_score_board_periods(day, timeout=timeout, opener=opener)
            if board:
                outcomes.update(board)
                dirty = True
            # Checkpoint periodically so long backfills survive interruption.
            if dirty and ((index + 1) % 25 == 0 or index + 1 == len(missing_dates)):
                save_period_outcome_cache(outcomes, cache_path)
                dirty = False
            if pause_seconds > 0 and index + 1 < len(missing_dates):
                time.sleep(pause_seconds)
        if dirty:
            save_period_outcome_cache(outcomes, cache_path)
    return {game.game_id: outcomes[game.game_id] for game in games if game.game_id in outcomes}


def annotate_period_outcomes(
    games: Sequence[NormalizedGame],
    outcomes: Mapping[str, PeriodOutcome],
) -> list[NormalizedGame]:
    """Attach decided_in_overtime from NHL lastPeriodType; keep shootout flags."""

    annotated: list[NormalizedGame] = []
    for game in games:
        outcome = outcomes.get(game.game_id)
        if outcome is None:
            if game.decided_in_shootout:
                annotated.append(replace(game, decided_in_overtime=False))
                continue
            # Uncached non-SO games are treated as regulation.
            annotated.append(replace(game, decided_in_overtime=False))
            continue
        period = outcome.last_period_type
        if game.decided_in_shootout and period not in {"SO", "OT"}:
            # SO MoneyPuck ties should resolve as SO; tolerate rare board drift.
            pass
        if game.decided_in_shootout and period == "OT":
            raise SchemaError(
                f"Game {game.game_id} is marked shootout but NHL lastPeriodType=OT"
            )
        annotated.append(
            replace(
                game,
                decided_in_overtime=(period == "OT" and not game.decided_in_shootout),
            )
        )
    return annotated


def went_to_extra_time(game: NormalizedGame) -> bool:
    return bool(game.decided_in_overtime or game.decided_in_shootout)
