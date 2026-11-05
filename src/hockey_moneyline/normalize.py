"""Normalize MoneyPuck published team-game rows into the canonical contract."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from hockey_moneyline.errors import SchemaError
from hockey_moneyline.provenance import (
    DEFAULT_SEASONS,
    MONEYPUCK_SITUATION,
    REGULAR_SEASON_GAME_TYPE,
    SOURCE_ID,
    season_from_moneypuck_year,
)
from hockey_moneyline.schema import (
    NormalizedGame,
    game_from_mapping,
    home_win_from_scores,
    parse_game_date,
    validate_games,
)

# MoneyPuck's career files use both dotted historical display abbreviations and
# current NHL tricodes. These four aliases are punctuation/code variants of the
# same clubs, not fuzzy franchise matching. They are normalized before feature
# state is keyed so history is continuous across a source-code transition.
MONEYPUCK_TEAM_ALIASES = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}

# Explicit target-era identities for Issue #11. ARI and UTA intentionally stay
# distinct: Issue #11 does not silently carry team state across relocation.
TARGET_ERA_TEAM_CODES = frozenset(
    {
        "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI",
        "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
        "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA",
        "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG",
        "WSH",
    }
)
TARGET_ERA_SEASONS = frozenset(DEFAULT_SEASONS)


@dataclass
class NormalizeReport:
    games: list[NormalizedGame]
    skipped_tied_scores: int = 0
    skipped_tied_scores_by_season: dict[int, int] = field(default_factory=dict)
    normalized_games_by_season: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


REQUIRED_SOURCE_COLUMNS = (
    "team",
    "season",
    "gameId",
    "opposingTeam",
    "home_or_away",
    "gameDate",
    "situation",
    "goalsFor",
    "goalsAgainst",
)


def canonical_moneypuck_team_code(code: Any, season: int) -> str:
    raw = str(code or "").strip().upper()
    if not raw:
        raise SchemaError(f"MoneyPuck season {season} row is missing a team code")
    canonical = MONEYPUCK_TEAM_ALIASES.get(raw, raw)
    if season in TARGET_ERA_SEASONS and canonical not in TARGET_ERA_TEAM_CODES:
        raise SchemaError(
            f"Unknown MoneyPuck team identity {raw!r} in target-era season {season}; "
            "add an explicit mapping rather than fuzzy-matching"
        )
    return canonical


def _truthy_playoff(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _game_type_from_row(row: Mapping[str, Any]) -> str:
    game_id = str(row.get("gameId") or "")
    if len(game_id) >= 6 and game_id[4:6] == "02" and not _truthy_playoff(row.get("playoffGame")):
        return REGULAR_SEASON_GAME_TYPE
    return "playoff" if _truthy_playoff(row.get("playoffGame")) or (len(game_id) >= 6 and game_id[4:6] == "03") else "other"


def _int_score(value: Any, *, game_id: str, field: str) -> int:
    if value in (None, ""):
        raise SchemaError(f"Completed game {game_id} is missing {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Game {game_id} has a non-numeric {field}: {value!r}") from exc
    if number != int(number):
        raise SchemaError(f"Game {game_id} has a non-integer {field}: {value!r}")
    return int(number)


def _row_season(row: Mapping[str, Any]) -> int:
    try:
        return season_from_moneypuck_year(int(row["season"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"MoneyPuck row has invalid season: {row.get('season')!r}") from exc


def _home_row_to_mapping(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    game_id = str(row.get("gameId") or "")
    if not game_id:
        raise SchemaError("Completed game is missing gameId")
    season = _row_season(row)
    home = canonical_moneypuck_team_code(row.get("team"), season)
    away = canonical_moneypuck_team_code(row.get("opposingTeam"), season)
    if home == away:
        raise SchemaError(f"Game {game_id} maps home and away to the same team {home}")
    home_score = _int_score(row.get("goalsFor"), game_id=game_id, field="goalsFor")
    away_score = _int_score(row.get("goalsAgainst"), game_id=game_id, field="goalsAgainst")
    return {
        "game_id": game_id,
        "game_date": parse_game_date(row["gameDate"]).isoformat(),
        "season": season,
        "game_type": _game_type_from_row(row),
        "home_team_id": home,
        "home_team_name": home,
        "away_team_id": away,
        "away_team_name": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win_from_scores(home_score, away_score),
        "home_team_abbrev": home,
        "away_team_abbrev": away,
        "source": source,
        "start_time_utc": None,
    }


def _validate_away_row(game_id: str, mapped: Mapping[str, Any], away: Mapping[str, Any]) -> None:
    away_season = _row_season(away)
    if away_season != int(mapped["season"]):
        raise SchemaError(f"Conflicting HOME/AWAY season for game_id {game_id}")
    if parse_game_date(away.get("gameDate") or "").isoformat() != mapped["game_date"]:
        raise SchemaError(f"Conflicting HOME/AWAY date for game_id {game_id}")
    away_team = canonical_moneypuck_team_code(away.get("team"), away_season)
    away_opponent = canonical_moneypuck_team_code(away.get("opposingTeam"), away_season)
    if away_team != mapped["away_team_id"] or away_opponent != mapped["home_team_id"]:
        raise SchemaError(f"Conflicting HOME/AWAY team identity for game_id {game_id}")
    away_for = _int_score(away.get("goalsFor"), game_id=game_id, field="away goalsFor")
    away_against = _int_score(away.get("goalsAgainst"), game_id=game_id, field="away goalsAgainst")
    if away_for != mapped["away_score"] or away_against != mapped["home_score"]:
        raise SchemaError(f"Conflicting HOME/AWAY scores for game_id {game_id}")


def games_from_moneypuck_rows(
    rows: Iterable[Mapping[str, Any]],
    seasons: Iterable[int] | None = None,
    source: str = SOURCE_ID,
) -> NormalizeReport:
    allowed = set(seasons) if seasons is not None else None
    homes: dict[str, dict[str, Any]] = {}
    aways: dict[str, Mapping[str, Any]] = {}
    skipped_by_season: Counter[int] = Counter()

    for row in rows:
        if str(row.get("situation") or "").strip() != MONEYPUCK_SITUATION:
            continue
        game_type = _game_type_from_row(row)
        if game_type != REGULAR_SEASON_GAME_TYPE:
            continue
        season = _row_season(row)
        if allowed is not None and season not in allowed:
            continue
        side = str(row.get("home_or_away") or "").strip().upper()
        game_id = str(row.get("gameId") or "")
        if side == "HOME":
            home_score = _int_score(row.get("goalsFor"), game_id=game_id, field="goalsFor")
            away_score = _int_score(row.get("goalsAgainst"), game_id=game_id, field="goalsAgainst")
            if home_score == away_score:
                skipped_by_season[season] += 1
                continue
            mapped = _home_row_to_mapping(row, source)
            existing = homes.get(game_id)
            if existing is not None and existing != mapped:
                raise SchemaError(f"Conflicting payloads for game_id {game_id}")
            homes[game_id] = mapped
        elif side == "AWAY":
            aways[game_id] = row

    if not homes:
        raise SchemaError("No regular-season MoneyPuck HOME rows were usable")

    games: list[NormalizedGame] = []
    for game_id, mapped in homes.items():
        away = aways.get(game_id)
        if away is not None:
            _validate_away_row(game_id, mapped, away)
        games.append(game_from_mapping(mapped))

    ordered = validate_games(games)
    normalized_by_season = Counter(game.season for game in ordered)
    skipped_tied_scores = sum(skipped_by_season.values())
    notes = []
    if skipped_tied_scores:
        notes.append(
            f"Excluded {skipped_tied_scores} regular-season HOME rows with tied "
            "MoneyPuck situation=all scores. This file does not identify a "
            "shootout winner, and no winner was invented."
        )
    return NormalizeReport(
        games=ordered,
        skipped_tied_scores=skipped_tied_scores,
        skipped_tied_scores_by_season=dict(sorted(skipped_by_season.items())),
        normalized_games_by_season=dict(sorted(normalized_by_season.items())),
        notes=notes,
    )


def games_from_moneypuck_csv(
    path: Path,
    seasons: Iterable[int] | None = None,
    source: str = SOURCE_ID,
) -> NormalizeReport:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SchemaError(f"MoneyPuck file has no header: {path}")
            missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise SchemaError(f"MoneyPuck file is missing columns {missing}")
            return games_from_moneypuck_rows(reader, seasons=seasons, source=source)
    except OSError as exc:
        raise SchemaError(f"Unreadable MoneyPuck file {path}") from exc
