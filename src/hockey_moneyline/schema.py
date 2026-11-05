"""Normalized historical-game contract with optional NHL identity/timing enrichment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from hockey_moneyline.errors import SchemaError

# Keep the Issue #2 MoneyPuck contract stable. NHL fields are optional enrichment.
REQUIRED_FIELDS = (
    "game_id",
    "game_date",
    "season",
    "game_type",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_score",
    "away_score",
    "home_win",
)

FEATURE_COLUMNS = (
    "home_prior_games",
    "away_prior_games",
    "home_win_rate",
    "away_win_rate",
    "home_goals_for_per_game",
    "away_goals_for_per_game",
    "home_goals_against_per_game",
    "away_goals_against_per_game",
    "home_goal_diff_per_game",
    "away_goal_diff_per_game",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_season_games",
    "away_season_games",
    "home_season_win_rate",
    "away_season_win_rate",
    "home_recent_win_rate",
    "away_recent_win_rate",
)


@dataclass(frozen=True, slots=True)
class NormalizedGame:
    game_id: str
    game_date: date
    season: int
    game_type: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    home_score: int
    away_score: int
    home_win: int
    start_time_utc: datetime | None = None
    home_team_abbrev: str = ""
    away_team_abbrev: str = ""
    source: str = "synthetic"
    nhl_game_id: str | None = None
    home_nhl_team_id: int | None = None
    away_nhl_team_id: int | None = None
    scheduled_start_time_utc: datetime | None = None
    game_state: str | None = None
    game_schedule_state: str | None = None

    @property
    def season_id(self) -> int:
        return self.season

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["game_date"] = self.game_date.isoformat()
        for field in ("start_time_utc", "scheduled_start_time_utc"):
            value = getattr(self, field)
            payload[field] = value.isoformat() if value is not None else None
        payload["season_id"] = self.season
        return payload


def parse_game_date(value: str | date | int) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise SchemaError(f"Invalid game_date: {value!r}") from exc


def parse_start_time(value: str | datetime | None) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise SchemaError(f"Invalid start_time_utc: {value!r}") from exc
    if parsed.tzinfo is None:
        raise SchemaError(f"start_time_utc must be timezone-aware: {value!r}")
    return parsed


def home_win_from_scores(home_score: int, away_score: int) -> int:
    if home_score == away_score:
        raise SchemaError("NHL regular-season games must have a winner; tied scores are invalid")
    return 1 if home_score > away_score else 0


def _optional_int(value: Any, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field} must be an integer when present: {value!r}") from exc


def game_from_mapping(row: Mapping[str, Any]) -> NormalizedGame:
    missing = [field for field in REQUIRED_FIELDS if field not in row or row[field] in (None, "")]
    if missing:
        raise SchemaError(f"Missing required fields: {missing}")

    home_score = int(row["home_score"])
    away_score = int(row["away_score"])
    if home_score < 0 or away_score < 0:
        raise SchemaError("Scores must be non-negative")

    expected = home_win_from_scores(home_score, away_score)
    home_win = int(row["home_win"])
    if home_win not in (0, 1):
        raise SchemaError(f"home_win must be 0 or 1, got {home_win!r}")
    if home_win != expected:
        raise SchemaError(
            f"home_win {home_win} does not match scores {home_score}-{away_score}"
        )

    home_team_id = str(row["home_team_id"])
    away_team_id = str(row["away_team_id"])
    if home_team_id == away_team_id:
        raise SchemaError(f"Home and away team ids are identical: {home_team_id}")

    start_time = parse_start_time(row.get("start_time_utc"))
    scheduled_start = parse_start_time(row.get("scheduled_start_time_utc"))
    if start_time and scheduled_start and start_time != scheduled_start:
        raise SchemaError("start_time_utc and scheduled_start_time_utc disagree")
    canonical_start = scheduled_start or start_time

    return NormalizedGame(
        game_id=str(row["game_id"]),
        game_date=parse_game_date(row["game_date"]),
        season=int(row.get("season_id") or row["season"]),
        game_type=str(row["game_type"]),
        home_team_id=home_team_id,
        home_team_name=str(row["home_team_name"]),
        away_team_id=away_team_id,
        away_team_name=str(row["away_team_name"]),
        home_score=home_score,
        away_score=away_score,
        home_win=home_win,
        start_time_utc=canonical_start,
        home_team_abbrev=str(row.get("home_team_abbrev") or row["home_team_id"]),
        away_team_abbrev=str(row.get("away_team_abbrev") or row["away_team_id"]),
        source=str(row.get("source") or "unknown"),
        nhl_game_id=(
            str(row["nhl_game_id"]) if row.get("nhl_game_id") not in (None, "") else None
        ),
        home_nhl_team_id=_optional_int(row.get("home_nhl_team_id"), "home_nhl_team_id"),
        away_nhl_team_id=_optional_int(row.get("away_nhl_team_id"), "away_nhl_team_id"),
        scheduled_start_time_utc=canonical_start,
        game_state=(str(row["game_state"]) if row.get("game_state") not in (None, "") else None),
        game_schedule_state=(
            str(row["game_schedule_state"])
            if row.get("game_schedule_state") not in (None, "")
            else None
        ),
    )


def sort_key(game: NormalizedGame) -> tuple[date, str]:
    return (game.game_date, game.game_id)


def sort_games(games: Iterable[NormalizedGame]) -> list[NormalizedGame]:
    return sorted(games, key=sort_key)


def validate_games(games: Sequence[NormalizedGame]) -> list[NormalizedGame]:
    if not games:
        raise SchemaError("No normalized games to validate")

    ordered = sort_games(games)
    seen: set[str] = set()
    previous: NormalizedGame | None = None
    for game in ordered:
        if game.game_id in seen:
            raise SchemaError(f"Duplicate game_id: {game.game_id}")
        seen.add(game.game_id)
        if previous is not None and sort_key(game) < sort_key(previous):
            raise SchemaError("Games are not in chronological order")
        previous = game
    return ordered
