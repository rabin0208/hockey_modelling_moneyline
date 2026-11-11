"""Build the frozen hockey_moneyline feature table.

The 25 columns and the calculations are the copied
``hockey_moneyline`` prospective model. Development-season rows come from
``build_frozen_training_rows``. Later games and upcoming matchups use
``build_unscored_frozen_feature_snapshot``, which that project checks
against the same rows.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hockey_moneyline.acquire import acquire_source_csv, cache_path
from hockey_moneyline.advanced_stats import inspect_advanced_stats_csv_for_cycle2_development
from hockey_moneyline.current_season_prior import (
    CURRENT_SEASON,
    parse_current_season_moneypuck_csv,
)
from hockey_moneyline.errors import FeatureError, SchemaError
from hockey_moneyline.normalize import canonical_moneypuck_team_code
from hockey_moneyline.prospective_artifact import (
    FROZEN_FEATURE_COLUMNS,
    build_frozen_training_rows,
    eligible_frozen_training_rows,
)
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    schedule_history_from_moneypuck_csv,
)
from hockey_moneyline.unscored_target import (
    UnscoredTarget,
    build_unscored_frozen_feature_snapshot,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "moneypuck"
OUTPUT_PATH = PROJECT_ROOT / "data" / "game_features.csv"

ID_COLUMNS = (
    "game_id",
    "date",
    "season",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "home_wins",
    "home_score",
    "away_score",
)

TEAM_NAMES = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}

_SOURCE = None
_SOURCE_STAT: tuple[int, int] | None = None


def season_id(game_date: date) -> int:
    """NHL season id, such as 20262027 for a November 2026 game."""
    start = game_date.year if game_date.month >= 10 else game_date.year - 1
    return int(f"{start}{start + 1}")


def team_display_name(code: str, season: int) -> str:
    if code == "UTA" and int(season) <= 20242025:
        return "Utah Hockey Club"
    return TEAM_NAMES.get(code, code)


class FeatureSource:
    """Completed MoneyPuck games, expected-goal records, and schedule history."""

    def __init__(self, games, records, schedule, development_rows):
        self.games = games
        self.records = records
        self.schedule = schedule
        self.development_rows = development_rows


def source_cache_token() -> str:
    """Identity of the MoneyPuck file currently on disk."""
    path = cache_path(CACHE_DIR)
    if not path.exists() or path.stat().st_size == 0:
        return "missing"
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def load_source(
    refresh: bool = False,
    max_age_seconds: float | None = None,
) -> FeatureSource:
    """Download MoneyPuck if needed and build the same inputs the other model uses.

    An in-memory source is reused only when it was built from the file still on
    disk. ``max_age_seconds`` re-downloads that file once it is older than the
    limit, which is how the deployed app picks up last night's results.
    """
    global _SOURCE, _SOURCE_STAT
    if _SOURCE is not None and not refresh and max_age_seconds is None:
        return _SOURCE

    path = acquire_source_csv(
        cache_dir=CACHE_DIR,
        refresh=refresh,
        max_age_seconds=max_age_seconds,
    )
    stat = (path.stat().st_mtime_ns, path.stat().st_size)
    if _SOURCE is not None and _SOURCE_STAT == stat and not refresh:
        return _SOURCE

    print(f"Loading MoneyPuck games from {path}...")
    inspection = inspect_advanced_stats_csv_for_cycle2_development(
        path,
        CYCLE2_DEVELOPMENT_SEASONS,
    )
    schedule = schedule_history_from_moneypuck_csv(path, seasons=CYCLE2_DEVELOPMENT_SEASONS)
    development_rows = eligible_frozen_training_rows(
        build_frozen_training_rows(inspection.games, inspection.records, schedule)
    )
    current = parse_current_season_moneypuck_csv(path, season=CURRENT_SEASON)
    games = list(inspection.games) + list(current.winner_labeled)
    records = dict(inspection.records)
    records.update(current.xgoal_records)
    full_schedule = list(schedule) + list(current.schedule_history)
    print(
        f"  {len(inspection.games)} development games, "
        f"{len(current.winner_labeled)} current-season games with a winner, "
        f"{len(development_rows)} eligible development rows"
    )
    _SOURCE = FeatureSource(games, records, full_schedule, development_rows)
    _SOURCE_STAT = stat
    return _SOURCE


def _public_row(row: dict) -> dict:
    season = int(row["season"])
    home = str(row["home_team_id"])
    away = str(row["away_team_id"])
    public = {
        "game_id": str(row["game_id"]),
        "date": row["game_date"] if isinstance(row["game_date"], date) else date.fromisoformat(str(row["game_date"])[:10]),
        "season": season,
        "home_team_id": home,
        "away_team_id": away,
        "home_team_name": team_display_name(home, season),
        "away_team_name": team_display_name(away, season),
        "home_wins": int(row["home_win"]),
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
    }
    for column in FROZEN_FEATURE_COLUMNS:
        public[column] = float(row[column])
    return public


def _snapshot_row(source: FeatureSource, game) -> dict | None:
    game_day = game.game_date
    prior_games = [item for item in source.games if item.game_date < game_day]
    prior_schedule = [item for item in source.schedule if item.game_date < game_day]
    target = UnscoredTarget(
        game_id=game.game_id,
        game_date=game_day,
        season=game.season,
        game_type="regular",
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
    )
    try:
        snapshot = build_unscored_frozen_feature_snapshot(
            target,
            prior_games,
            source.records,
            prior_schedule,
        )
    except FeatureError:
        return None
    row = {
        "game_id": game.game_id,
        "game_date": game_day,
        "season": game.season,
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "home_win": game.home_win,
        "home_score": game.home_score,
        "away_score": game.away_score,
    }
    row.update(snapshot.as_feature_row())
    return _public_row(row)


def feature_frame(source: FeatureSource | None = None) -> pd.DataFrame:
    """One row per eligible game, with the frozen 25 features and the home-win label."""
    source = source or load_source()
    rows = [_public_row(row) for row in source.development_rows]
    current_games = [game for game in source.games if game.season == CURRENT_SEASON]
    print(f"Scoring {len(current_games)} current-season games with the frozen snapshot...")
    for index, game in enumerate(current_games, start=1):
        if index % 200 == 0:
            print(f"  {index}/{len(current_games)}")
        built = _snapshot_row(source, game)
        if built is not None:
            rows.append(built)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = [value if isinstance(value, date) else date.fromisoformat(str(value)[:10]) for value in frame["date"]]
    return frame


def features_for_game(game: dict, source: FeatureSource | None = None) -> dict | None:
    """Pregame 25-vector for one matchup. Completed results on that date are excluded."""
    source = source or load_source()
    game_date = game["date"]
    if isinstance(game_date, str):
        game_date = date.fromisoformat(game_date[:10])
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()
    season = season_id(game_date)
    try:
        home = canonical_moneypuck_team_code(
            game.get("home_team_abbrev") or game.get("home_team_id"),
            season,
        )
        away = canonical_moneypuck_team_code(
            game.get("away_team_abbrev") or game.get("away_team_id"),
            season,
        )
    except SchemaError:
        return None

    prior_games = [item for item in source.games if item.game_date < game_date]
    prior_schedule = [item for item in source.schedule if item.game_date < game_date]
    target = UnscoredTarget(
        game_id=str(game.get("game_id") or f"{game_date.isoformat()}-{home}-{away}"),
        game_date=game_date,
        season=season,
        game_type="regular",
        home_team_id=home,
        away_team_id=away,
    )
    try:
        snapshot = build_unscored_frozen_feature_snapshot(
            target,
            prior_games,
            source.records,
            prior_schedule,
        )
    except FeatureError:
        return None
    return snapshot.as_feature_row()


def main() -> None:
    print("=" * 60)
    print("Frozen hockey_moneyline features")
    print("=" * 60)
    frame = feature_frame(load_source())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    for column in FROZEN_FEATURE_COLUMNS:
        export[column] = [repr(float(value)) for value in export[column]]
    export.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(frame)} games to {OUTPUT_PATH}")
    print(f"Feature columns ({len(FROZEN_FEATURE_COLUMNS)}):")
    for column in FROZEN_FEATURE_COLUMNS:
        print(f"  - {column}")
    if not frame.empty:
        print(f"Home team win rate: {frame['home_wins'].mean():.2%}")
        print(f"Seasons: {sorted(frame['season'].unique())}")


if __name__ == "__main__":
    main()
