"""
Backfill historical NHL opening moneylines from Action Network book 30 ("Open").

Book 30 is posted days before puck drop. Coverage starts with the 2023-24 season;
earlier scoreboards only have the consensus line. Writes data/opening_odds.csv
keyed by game_id so the results tab can rescore ROI at the opening price.

Usage:
  python src/fetch_opening_odds.py
  python src/fetch_opening_odds.py --refresh
"""
from __future__ import annotations

import argparse
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "game_features.csv"
QUOTES_CSV = PROJECT_ROOT / "data" / "an_opening_quotes.csv"
FETCHED_DATES_CSV = PROJECT_ROOT / "data" / "an_opening_fetched_dates.csv"
OPENING_CSV = PROJECT_ROOT / "data" / "opening_odds.csv"
ACTIONNETWORK_NHL = "https://api.actionnetwork.com/web/v1/scoreboard/nhl"
AN_OPEN_BOOK = 30
MIN_DATE = date(2023, 10, 1)
HEADERS = {"User-Agent": "Mozilla/5.0"}
ALIASES = {
    "utah utah hockey club": "utah mammoth",
    "utah hockey club": "utah mammoth",
    # Action Network rewrote 2023-24 Coyotes games under the later franchise name.
    "arizona coyotes": "utah mammoth",
}


def normalize_team_name(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(".", "").casefold().strip()
    return ALIASES.get(text, text)


def load_games(min_date: date = MIN_DATE) -> pd.DataFrame:
    games = pd.read_csv(
        FEATURES_CSV,
        usecols=["game_id", "date", "home_team_name", "away_team_name"],
    )
    games["date"] = pd.to_datetime(games["date"]).dt.date
    games = games.loc[games["date"] >= min_date].copy()
    games["home_key"] = games["home_team_name"].map(normalize_team_name)
    games["away_key"] = games["away_team_name"].map(normalize_team_name)
    return games


def _candidate_dates(games: pd.DataFrame) -> list[str]:
    dates: set[date] = set()
    for game_date in games["date"]:
        dates.add(game_date)
        dates.add(game_date + timedelta(days=1))
        dates.add(game_date - timedelta(days=1))
    return [day.strftime("%Y%m%d") for day in sorted(dates)]


def _an_date(url_date: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                ACTIONNETWORK_NHL,
                params={"date": url_date},
                headers=HEADERS,
                timeout=30,
            )
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{url_date}: {last_error}")


def _open_quote(game: dict) -> dict | None:
    teams = {team.get("id"): team.get("full_name") for team in (game.get("teams") or [])}
    home = teams.get(game.get("home_team_id"))
    away = teams.get(game.get("away_team_id"))
    if not home or not away:
        return None
    offer = next(
        (
            odds
            for odds in (game.get("odds") or [])
            if odds.get("book_id") == AN_OPEN_BOOK
            and odds.get("type") == "game"
            and odds.get("ml_home") not in (None, 0)
            and odds.get("ml_away") not in (None, 0)
        ),
        None,
    )
    if offer is None:
        return None
    kick = pd.to_datetime(game.get("start_time"), utc=True, errors="coerce")
    return {
        "an_id": game.get("id"),
        "kickoff_utc": kick,
        "home": home,
        "away": away,
        "open_ml_home": int(offer["ml_home"]),
        "open_ml_away": int(offer["ml_away"]),
        "open_inserted": offer.get("inserted"),
    }


def fetch_opening_quotes(dates: list[str], *, pause: float = 0.05) -> list[dict]:
    quotes: list[dict] = []
    seen: set = set()
    for index, day in enumerate(dates, start=1):
        try:
            payload = _an_date(day)
        except Exception as exc:
            print(f"  skip {day}: {exc}")
            continue
        for game in payload.get("games") or []:
            an_id = game.get("id")
            if an_id in seen:
                continue
            quote = _open_quote(game)
            if quote is None:
                continue
            seen.add(an_id)
            quotes.append(quote)
        if index % 25 == 0 or index == len(dates):
            print(f"  {index}/{len(dates)} dates  quotes={len(quotes)}")
        if pause:
            time.sleep(pause)
    return quotes


def align_opening_odds(games: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    by_teams: dict[tuple[str, str], list[tuple[date, int]]] = {}
    for row in games.itertuples(index=False):
        by_teams.setdefault((row.home_key, row.away_key), []).append((row.date, int(row.game_id)))

    rows = []
    unmatched = 0
    for quote in quotes.itertuples(index=False):
        kick = quote.kickoff_utc
        if pd.isna(kick):
            unmatched += 1
            continue
        kick = pd.Timestamp(kick)
        if kick.tzinfo is None:
            kick = kick.tz_localize("UTC")
        et_date = kick.tz_convert("America/New_York").date()
        candidates = by_teams.get(
            (normalize_team_name(quote.home), normalize_team_name(quote.away)),
            [],
        )
        hits = [(abs((game_date - et_date).days), game_id) for game_date, game_id in candidates]
        hits = [hit for hit in hits if hit[0] <= 1]
        if not hits:
            unmatched += 1
            continue
        hits.sort()
        rows.append(
            {
                "game_id": hits[0][1],
                "open_ml_home": int(quote.open_ml_home),
                "open_ml_away": int(quote.open_ml_away),
                "open_inserted": quote.open_inserted,
                "open_source": "actionnetwork",
                "open_book": "Open",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates("game_id", keep="last")
    print(f"  matched {len(out)}  unmatched quotes {unmatched}")
    return out


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_opening_odds(*, refresh: bool = False) -> pd.DataFrame:
    games = load_games()
    cached_quotes = pd.DataFrame() if refresh else _load_csv(QUOTES_CSV)
    fetched = set() if refresh or not FETCHED_DATES_CSV.is_file() else set(
        pd.read_csv(FETCHED_DATES_CSV)["date"].astype(str)
    )
    needed = [day for day in _candidate_dates(games) if day not in fetched]
    fresh_quotes = pd.DataFrame()
    if needed:
        print(f"Fetching Action Network Open moneylines for {len(needed)} dates...")
        fresh_quotes = pd.DataFrame(fetch_opening_quotes(needed))
        fetched.update(needed)
    else:
        print("Opening quote cache is complete. Realigning to games.")

    if cached_quotes.empty:
        quotes = fresh_quotes
    elif fresh_quotes.empty:
        quotes = cached_quotes
    else:
        quotes = pd.concat([cached_quotes, fresh_quotes], ignore_index=True)
    if not quotes.empty:
        quotes["kickoff_utc"] = pd.to_datetime(quotes["kickoff_utc"], utc=True, errors="coerce")
        quotes = quotes.drop_duplicates("an_id", keep="last")

    OPENING_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not quotes.empty:
        quotes.to_csv(QUOTES_CSV, index=False)
    pd.DataFrame({"date": sorted(fetched)}).to_csv(FETCHED_DATES_CSV, index=False)

    aligned = align_opening_odds(games, quotes) if not quotes.empty else pd.DataFrame()
    if not aligned.empty:
        aligned = aligned.sort_values("game_id")
        aligned.to_csv(OPENING_CSV, index=False)
    print(f"Wrote {OPENING_CSV} ({len(aligned)} games)")
    covered = games["game_id"].astype(int).isin(set(aligned["game_id"].astype(int)) if not aligned.empty else set())
    print(f"  Coverage since {MIN_DATE}: {int(covered.sum())}/{len(games)} games")
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical NHL opening moneylines.")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cache and refetch.")
    args = parser.parse_args()
    build_opening_odds(refresh=args.refresh)


if __name__ == "__main__":
    main()
