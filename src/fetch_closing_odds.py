"""
Backfill historical NHL closing moneylines from Action Network book 15.

Book 15 is the consensus price, last updated around puck drop. Coverage goes
back through 2021-22. Writes data/closing_odds.csv keyed by game_id.

Usage:
  python src/fetch_closing_odds.py
  python src/fetch_closing_odds.py --refresh
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_opening_odds import (  # noqa: E402
    _an_date,
    align_opening_odds,
    load_games,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUOTES_CSV = PROJECT_ROOT / "data" / "an_closing_quotes.csv"
FETCHED_DATES_CSV = PROJECT_ROOT / "data" / "an_closing_fetched_dates.csv"
CLOSING_CSV = PROJECT_ROOT / "data" / "closing_odds.csv"
AN_CLOSE_BOOK = 15
MIN_DATE = date(2021, 10, 1)


def _close_quote(game: dict) -> dict | None:
    teams = {team.get("id"): team.get("full_name") for team in (game.get("teams") or [])}
    home = teams.get(game.get("home_team_id"))
    away = teams.get(game.get("away_team_id"))
    if not home or not away:
        return None
    offer = next(
        (
            odds
            for odds in (game.get("odds") or [])
            if odds.get("book_id") == AN_CLOSE_BOOK
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


def fetch_closing_quotes(dates: list[str], *, pause: float = 0.05) -> list[dict]:
    import time

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
            quote = _close_quote(game)
            if quote is None:
                continue
            seen.add(an_id)
            quotes.append(quote)
        if index % 25 == 0 or index == len(dates):
            print(f"  {index}/{len(dates)} dates  quotes={len(quotes)}")
        if pause:
            time.sleep(pause)
    return quotes


def _candidate_dates(games: pd.DataFrame) -> list[str]:
    from datetime import timedelta

    dates: set[date] = set()
    for game_date in games["date"]:
        dates.add(game_date)
        dates.add(game_date + timedelta(days=1))
        dates.add(game_date - timedelta(days=1))
    return [day.strftime("%Y%m%d") for day in sorted(dates)]


def build_closing_odds(*, refresh: bool = False) -> pd.DataFrame:
    games = load_games(MIN_DATE)
    cached = pd.DataFrame() if refresh or not QUOTES_CSV.is_file() else pd.read_csv(QUOTES_CSV)
    fetched = set() if refresh or not FETCHED_DATES_CSV.is_file() else set(
        pd.read_csv(FETCHED_DATES_CSV)["date"].astype(str)
    )
    needed = [day for day in _candidate_dates(games) if day not in fetched]
    fresh = pd.DataFrame()
    if needed:
        print(f"Fetching Action Network closing moneylines for {len(needed)} dates...")
        fresh = pd.DataFrame(fetch_closing_quotes(needed))
        fetched.update(needed)
    else:
        print("Closing quote cache is complete. Realigning to games.")

    if cached.empty:
        quotes = fresh
    elif fresh.empty:
        quotes = cached
    else:
        quotes = pd.concat([cached, fresh], ignore_index=True)
    if not quotes.empty:
        quotes["kickoff_utc"] = pd.to_datetime(quotes["kickoff_utc"], utc=True, errors="coerce")
        quotes = quotes.drop_duplicates("an_id", keep="last")

    CLOSING_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not quotes.empty:
        quotes.to_csv(QUOTES_CSV, index=False)
    pd.DataFrame({"date": sorted(fetched)}).to_csv(FETCHED_DATES_CSV, index=False)

    # align_opening_odds reads open_ml_* column names; the quote frame uses those
    # names so the same matcher can attach book 15. Rename after the join.
    aligned = align_opening_odds(games, quotes) if not quotes.empty else pd.DataFrame()
    if not aligned.empty:
        aligned = aligned.rename(
            columns={
                "open_ml_home": "close_ml_home",
                "open_ml_away": "close_ml_away",
                "open_inserted": "close_inserted",
                "open_source": "close_source",
                "open_book": "close_book",
            }
        )
        aligned["close_book"] = "Consensus"
        aligned = aligned.sort_values("game_id")
        aligned.to_csv(CLOSING_CSV, index=False)
    print(f"Wrote {CLOSING_CSV} ({len(aligned)} games)")
    covered = games["game_id"].astype(int).isin(
        set(aligned["game_id"].astype(int)) if not aligned.empty else set()
    )
    print(f"  Coverage since {MIN_DATE}: {int(covered.sum())}/{len(games)} games")
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical NHL closing moneylines.")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cache and refetch.")
    args = parser.parse_args()
    build_closing_odds(refresh=args.refresh)


if __name__ == "__main__":
    main()
