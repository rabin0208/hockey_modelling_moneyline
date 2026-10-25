"""
Compare the model with Polymarket NHL prices from 2 to 7 days before puck drop.

Price history on Polymarket only survives for recent markets, so this covers
games from late September 2026 onward. Earlier 2026 games are listed but their
trade path is gone.

Usage:
  python src/check_polymarket_week.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical import complete_rows, feature_columns, make_model  # noqa: E402
from fetch_opening_odds import normalize_team_name  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERIES_ID = 10346
HEADERS = {"User-Agent": "Mozilla/5.0"}
CACHE = PROJECT_ROOT / "data" / "polymarket_week_prices.csv"
NICKNAMES = {
    "bruins": "boston bruins",
    "sabres": "buffalo sabres",
    "flames": "calgary flames",
    "hurricanes": "carolina hurricanes",
    "blackhawks": "chicago blackhawks",
    "avalanche": "colorado avalanche",
    "blue jackets": "columbus blue jackets",
    "stars": "dallas stars",
    "red wings": "detroit red wings",
    "oilers": "edmonton oilers",
    "panthers": "florida panthers",
    "kings": "los angeles kings",
    "wild": "minnesota wild",
    "canadiens": "montreal canadiens",
    "predators": "nashville predators",
    "devils": "new jersey devils",
    "islanders": "new york islanders",
    "rangers": "new york rangers",
    "senators": "ottawa senators",
    "flyers": "philadelphia flyers",
    "penguins": "pittsburgh penguins",
    "sharks": "san jose sharks",
    "kraken": "seattle kraken",
    "blues": "st louis blues",
    "lightning": "tampa bay lightning",
    "maple leafs": "toronto maple leafs",
    "leafs": "toronto maple leafs",
    "mammoth": "utah mammoth",
    "canucks": "vancouver canucks",
    "golden knights": "vegas golden knights",
    "knights": "vegas golden knights",
    "capitals": "washington capitals",
    "jets": "winnipeg jets",
    "ducks": "anaheim ducks",
}


def canonical(name: str) -> str | None:
    key = name.replace(".", "").strip().lower()
    return NICKNAMES.get(key)


def list_events() -> list[dict]:
    events = []
    offset = 1400
    while offset < 2000:
        response = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"series_id": SERIES_ID, "limit": 100, "offset": offset},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        events.extend(page)
        if len(page) < 100:
            break
        offset += 100
        time.sleep(0.05)
    return events


def price_history(token: str) -> list[dict]:
    response = requests.get(
        "https://clob.polymarket.com/prices-history",
        params={"market": token, "interval": "max", "fidelity": 180},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("history") or []


def snapshot(history: list[dict], game_ts: float, low_h: float, high_h: float, target_h: float):
    choices = []
    for point in history:
        lead = (game_ts - point["t"]) / 3600.0
        price = float(point["p"])
        if low_h <= lead <= high_h and 0.02 < price < 0.98:
            choices.append((abs(lead - target_h), lead, price))
    if not choices:
        return None
    choices.sort()
    return choices[0][1], choices[0][2]


def collect_prices() -> pd.DataFrame:
    if CACHE.is_file():
        cached = pd.read_csv(CACHE)
        if not cached.empty:
            print(f"Using cached Polymarket prices ({len(cached)} markets)")
            return cached
    rows = []
    events = list_events()
    print(f"Events scanned: {len(events)}")
    for index, event in enumerate(events, start=1):
        title = event.get("title") or ""
        if " vs. " not in title or "O/U" in title or "Spread" in title:
            continue
        markets = event.get("markets") or []
        if not markets:
            continue
        market = markets[0]
        start = market.get("gameStartTime") or event.get("endDate")
        if not start or start < "2026-09-26":
            continue
        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        tokens = market.get("clobTokenIds")
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if not outcomes or not tokens or len(outcomes) < 2:
            continue
        try:
            history = price_history(tokens[0])
        except Exception as exc:
            print(f"  skip {title}: {exc}")
            continue
        game_ts = pd.Timestamp(start, tz="UTC").timestamp()
        week = snapshot(history, game_ts, 48, 168, 96)
        close = snapshot(history, game_ts, 1, 12, 3)
        if week is None:
            continue
        rows.append(
            {
                "title": title,
                "start": start,
                "outcome": outcomes[0],
                "other": outcomes[1],
                "week_lead_h": week[0],
                "week_price": week[1],
                "close_lead_h": None if close is None else close[0],
                "close_price": None if close is None else close[1],
                "created": market.get("createdAt"),
            }
        )
        if index % 25 == 0:
            print(f"  {index}/{len(events)} events  kept={len(rows)}")
        time.sleep(0.04)
    out = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    print(f"Wrote {CACHE} ({len(out)} markets)")
    return out


def model_probs() -> pd.DataFrame:
    features = pd.read_csv(PROJECT_ROOT / "data" / "game_features.csv")
    features["date"] = pd.to_datetime(features["date"])
    features = complete_rows(features)
    columns = feature_columns(features)
    pieces = []
    for season in sorted(features["season"].unique()):
        train = features[features["season"] != season]
        test = features[features["season"] == season].copy()
        model = make_model()
        model.fit(train[columns], train["home_wins"].astype(int))
        test["p_home"] = model.predict_proba(test[columns])[:, 1]
        pieces.append(test)
    scored = pd.concat(pieces, ignore_index=True)
    scored["home_key"] = scored["home_team_name"].map(normalize_team_name)
    scored["away_key"] = scored["away_team_name"].map(normalize_team_name)
    return scored


def match_game(row, games: pd.DataFrame):
    first = canonical(row.outcome)
    second = canonical(row.other)
    if not first or not second:
        return None
    start = pd.Timestamp(row.start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    day = start.tz_convert("America/New_York").date()
    keys = {first, second}
    hits = []
    for game in games.itertuples(index=False):
        if {game.home_key, game.away_key} != keys:
            continue
        delta = abs((game.date.date() - day).days)
        if delta <= 1:
            hits.append((delta, game))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    return hits[0][1]


def score(prices: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in prices.itertuples(index=False):
        game = match_game(row, games)
        if game is None or pd.isna(game.home_wins):
            continue
        home_key = normalize_team_name(game.home_team_name)
        week_home = float(row.week_price) if canonical(row.outcome) == home_key else 1.0 - float(row.week_price)
        edge = float(game.p_home) - week_home
        bet_home = edge >= 0
        price = week_home if bet_home else 1.0 - week_home
        if not 0.02 < price < 0.98:
            continue
        won = (game.home_wins == 1) if bet_home else (game.home_wins == 0)
        profit = (1.0 - price) / price if won else -1.0
        close_home = np.nan
        if pd.notna(row.close_price):
            close_home = float(row.close_price) if canonical(row.outcome) == home_key else 1.0 - float(row.close_price)
        close_side = close_home if bet_home else (1.0 - close_home if pd.notna(close_home) else np.nan)
        records.append(
            {
                "start": row.start,
                "matchup": row.title,
                "home": game.home_team_name,
                "week_lead_h": row.week_lead_h,
                "edge": abs(edge),
                "signed_edge": edge,
                "price": price,
                "profit": profit,
                "clv": (close_side - price) if pd.notna(close_side) else np.nan,
            }
        )
    return pd.DataFrame(records)


def summarize(fills: pd.DataFrame, minimum: float) -> str:
    group = fills[fills["edge"] >= minimum]
    if group.empty:
        return f"  edge >= {minimum:.0%}: no bets"
    clv = group["clv"].mean()
    clv_txt = "n/a" if pd.isna(clv) else f"{clv:+.1%}"
    return (
        f"  edge >= {minimum:.0%}: n={len(group):3d}  "
        f"ROI={group['profit'].mean():+.1%}  "
        f"hit={(group['profit'] > 0).mean():.0%}  "
        f"price move to the close={clv_txt}"
    )


def main() -> None:
    prices = collect_prices()
    print(f"Markets with a price 2–7 days out: {len(prices)}")
    games = model_probs()
    fills = score(prices, games)
    print(f"Matched to a played game: {len(fills)}")
    if fills.empty:
        return
    print(f"Lead time of the quote used: median {fills['week_lead_h'].median():.0f}h")
    print("Buying the side the model prefers, at the Polymarket price from that window:")
    for minimum in (0.0, 0.03, 0.05, 0.08, 0.10):
        print(summarize(fills, minimum))
    best = fills[fills["edge"] >= 0.08].sort_values("edge", ascending=False)
    print("\nLargest 8%+ disagreements:")
    if best.empty:
        print("  none")
    else:
        show = best.head(8).copy()
        show["edge"] = (show["edge"] * 100).round(1)
        show["price"] = show["price"].round(2)
        show["profit"] = show["profit"].round(2)
        print(show[["start", "matchup", "edge", "price", "profit"]].to_string(index=False))


if __name__ == "__main__":
    main()
