"""
Choose a minimum edge and a pregame quoting window for making markets.

The proxy is the Action Network open: post on the side the model likes once
that line is inside the window, get filled at the open, and judge the fill by
settlement ROI and by whether the close moved toward that side. Parameters are
picked on 2023-24 and 2024-25, then checked on 2025-26 and 2026-27.

Usage:
  python src/optimize_quote_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical import (  # noqa: E402
    attach_two_way,
    complete_rows,
    devig,
    feature_columns,
    make_model,
    profit_from_american,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECT_SEASONS = {20232024, 20242025}
HOLD_SEASONS = {20252026, 20262027}
EDGES = (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
WINDOWS = (
    (0, 6),
    (0, 12),
    (0, 24),
    (0, 48),
    (0, 168),
    (3, 24),
    (6, 24),
    (6, 48),
    (12, 48),
    (48, 72),
    (48, 96),
    (48, 168),
    (72, 168),
    (96, 168),
)
LONG_WINDOWS = ((48, 72), (48, 96), (48, 168), (72, 168), (96, 168))
MIN_SELECT_BETS = 150


def leave_one_season_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = complete_rows(df)
    columns = feature_columns(df)
    pieces = []
    for season in sorted(df["season"].unique()):
        train = df[df["season"] != season]
        test = df[df["season"] == season].copy()
        model = make_model()
        model.fit(train[columns], train["home_wins"].astype(int))
        test["p_home"] = model.predict_proba(test[columns])[:, 1]
        pieces.append(test)
    return pd.concat(pieces, ignore_index=True)


def load_quotes() -> pd.DataFrame:
    features = pd.read_csv(PROJECT_ROOT / "data" / "game_features.csv")
    features["date"] = pd.to_datetime(features["date"])
    features = attach_two_way(features, str(PROJECT_ROOT / "data" / "opening_odds.csv"), "open")
    features = attach_two_way(features, str(PROJECT_ROOT / "data" / "closing_odds.csv"), "close")
    features = leave_one_season_probs(features)

    opening = pd.read_csv(PROJECT_ROOT / "data" / "opening_odds.csv")
    quotes = pd.read_csv(PROJECT_ROOT / "data" / "an_opening_quotes.csv")
    for frame in (opening, quotes):
        frame["open_inserted"] = pd.to_datetime(frame["open_inserted"], utc=True)
    quotes["kickoff_utc"] = pd.to_datetime(quotes["kickoff_utc"], utc=True)
    quotes = quotes.drop_duplicates(["open_inserted", "open_ml_home", "open_ml_away"])
    timed = opening.merge(
        quotes[["open_inserted", "open_ml_home", "open_ml_away", "kickoff_utc"]],
        on=["open_inserted", "open_ml_home", "open_ml_away"],
        how="left",
    )
    timed = timed.drop_duplicates("game_id")
    features["game_id"] = features["game_id"].astype(str)
    timed["game_id"] = timed["game_id"].astype(str)
    features = features.merge(
        timed[["game_id", "open_inserted", "kickoff_utc"]],
        on="game_id",
        how="left",
    )
    features["lead_h"] = (
        features["kickoff_utc"] - features["open_inserted"]
    ).dt.total_seconds() / 3600.0
    return features


def score_fills(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "roi": np.nan, "clv": np.nan, "hit": np.nan}
    return {
        "n": int(len(frame)),
        "roi": float(frame["profit"].mean()),
        "clv": float(frame["clv"].mean()),
        "hit": float((frame["profit"] > 0).mean()),
    }


def candidate_fills(quotes: pd.DataFrame) -> pd.DataFrame:
    """One fill per game: the side with edge versus the open, inside a positive lead time."""
    rows = quotes[
        quotes["open_odds_valid"]
        & quotes["close_odds_valid"]
        & quotes["lead_h"].notna()
        & (quotes["lead_h"] > 0)
    ].copy()
    records = []
    for row in rows.itertuples(index=False):
        fair_home, fair_away = devig(row.open_ml_home, row.open_ml_away)
        close_home, close_away = devig(row.close_ml_home, row.close_ml_away)
        if not np.isfinite([fair_home, fair_away, close_home, close_away]).all():
            continue
        edge = float(row.p_home) - fair_home
        bet_home = edge >= 0
        side_edge = abs(edge)
        odds = row.open_ml_home if bet_home else row.open_ml_away
        won = (row.home_wins == 1) if bet_home else (row.home_wins == 0)
        profit = profit_from_american(odds) if won else -1.0
        fair_open = fair_home if bet_home else fair_away
        fair_close = close_home if bet_home else close_away
        records.append(
            {
                "season": int(row.season),
                "lead_h": float(row.lead_h),
                "edge": side_edge,
                "profit": profit,
                "clv": fair_close - fair_open,
            }
        )
    return pd.DataFrame(records)


def evaluate_grid(fills: pd.DataFrame) -> pd.DataFrame:
    rows = []
    select = fills[fills["season"].isin(SELECT_SEASONS)]
    hold = fills[fills["season"].isin(HOLD_SEASONS)]
    for min_h, max_h in WINDOWS:
        for edge in EDGES:
            def subset(frame: pd.DataFrame) -> pd.DataFrame:
                return frame[
                    frame["edge"].ge(edge)
                    & frame["lead_h"].gt(min_h)
                    & frame["lead_h"].le(max_h)
                ]

            chosen = score_fills(subset(select))
            future = score_fills(subset(hold))
            rows.append(
                {
                    "min_hours": min_h,
                    "max_hours": max_h,
                    "min_edge": edge,
                    "select_n": chosen["n"],
                    "select_roi": chosen["roi"],
                    "select_clv": chosen["clv"],
                    "hold_n": future["n"],
                    "hold_roi": future["roi"],
                    "hold_clv": future["clv"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    print("Building leave-one-season quotes...")
    fills = candidate_fills(load_quotes())
    print(f"  Fills with an open inside a positive lead time: {len(fills)}")
    grid = evaluate_grid(fills)
    eligible = grid[grid["select_n"] >= MIN_SELECT_BETS].copy()
    eligible = eligible.sort_values(
        ["select_clv", "select_roi"],
        ascending=False,
    )
    out = PROJECT_ROOT / "results" / "quote_param_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.sort_values(["min_hours", "max_hours", "min_edge"]).to_csv(out, index=False)
    print(f"Wrote {out}")
    show = eligible.head(12).copy()
    for column in ("min_edge", "select_roi", "select_clv", "hold_roi", "hold_clv"):
        show[column] = (show[column] * 100).round(2)
    print("\nBest cells on 2023-24 and 2024-25, ranked by closing-line move")
    print("Window is hours before puck drop. CLV is how far the close moved toward the side we bought.")
    print(show.to_string(index=False))
    long_lead = grid[grid.apply(lambda row: (row.min_hours, row.max_hours) in LONG_WINDOWS, axis=1)]
    long_lead = long_lead.sort_values(["min_hours", "max_hours", "min_edge"])
    print("\n48 hours to one week")
    print("Only games whose open was actually posted in that window. Most nightly lines are not.")
    show_long = long_lead.copy()
    for column in ("min_edge", "select_roi", "select_clv", "hold_roi", "hold_clv"):
        show_long[column] = (show_long[column] * 100).round(2)
    print(show_long.to_string(index=False))
    if eligible.empty:
        return
    best = eligible.iloc[0]
    print(
        f"\nSelected: edge >= {best.min_edge:.0%}, "
        f"quote from {int(best.min_hours)}h to {int(best.max_hours)}h before puck drop"
    )
    print(
        f"  Selection: n={int(best.select_n)}  ROI={best.select_roi:.1%}  CLV={best.select_clv:.1%}"
    )
    print(
        f"  Holdout:   n={int(best.hold_n)}  ROI={best.hold_roi:.1%}  CLV={best.hold_clv:.1%}"
    )


if __name__ == "__main__":
    main()
