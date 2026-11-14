"""Sweep shootout Elo K for the SO-inclusive model.

Compares leave-one-season-out fit and opening-edge strategy metrics on all
moneyline games (including shootouts). Uses the reference SO-inclusive code
under .so_sweep_src with a patched K_SHOOTOUT.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SWEEP_SRC = ROOT / ".so_sweep_src"
sys.path.insert(0, str(SWEEP_SRC))
sys.path.insert(0, str(ROOT / "src"))  # fallback only; sweep src should win for hockey_moneyline

# Prefer sweep package.
if str(SWEEP_SRC) in sys.path:
    # Ensure hockey_moneyline resolves from sweep src.
    while str(ROOT / "src") in sys.path:
        sys.path.remove(str(ROOT / "src"))
    sys.path.insert(0, str(SWEEP_SRC))

from evaluate_historical import (  # noqa: E402
    SEASON_LABELS,
    attach_two_way,
    complete_rows,
    make_model,
    priced_bet,
)
from hockey_moneyline.prospective_artifact import FROZEN_FEATURE_COLUMNS  # noqa: E402
from hockey_moneyline import elo as elo_mod  # noqa: E402
import hockey_moneyline.features as features_mod  # noqa: E402
import hockey_moneyline.unscored_target as unscored_mod  # noqa: E402
import create_features as cf  # noqa: E402

K_VALUES = (0.0, 5.0, 10.0, 15.0, 20.0)
OUTPUT = ROOT / "data/derived/so_elo_k_sweep.json"


def _install_k(k_shootout: float) -> None:
    def updated_ratings(
        home_rating: float,
        away_rating: float,
        home_won: bool,
        *,
        decided_in_shootout: bool = False,
        k: float | None = None,
    ):
        if k is None:
            k = k_shootout if decided_in_shootout else elo_mod.ELO_K
        home_expected = elo_mod.expected_score(home_rating, away_rating)
        home_score = 1.0 if home_won else 0.0
        new_home = home_rating + k * (home_score - home_expected)
        new_away = away_rating + k * ((1.0 - home_score) - (1.0 - home_expected))
        return new_home, new_away

    elo_mod.ELO_K_SHOOTOUT = k_shootout
    elo_mod.updated_ratings = updated_ratings
    features_mod.updated_ratings = updated_ratings
    unscored_mod.updated_ratings = updated_ratings


def _build_feature_frame() -> pd.DataFrame:
    cf._SOURCE = None
    cf._SOURCE_STAT = None
    # Force reload of modules that may cache nothing else
    frame = cf.feature_frame(cf.load_source(refresh=False))
    if "decided_in_shootout" not in frame.columns:
        frame["decided_in_shootout"] = (
            (frame["home_score"] == frame["away_score"]) & frame["home_wins"].notna()
        ).astype(int)
    return frame


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["game_id"] = df["game_id"].astype(str)
    df = attach_two_way(df, str(ROOT / "data/closing_odds.csv"), "close")
    df = attach_two_way(df, str(ROOT / "data/opening_odds.csv"), "open")
    df["odds_valid"] = df["close_odds_valid"]
    return complete_rows(df)


def _loso_metrics(df: pd.DataFrame) -> dict:
    cols = list(FROZEN_FEATURE_COLUMNS)
    seasons = sorted(df["season"].unique())
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    test_parts: list[pd.DataFrame] = []

    for season in seasons:
        test = df[df["season"] == season].copy()
        train = df[df["season"] != season].copy()
        model = make_model()
        model.fit(train[cols], train["home_wins"].astype(int))
        y = test["home_wins"].astype(int).to_numpy()
        p = model.predict_proba(test[cols])[:, 1]
        pred = (p >= 0.5).astype(int)
        y_all.append(y)
        p_all.append(p)
        pred_all.append(pred)
        test_parts.append(test.reset_index(drop=True))

    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    pred = np.concatenate(pred_all)
    test = pd.concat(test_parts, ignore_index=True)

    metrics = {
        "n": int(len(y)),
        "n_shootout": int(test["decided_in_shootout"].fillna(0).astype(int).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }

    open_rows = []
    for actual, pick, p_home, ovalid, oh, oa, cvalid, ch, ca in zip(
        y,
        pred,
        p,
        test["open_odds_valid"].to_numpy(),
        test["open_ml_home"].to_numpy(),
        test["open_ml_away"].to_numpy(),
        test["close_odds_valid"].to_numpy(),
        test["close_ml_home"].to_numpy(),
        test["close_ml_away"].to_numpy(),
    ):
        if not ovalid or pd.isna(oh) or pd.isna(oa):
            continue
        open_edge, open_profit = priced_bet(actual, pick, p_home, oh, oa)
        toward = None
        if cvalid and pd.notna(ch) and pd.notna(ca):
            close_edge, _ = priced_bet(actual, pick, p_home, ch, ca)
            toward = bool(close_edge < open_edge)
        open_rows.append(
            {
                "edge": float(open_edge),
                "profit": float(open_profit),
                "toward": toward,
            }
        )
    rec = pd.DataFrame(open_rows)
    metrics["n_open"] = int(len(rec))
    metrics["open_roi_all_picks"] = float(rec["profit"].mean()) if len(rec) else None
    for thr in (0.05, 0.10, 0.15):
        subset = rec[rec["edge"] >= thr]
        key = int(thr * 100)
        metrics[f"n_open_edge_{key}"] = int(len(subset))
        metrics[f"open_roi_edge_{key}"] = (
            float(subset["profit"].mean()) if len(subset) else None
        )
        toward = subset["toward"].dropna()
        metrics[f"toward_edge_{key}"] = float(toward.mean()) if len(toward) else None
    return metrics


def main() -> None:
    results = []
    print(f"Sweeping SO Elo K over {K_VALUES}")
    for k in K_VALUES:
        print(f"\n=== K_SHOOTOUT = {k:g} ===")
        _install_k(k)
        frame = _build_feature_frame()
        prepared = _prepare(frame)
        metrics = _loso_metrics(prepared)
        metrics["k_shootout"] = k
        results.append(metrics)
        print(
            f"  n={metrics['n']} so={metrics['n_shootout']} "
            f"acc={metrics['accuracy']:.3%} ll={metrics['log_loss']:.4f} "
            f"brier={metrics['brier']:.4f}"
        )
        print(
            f"  open>=10%: n={metrics['n_open_edge_10']} "
            f"roi={metrics['open_roi_edge_10']:+.1%} "
            f"toward={metrics['toward_edge_10']:.1%}"
            if metrics["n_open_edge_10"]
            else "  open>=10%: none"
        )

    # Score: usable strategy wants enough >=10% bets and non-terrible ROI,
    # without wrecking log loss. Rank by a simple utility.
    scored = []
    for row in results:
        n10 = row["n_open_edge_10"] or 0
        roi10 = row["open_roi_edge_10"] if row["open_roi_edge_10"] is not None else -1.0
        toward = row["toward_edge_10"] if row["toward_edge_10"] is not None else 0.0
        # Prefer more tickets than the too-quiet K=10 reference, with ROI not awful.
        utility = (
            -row["log_loss"] * 10.0
            + (roi10 * 2.0)
            + (toward * 0.5)
            + min(n10, 450) / 450.0
        )
        scored.append({**row, "utility": utility})

    ranked = sorted(scored, key=lambda item: item["utility"], reverse=True)
    payload = {
        "k_values": list(K_VALUES),
        "results": scored,
        "recommended_k_shootout": ranked[0]["k_shootout"],
        "recommendation_note": (
            "Utility balances log loss, >=10% open ROI, close-toward rate, "
            "and number of >=10% tickets (capped)."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT}")
    print(f"Recommended K_SHOOTOUT = {ranked[0]['k_shootout']:g}")
    print(
        f"{'K':>6} {'acc':>7} {'ll':>8} {'brier':>8} {'n10':>5} {'roi10':>8} {'toward10':>9} {'util':>7}"
    )
    for row in scored:
        print(
            f"{row['k_shootout']:6.1f} {row['accuracy']:7.3%} {row['log_loss']:8.4f} "
            f"{row['brier']:8.4f} {row['n_open_edge_10']:5d} "
            f"{(row['open_roi_edge_10'] or 0):+8.1%} "
            f"{(row['toward_edge_10'] or 0):9.1%} {row['utility']:7.3f}"
        )


if __name__ == "__main__":
    main()
