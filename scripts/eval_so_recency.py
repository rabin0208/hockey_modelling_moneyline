"""Compare SO-branch feature families (extra-time recency vs prior shootout rate)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SWEEP_SRC = ROOT / ".so_sweep_src"
sys.path.insert(0, str(SWEEP_SRC))

from evaluate_historical import (  # noqa: E402
    attach_two_way,
    make_model,
    priced_bet,
)
from hockey_moneyline.features import (  # noqa: E402
    EXTRA_TIME_RECENCY_FEATURE_COLUMNS,
    REDUNDANT_GOAL_DIFF_COLUMNS,
)
from hockey_moneyline.prospective_artifact import FROZEN_FEATURE_COLUMNS  # noqa: E402
import create_features as cf  # noqa: E402

OUTPUT = ROOT / "data/derived/so_extra_time_ablation.json"

BASE_WITHOUT_DIFFS = tuple(
    column
    for column in FROZEN_FEATURE_COLUMNS
    if column not in REDUNDANT_GOAL_DIFF_COLUMNS
)
SO_MODEL_FEATURE_COLUMNS = BASE_WITHOUT_DIFFS + EXTRA_TIME_RECENCY_FEATURE_COLUMNS


def _frame_from_source() -> pd.DataFrame:
    cf._SOURCE = None
    cf._SOURCE_STAT = None
    source = cf.load_source(refresh=False)
    rows = list(source.development_rows)
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["game_date"])
    frame["home_wins"] = frame["home_win"].astype(int)
    so_by_id = {
        game.game_id: int(bool(game.decided_in_shootout)) for game in source.games
    }
    ot_by_id = {
        game.game_id: int(bool(game.decided_in_overtime)) for game in source.games
    }
    frame["game_id"] = frame["game_id"].astype(str)
    frame["decided_in_shootout"] = frame["game_id"].map(so_by_id).fillna(0).astype(int)
    frame["decided_in_overtime"] = frame["game_id"].map(ot_by_id).fillna(0).astype(int)
    return frame


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = attach_two_way(df, str(ROOT / "data/closing_odds.csv"), "close")
    df = attach_two_way(df, str(ROOT / "data/opening_odds.csv"), "open")
    df["odds_valid"] = df["close_odds_valid"]
    cols = list(FROZEN_FEATURE_COLUMNS)
    mask = df[cols].notna().all(axis=1)
    for column in EXTRA_TIME_RECENCY_FEATURE_COLUMNS:
        if column in df.columns:
            mask &= df[column].notna()
    return df.loc[mask].copy()


def _eval(df: pd.DataFrame, columns: list[str]) -> dict:
    seasons = sorted(df["season"].unique())
    y_all, p_all, pred_all = [], [], []
    tests = []
    for season in seasons:
        test = df[df["season"] == season]
        train = df[df["season"] != season]
        model = make_model()
        model.fit(train[columns], train["home_wins"].astype(int))
        y = test["home_wins"].astype(int).to_numpy()
        p = model.predict_proba(test[columns])[:, 1]
        pred = (p >= 0.5).astype(int)
        y_all.append(y)
        p_all.append(p)
        pred_all.append(pred)
        tests.append(test.reset_index(drop=True))
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    pred = np.concatenate(pred_all)
    test = pd.concat(tests, ignore_index=True)
    out = {
        "n": int(len(y)),
        "n_shootout": int(test["decided_in_shootout"].sum()),
        "n_overtime": int(test["decided_in_overtime"].sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "feature_count": len(columns),
        "feature_columns": list(columns),
    }
    rec = []
    for actual, pick, p_home, ovalid, oh, oa, cvalid, ch, ca in zip(
        y,
        pred,
        p,
        test["open_odds_valid"],
        test["open_ml_home"],
        test["open_ml_away"],
        test["close_odds_valid"],
        test["close_ml_home"],
        test["close_ml_away"],
    ):
        if not ovalid or pd.isna(oh) or pd.isna(oa):
            continue
        edge, profit = priced_bet(actual, pick, p_home, oh, oa)
        toward = None
        if cvalid and pd.notna(ch) and pd.notna(ca):
            close_edge, _ = priced_bet(actual, pick, p_home, ch, ca)
            toward = bool(close_edge < edge)
        rec.append({"edge": edge, "profit": profit, "toward": toward})
    rec_df = pd.DataFrame(rec)
    out["n_open"] = int(len(rec_df))
    out["open_roi_all_picks"] = float(rec_df["profit"].mean()) if len(rec_df) else None
    for thr in (0.05, 0.10, 0.15):
        key = int(thr * 100)
        subset = rec_df[rec_df["edge"] >= thr]
        out[f"n_open_edge_{key}"] = int(len(subset))
        out[f"open_roi_edge_{key}"] = float(subset["profit"].mean()) if len(subset) else None
        toward = subset["toward"].dropna()
        out[f"toward_edge_{key}"] = float(toward.mean()) if len(toward) else None
    return out


def _print(name: str, metrics: dict) -> None:
    print(
        f"\n=== {name} ({metrics['feature_count']} features) ===\n"
        f"acc={metrics['accuracy']:.3%} ll={metrics['log_loss']:.4f} "
        f"n10={metrics['n_open_edge_10']} roi10={metrics['open_roi_edge_10']:+.1%} "
        f"toward={metrics['toward_edge_10']:.1%}"
    )


def main() -> None:
    print("Building SO-inclusive rows with OT/SO labels (K_SHOOTOUT=5)...")
    frame = _prepare(_frame_from_source())
    missing = [c for c in EXTRA_TIME_RECENCY_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(f"Missing extra-time recency columns: {missing}")
    print(
        f"Rows={len(frame)} SO={int(frame.decided_in_shootout.sum())} "
        f"OT={int(frame.decided_in_overtime.sum())} "
        f"mean home extra rate={frame['home_recent_extra_time_rate'].mean():.3f} "
        f"mean home extra wins={frame['home_recent_extra_time_wins'].mean():.2f}"
    )

    baseline = _eval(frame, list(BASE_WITHOUT_DIFFS))
    _print("baseline (no extra-time feats)", baseline)

    current = _eval(frame, list(SO_MODEL_FEATURE_COLUMNS))
    _print("current (extra-time rate + wins)", current)

    payload = {
        "k_shootout": 5.0,
        "feature_columns": list(SO_MODEL_FEATURE_COLUMNS),
        "baseline_no_extra_time": baseline,
        "current_extra_time_recency": current,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
