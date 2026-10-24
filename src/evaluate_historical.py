"""
Leave-one-season-out accuracy and betting ROI for the best model type.

Trains optimized logistic regression on all seasons except the test season,
then scores that held-out year. ROI assumes a $1 moneyline bet on the
model's predicted winner whenever a real closing line is available.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_features import get_odds_for_game

EXCLUDE_COLS = [
    'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
    'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score',
    'home_moneyline', 'odds_valid',
]

SEASON_LABELS = {
    20212022: '2021-22',
    20222023: '2022-23',
    20232024: '2023-24',
    20242025: '2024-25',
    20252026: '2025-26',
    20262027: '2026-27',
}


def american_to_implied(odds):
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def implied_to_american(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def profit_from_american(odds, stake=1.0):
    """Profit (not including stake return) if the bet wins."""
    odds = float(odds)
    if odds < 0:
        return stake * (100.0 / abs(odds))
    return stake * (odds / 100.0)


def attach_odds(df):
    """Join Kaggle home moneylines onto feature rows."""
    lines = []
    for row in df.itertuples(index=False):
        odds = get_odds_for_game(
            row.game_id, row.date, row.home_team_name, row.away_team_name
        )
        lines.append(odds['home_moneyline'] if odds else np.nan)
    df = df.copy()
    df['home_moneyline'] = pd.to_numeric(lines, errors='coerce')
    return df


def mark_valid_odds(df):
    """Drop placeholder / extreme lines that would distort ROI."""
    df = df.copy()
    season_placeholder_rate = df.groupby('season')['home_moneyline'].transform(
        lambda s: (s == -105).mean()
    )
    abs_line = df['home_moneyline'].abs()
    reasonable = abs_line.between(101, 800)
    not_placeholder = ~((df['home_moneyline'] == -105) & (season_placeholder_rate > 0.5))
    df['odds_valid'] = df['home_moneyline'].notna() & reasonable & not_placeholder
    return df


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.001,
            max_iter=5000,
            random_state=42,
            solver='lbfgs',
        ),
    )


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def complete_rows(df):
    cols = feature_columns(df)
    return df[df[cols].notna().all(axis=1)].copy()


def bet_results(y_true, pred, home_prob, home_ml):
    """$1 on the model's pick; away line inferred as the no-vig complement."""
    records = []
    for actual, pick_home, p_home, ml in zip(y_true, pred, home_prob, home_ml):
        if pd.isna(ml):
            continue
        implied_home = american_to_implied(ml)
        away_ml = implied_to_american(1.0 - implied_home)
        bet_home = bool(pick_home)
        odds = ml if bet_home else away_ml
        won = (actual == 1) if bet_home else (actual == 0)
        profit = profit_from_american(odds) if won else -1.0
        edge = (p_home - implied_home) if bet_home else ((1.0 - p_home) - (1.0 - implied_home))
        records.append({
            'profit': profit,
            'value_bet': edge > 0,
            'value_profit': profit if edge > 0 else 0.0,
        })
    if not records:
        return {
            'n_bets': 0,
            'profit': np.nan,
            'roi': np.nan,
            'n_value_bets': 0,
            'value_profit': np.nan,
            'value_roi': np.nan,
        }
    n = len(records)
    profit = sum(r['profit'] for r in records)
    n_value = sum(r['value_bet'] for r in records)
    value_profit = sum(r['value_profit'] for r in records)
    return {
        'n_bets': n,
        'profit': profit,
        'roi': profit / n,
        'n_value_bets': int(n_value),
        'value_profit': value_profit if n_value else np.nan,
        'value_roi': (value_profit / n_value) if n_value else np.nan,
    }


def market_favorite_accuracy(y_true, home_ml):
    """Home is the favorite when the moneyline is negative."""
    mask = pd.notna(home_ml)
    if mask.sum() == 0:
        return np.nan
    fav_home = home_ml[mask] < 0
    actual = np.asarray(y_true)[mask]
    return float(np.mean((fav_home.astype(int) == actual)))


def evaluate_season(train_df, test_df, feature_cols):
    model = make_model()
    model.fit(train_df[feature_cols], train_df['home_wins'].astype(int))

    y_true = test_df['home_wins'].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    odds_mask = test_df['odds_valid'].to_numpy()
    bets = bet_results(
        y_true[odds_mask],
        pred[odds_mask],
        proba[odds_mask],
        test_df.loc[odds_mask, 'home_moneyline'].to_numpy(),
    )

    return {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_with_odds': int(odds_mask.sum()),
        'accuracy': accuracy_score(y_true, pred),
        'roc_auc': roc_auc_score(y_true, proba),
        'log_loss': log_loss(y_true, proba),
        'brier': brier_score_loss(y_true, proba),
        'baseline_home_acc': float(y_true.mean()),
        'market_favorite_acc': market_favorite_accuracy(
            y_true[odds_mask], test_df.loc[odds_mask, 'home_moneyline']
        ) if odds_mask.any() else np.nan,
        **bets,
    }


def main():
    print("=" * 72)
    print("Historical accuracy and ROI (leave-one-season-out)")
    print("=" * 72)

    df = pd.read_csv('data/game_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} games")

    print("\nMatching moneylines...")
    df = attach_odds(df)
    df = mark_valid_odds(df)
    print(f"  Odds matched: {df['home_moneyline'].notna().sum()}/{len(df)}")
    print(f"  Usable moneylines: {df['odds_valid'].sum()}/{len(df)}")

    df = complete_rows(df)
    feature_cols = feature_columns(df)
    print(f"  Complete-feature games: {len(df)}")

    seasons = sorted(df['season'].unique())
    rows = []
    game_rows = []
    held_out_true = []
    held_out_pred = []
    held_out_proba = []
    held_out_odds_valid = []
    held_out_ml = []

    for season in seasons:
        test_df = df[df['season'] == season]
        train_df = df[df['season'] != season]
        label = SEASON_LABELS.get(int(season), str(season))
        print(f"\n{label}: train {len(train_df)} / test {len(test_df)}")
        metrics = evaluate_season(train_df, test_df, feature_cols)
        print(
            f"  acc={metrics['accuracy']:.1%}  "
            f"home baseline={metrics['baseline_home_acc']:.1%}  "
            f"ROI={metrics['roi']:.1%}" if pd.notna(metrics['roi']) else
            f"  acc={metrics['accuracy']:.1%}  home baseline={metrics['baseline_home_acc']:.1%}  ROI=n/a"
        )
        rows.append({'season': label, 'season_id': int(season), **metrics})

        model = make_model()
        model.fit(train_df[feature_cols], train_df['home_wins'].astype(int))
        y_true = test_df['home_wins'].astype(int).to_numpy()
        pred = model.predict(test_df[feature_cols])
        proba = model.predict_proba(test_df[feature_cols])[:, 1]
        odds_valid = test_df['odds_valid'].to_numpy()
        home_ml = test_df['home_moneyline'].to_numpy()
        held_out_true.append(y_true)
        held_out_pred.append(pred)
        held_out_proba.append(proba)
        held_out_odds_valid.append(odds_valid)
        held_out_ml.append(home_ml)
        for actual, pick, p_home, valid, ml in zip(y_true, pred, proba, odds_valid, home_ml):
            edge = np.nan
            profit = np.nan
            if valid and pd.notna(ml):
                implied_home = american_to_implied(ml)
                bet_home = bool(pick)
                odds = ml if bet_home else implied_to_american(1.0 - implied_home)
                won = (actual == 1) if bet_home else (actual == 0)
                edge = (p_home - implied_home) if bet_home else (implied_home - p_home)
                profit = profit_from_american(odds) if won else -1.0
            game_rows.append({
                'season': label,
                'home_win': int(actual),
                'pick_home': int(pick),
                'p_home': float(p_home),
                'odds_valid': bool(valid),
                'edge': edge,
                'profit': profit,
            })

    y_all = np.concatenate(held_out_true)
    pred_all = np.concatenate(held_out_pred)
    proba_all = np.concatenate(held_out_proba)
    odds_all = np.concatenate(held_out_odds_valid)
    ml_all = np.concatenate(held_out_ml)
    overall_bets = bet_results(y_all[odds_all], pred_all[odds_all], proba_all[odds_all], ml_all[odds_all])
    overall = {
        'season': 'Overall',
        'season_id': 0,
        'n_train': len(df),
        'n_test': len(df),
        'n_with_odds': int(odds_all.sum()),
        'accuracy': accuracy_score(y_all, pred_all),
        'roc_auc': roc_auc_score(y_all, proba_all),
        'log_loss': log_loss(y_all, proba_all),
        'brier': brier_score_loss(y_all, proba_all),
        'baseline_home_acc': float(y_all.mean()),
        'market_favorite_acc': market_favorite_accuracy(y_all[odds_all], ml_all[odds_all]),
        **overall_bets,
    }
    rows.append(overall)

    out = pd.DataFrame(rows)
    # Friendly column order / names for the exported table
    export = pd.DataFrame({
        'season': out['season'],
        'n_games': out['n_test'],
        'n_with_odds': out['n_with_odds'],
        'accuracy_pct': (out['accuracy'] * 100).round(2),
        'home_baseline_pct': (out['baseline_home_acc'] * 100).round(2),
        'market_favorite_pct': (out['market_favorite_acc'] * 100).round(2),
        'roc_auc': out['roc_auc'].round(4),
        'n_bets': out['n_bets'],
        'profit_1_unit': out['profit'].round(2),
        'roi_pct': (out['roi'] * 100).round(2),
        'n_value_bets': out['n_value_bets'],
        'value_profit_1_unit': out['value_profit'].round(2),
        'value_roi_pct': (out['value_roi'] * 100).round(2),
    })

    output_path = 'data/historical_accuracy_roi.csv'
    export.to_csv(output_path, index=False)
    games_path = 'data/historical_bet_games.csv'
    pd.DataFrame(game_rows).to_csv(games_path, index=False)
    print("\n" + "=" * 72)
    print(export.to_string(index=False))
    print(f"\n✓ Saved {output_path}")
    print(f"✓ Saved {games_path}")


if __name__ == '__main__':
    main()
