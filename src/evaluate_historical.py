"""
Leave-one-season-out accuracy and betting ROI for the best model type.

Trains optimized logistic regression on all seasons except the test season,
then scores that held-out year. ROI assumes a $1 moneyline bet on the
model's predicted winner at the Action Network closing price. When
data/opening_odds.csv exists, the same picks are also scored at the
opening price.
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

EXCLUDE_COLS = [
    'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
    'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score',
    'close_ml_home', 'close_ml_away', 'close_odds_valid', 'odds_valid',
    'open_ml_home', 'open_ml_away', 'open_odds_valid',
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


def profit_from_american(odds, stake=1.0):
    """Profit (not including stake return) if the bet wins."""
    odds = float(odds)
    if odds < 0:
        return stake * (100.0 / abs(odds))
    return stake * (odds / 100.0)


def attach_two_way(df, path, prefix):
    """Join a two-sided American moneyline file onto feature rows."""
    home_col = f'{prefix}_ml_home'
    away_col = f'{prefix}_ml_away'
    valid_col = f'{prefix}_odds_valid'
    df = df.copy()
    df[home_col] = np.nan
    df[away_col] = np.nan
    if os.path.exists(path):
        odds = pd.read_csv(path, usecols=['game_id', home_col, away_col])
        odds = odds.drop_duplicates('game_id')
        df['_gid'] = df['game_id'].astype(str)
        odds['_gid'] = odds['game_id'].astype(str)
        df = df.drop(columns=[home_col, away_col])
        df = df.merge(odds[['_gid', home_col, away_col]], on='_gid', how='left')
        df = df.drop(columns=['_gid'])
    df[home_col] = pd.to_numeric(df[home_col], errors='coerce')
    df[away_col] = pd.to_numeric(df[away_col], errors='coerce')
    reasonable = df[home_col].abs().between(100, 2000) & df[away_col].abs().between(100, 2000)
    df[valid_col] = df[home_col].notna() & df[away_col].notna() & reasonable
    return df


def devig(home_ml, away_ml):
    home = american_to_implied(home_ml)
    away = american_to_implied(away_ml)
    total = home + away
    if total <= 0:
        return np.nan, np.nan
    return home / total, away / total


def priced_bet(actual, pick_home, p_home, home_ml, away_ml):
    """Edge and $1 profit for the model pick at a two-sided American line."""
    fair_home, fair_away = devig(home_ml, away_ml)
    bet_home = bool(pick_home)
    odds = home_ml if bet_home else away_ml
    won = (actual == 1) if bet_home else (actual == 0)
    edge = (p_home - fair_home) if bet_home else ((1.0 - p_home) - fair_away)
    profit = profit_from_american(odds) if won else -1.0
    return edge, profit


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


def bet_results(y_true, pred, home_prob, home_ml, away_ml):
    """$1 on the model's pick at the side's American price."""
    records = []
    for actual, pick_home, p_home, home_price, away_price in zip(
        y_true, pred, home_prob, home_ml, away_ml
    ):
        if pd.isna(home_price) or pd.isna(away_price):
            continue
        edge, profit = priced_bet(actual, pick_home, p_home, home_price, away_price)
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


def market_favorite_accuracy(y_true, home_ml, away_ml):
    """Home is the favorite when its vigged price is shorter than the away price."""
    home_ml = np.asarray(home_ml, dtype=float)
    away_ml = np.asarray(away_ml, dtype=float)
    y_true = np.asarray(y_true)
    mask = np.isfinite(home_ml) & np.isfinite(away_ml)
    if mask.sum() == 0:
        return np.nan
    home_imp = np.array([american_to_implied(price) for price in home_ml[mask]])
    away_imp = np.array([american_to_implied(price) for price in away_ml[mask]])
    return float(np.mean((home_imp >= away_imp).astype(int) == y_true[mask]))


def evaluate_season(train_df, test_df, feature_cols):
    model = make_model()
    model.fit(train_df[feature_cols], train_df['home_wins'].astype(int))

    y_true = test_df['home_wins'].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    odds_mask = test_df['close_odds_valid'].to_numpy()
    bets = bet_results(
        y_true[odds_mask],
        pred[odds_mask],
        proba[odds_mask],
        test_df.loc[odds_mask, 'close_ml_home'].to_numpy(),
        test_df.loc[odds_mask, 'close_ml_away'].to_numpy(),
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
            y_true[odds_mask],
            test_df.loc[odds_mask, 'close_ml_home'],
            test_df.loc[odds_mask, 'close_ml_away'],
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
    df = attach_two_way(df, os.path.join('data', 'closing_odds.csv'), 'close')
    df = attach_two_way(df, os.path.join('data', 'opening_odds.csv'), 'open')
    df['odds_valid'] = df['close_odds_valid']
    print(f"  Closing moneylines: {df['close_odds_valid'].sum()}/{len(df)}")
    print(f"  Opening moneylines: {df['open_odds_valid'].sum()}/{len(df)}")

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
    held_out_home_ml = []
    held_out_away_ml = []

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
        odds_valid = test_df['close_odds_valid'].to_numpy()
        home_ml = test_df['close_ml_home'].to_numpy()
        away_ml = test_df['close_ml_away'].to_numpy()
        open_valid = test_df['open_odds_valid'].to_numpy()
        open_home = test_df['open_ml_home'].to_numpy()
        open_away = test_df['open_ml_away'].to_numpy()
        held_out_true.append(y_true)
        held_out_pred.append(pred)
        held_out_proba.append(proba)
        held_out_odds_valid.append(odds_valid)
        held_out_home_ml.append(home_ml)
        held_out_away_ml.append(away_ml)
        for actual, pick, p_home, valid, ml, away, ovalid, oh, oa in zip(
            y_true, pred, proba, odds_valid, home_ml, away_ml, open_valid, open_home, open_away
        ):
            edge = np.nan
            profit = np.nan
            if valid and pd.notna(ml) and pd.notna(away):
                edge, profit = priced_bet(actual, pick, p_home, ml, away)
            open_edge = np.nan
            open_profit = np.nan
            if ovalid and pd.notna(oh) and pd.notna(oa):
                open_edge, open_profit = priced_bet(actual, pick, p_home, oh, oa)
            game_rows.append({
                'season': label,
                'home_win': int(actual),
                'pick_home': int(pick),
                'p_home': float(p_home),
                'odds_valid': bool(valid),
                'edge': edge,
                'profit': profit,
                'open_odds_valid': bool(ovalid),
                'open_edge': open_edge,
                'open_profit': open_profit,
            })

    y_all = np.concatenate(held_out_true)
    pred_all = np.concatenate(held_out_pred)
    proba_all = np.concatenate(held_out_proba)
    odds_all = np.concatenate(held_out_odds_valid)
    home_all = np.concatenate(held_out_home_ml)
    away_all = np.concatenate(held_out_away_ml)
    overall_bets = bet_results(
        y_all[odds_all], pred_all[odds_all], proba_all[odds_all], home_all[odds_all], away_all[odds_all]
    )
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
        'market_favorite_acc': market_favorite_accuracy(
            y_all[odds_all], home_all[odds_all], away_all[odds_all]
        ),
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
    print_opening_comparison(pd.DataFrame(game_rows))


def print_opening_comparison(games):
    """Same model winners, priced at the close and at the open."""
    both = games[games['odds_valid'] & games['open_odds_valid']].copy()
    print("\nOpening vs closing, same predicted winners")
    if both.empty:
        print("  No games have both prices.")
        return
    print(f"  Overlap: {len(both)} games")
    for season, group in list(both.groupby('season', sort=False)) + [('Overlap', both)]:
        close_roi = group['profit'].sum() / len(group)
        open_roi = group['open_profit'].sum() / len(group)
        print(
            f"  {season}: n={len(group)}  "
            f"Closing ROI={close_roi:.1%}  Opening ROI={open_roi:.1%}"
        )


if __name__ == '__main__':
    main()
