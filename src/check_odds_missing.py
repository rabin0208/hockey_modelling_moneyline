"""Check missing odds data in game_features.csv"""

import pandas as pd
import os

def main():
    print("=" * 60)
    print("Odds Data Missing Values Check")
    print("=" * 60)
    
    features_file = 'data/game_features.csv'
    if not os.path.exists(features_file):
        print(f"✗ File not found: {features_file}")
        print("   Run create_features.py first to generate features")
        return
    
    df = pd.read_csv(features_file)
    print(f"\nTotal games: {len(df)}")
    
    # Check odds columns
    odds_cols = ['spread', 'over_under', 'home_moneyline', 'home_is_favorite']
    
    print(f"\nMissing values for odds features:")
    print("-" * 60)
    
    for col in odds_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = (missing / len(df)) * 100
            available = len(df) - missing
            print(f"  {col}:")
            print(f"    Missing: {missing}/{len(df)} ({pct:.1f}%)")
            print(f"    Available: {available}/{len(df)} ({100-pct:.1f}%)")
        else:
            print(f"  {col}: Column not found in features")
    
    # Check games with ALL odds missing vs SOME missing
    print(f"\n" + "=" * 60)
    print("Games with odds data:")
    print("=" * 60)
    
    if all(col in df.columns for col in odds_cols):
        missing_all = df[odds_cols].isna().all(axis=1)
        missing_some = df[odds_cols].isna().any(axis=1) & ~missing_all
        complete = ~df[odds_cols].isna().any(axis=1)
        
        print(f"Games with ALL odds features: {complete.sum()} ({complete.sum()/len(df)*100:.1f}%)")
        print(f"Games with SOME odds features missing: {missing_some.sum()} ({missing_some.sum()/len(df)*100:.1f}%)")
        print(f"Games with NO odds features: {missing_all.sum()} ({missing_all.sum()/len(df)*100:.1f}%)")
        
        # Check date range of missing vs available
        if missing_all.sum() > 0:
            missing_games = df[missing_all]
            print(f"\nGames WITHOUT odds data:")
            print(f"  Date range: {missing_games['date'].min()} to {missing_games['date'].max()}")
            print(f"  Seasons: {sorted(missing_games['season'].unique())}")
        
        if complete.sum() > 0:
            complete_games = df[complete]
            print(f"\nGames WITH complete odds data:")
            print(f"  Date range: {complete_games['date'].min()} to {complete_games['date'].max()}")
            print(f"  Seasons: {sorted(complete_games['season'].unique())}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()

