"""
Process Kaggle NHL extensive data to extract odds.

The dataset appears to be one day ahead, so we adjust dates.
Creates a file with: date, season, home_team, away_team, spread, over_under, home_moneyline

Note: The 'favorite_moneyline' column in the source data is actually the home team's moneyline odds,
not the favorite's odds. We rename it to 'home_moneyline' for clarity.
"""

import pandas as pd
import os
from datetime import timedelta


def main():
    """Process nhl_data_extensive.csv to create odds file."""
    print("=" * 60)
    print("Kaggle NHL Odds Processor")
    print("=" * 60)
    
    input_file = 'data/nhl_data_extensive.csv'
    output_file = 'data/kaggle_odds.csv'
    
    if not os.path.exists(input_file):
        print(f"✗ File not found: {input_file}")
        return
    
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows")
    
    # Extract odds columns
    odds_columns = ['game_id', 'date', 'season', 'team_name', 'is_home', 
                    'spread', 'over_under', 'favorite_moneyline']
    
    # Check which columns exist
    available_cols = [col for col in odds_columns if col in df.columns]
    missing_cols = [col for col in odds_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️  Missing columns: {missing_cols}")
        return
    
    # Process the data
    print("\nProcessing odds data...")
    print("  Note: Adjusting dates by -1 day (Kaggle data appears to be one day ahead)")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Adjust date by -1 day
    df['date'] = df['date'] - timedelta(days=1)
    
    # Group by game_id to combine home and away teams into one row
    processed_games = []
    
    for game_id, game_data in df.groupby('game_id'):
        # Get home and away team rows
        home_row = game_data[game_data['is_home'] == 1].iloc[0] if len(game_data[game_data['is_home'] == 1]) > 0 else None
        away_row = game_data[game_data['is_home'] == 0].iloc[0] if len(game_data[game_data['is_home'] == 0]) > 0 else None
        
        if home_row is not None and away_row is not None:
            # Both teams should have same odds (game-level)
            # Note: favorite_moneyline is actually the home team's moneyline odds
            processed_games.append({
                'game_id': game_id,
                'date': home_row['date'].date(),  # Convert to date only
                'season': home_row['season'],
                'home_team': home_row['team_name'],
                'away_team': away_row['team_name'],
                'spread': home_row['spread'],
                'over_under': home_row['over_under'],
                'home_moneyline': home_row['favorite_moneyline']  # Actually home team's odds
            })
    
    odds_df = pd.DataFrame(processed_games)
    
    # Remove rows with missing odds
    before = len(odds_df)
    odds_df = odds_df.dropna(subset=['spread', 'over_under', 'home_moneyline'], how='all')
    after = len(odds_df)
    
    print(f"\nProcessed odds data:")
    print(f"  Total games: {len(odds_df)}")
    print(f"  Games with odds: {after} (dropped {before - after} games without odds)")
    print(f"  Date range: {odds_df['date'].min()} to {odds_df['date'].max()}")
    print(f"  Seasons: {sorted(odds_df['season'].unique())}")
    
    # Save to CSV
    odds_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()

