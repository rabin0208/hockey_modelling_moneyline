"""
Predict outcomes for upcoming NHL games using trained models.

This script:
1. Loads a saved model (logistic regression or random forest)
2. Fetches upcoming games from the NHL API
3. Creates features for those games
4. Makes predictions with probabilities
5. Displays results in a readable format
"""

import os
import sys
import pickle
import pandas as pd
from datetime import datetime, timedelta, date
from nhlpy import NHLClient

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_features import (
    calculate_team_stats_up_to_date,
    calculate_recent_form,
    calculate_head_to_head,
    calculate_nst_stats_up_to_date,
    match_team_name_to_nst,
    get_season_from_date
)


def load_model(model_path):
    """Load a saved model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle different model formats
    if isinstance(model_data, dict):
        # Format from train_model.py or optimize_model.py (includes feature_columns)
        model = model_data['model']
        feature_columns = model_data.get('feature_columns', None)
    else:
        # Format from optimize_random_forest.py or train_random_forest.py (just the model)
        model = model_data
        # Try to infer feature columns from training data
        feature_columns = None
    
    print("✓ Model loaded")
    return model, feature_columns


def fetch_upcoming_games(client, days_ahead=7):
    """Fetch upcoming games from NHL API."""
    print(f"\nFetching upcoming games (next {days_ahead} days)...")
    
    today = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    
    all_games = []
    current_date = today
    
    while current_date <= end_date:
        try:
            # Convert date to string format (YYYY-MM-DD) for API
            date_str = current_date.strftime('%Y-%m-%d')
            schedule = client.schedule.daily_schedule(date=date_str)
            
            # Handle different response formats
            if isinstance(schedule, dict):
                if 'games' in schedule:
                    games = schedule['games']
                elif 'dates' in schedule and len(schedule['dates']) > 0:
                    # Sometimes the API returns dates array
                    games = schedule['dates'][0].get('games', [])
                else:
                    games = []
            elif isinstance(schedule, list):
                games = schedule
            else:
                games = []
            
            # Debug: print what we got
            if len(games) > 0:
                print(f"  {date_str}: Found {len(games)} games")
                # Check first game structure for debugging
                if len(all_games) == 0 and len(games) > 0:
                    first_game = games[0]
                    status = first_game.get('status', {})
                    print(f"    Sample game status: {status.get('abstractGameState', 'N/A')} / {status.get('detailedState', 'N/A')}")
            
            for game in games:
                # Ensure game has a date - use the date_str we're iterating with
                if 'gameDate' not in game or not game.get('gameDate'):
                    game['gameDate'] = date_str
                
                # Get game status - try multiple ways
                game_status = game.get('status', {})
                if not game_status:
                    # Sometimes status is at top level
                    abstract_state = game.get('abstractGameState', '')
                    detailed_state = game.get('detailedState', '')
                else:
                    abstract_state = game_status.get('abstractGameState', '')
                    detailed_state = game_status.get('detailedState', '')
                
                # Get games that are scheduled/preview (not finished)
                # Be more lenient - include any game that's not 'Final' or 'Live'
                if abstract_state not in ['Final', 'Live', 'Official']:
                    all_games.append(game)
        except Exception as e:
            print(f"  Warning: Could not fetch games for {current_date}: {e}")
            import traceback
            traceback.print_exc()
        
        current_date += timedelta(days=1)
    
    print(f"  Found {len(all_games)} upcoming games")
    return all_games


def parse_game_data(game):
    """Parse game data from NHL API format."""
    try:
        game_id = game.get('gamePk')
        game_date = game.get('gameDate', '')
        
        # Parse date - handle multiple formats
        if not game_date:
            # Try alternative date fields
            game_date = game.get('startTimeUTC', '')[:10] if game.get('startTimeUTC') else ''
        
        if isinstance(game_date, str):
            # Parse ISO format date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
            if 'T' in game_date:
                game_date = game_date.split('T')[0]
            game_date = pd.to_datetime(game_date).date()
        elif isinstance(game_date, (datetime, pd.Timestamp)):
            game_date = pd.to_datetime(game_date).date()
        else:
            print(f"  Warning: Could not parse date for game {game_id}: {game_date}")
            return None
        
        if pd.isna(game_date) or game_date is None:
            print(f"  Warning: Invalid date for game {game_id}")
            return None
        
        # Get teams - try multiple formats
        teams = game.get('teams', {})
        
        # Try different API response formats
        home_team = None
        away_team = None
        
        if 'home' in teams:
            home_team = teams.get('home', {})
            # Sometimes team info is nested
            if 'team' in home_team:
                home_team = home_team.get('team', {})
        elif 'homeTeam' in game:
            home_team = game.get('homeTeam', {})
        
        if 'away' in teams:
            away_team = teams.get('away', {})
            # Sometimes team info is nested
            if 'team' in away_team:
                away_team = away_team.get('team', {})
        elif 'awayTeam' in game:
            away_team = game.get('awayTeam', {})
        
        # Extract team ID
        home_team_id = home_team.get('id') if home_team else None
        away_team_id = away_team.get('id') if away_team else None
        
        # Extract team name - try multiple formats
        home_team_name = 'Unknown'
        away_team_name = 'Unknown'
        
        if home_team:
            # Try different name formats
            home_team_name = (
                home_team.get('name') or
                home_team.get('teamName') or
                (home_team.get('placeName', {}).get('default', '') + ' ' + home_team.get('commonName', {}).get('default', '')).strip() or
                'Unknown'
            )
        
        if away_team:
            away_team_name = (
                away_team.get('name') or
                away_team.get('teamName') or
                (away_team.get('placeName', {}).get('default', '') + ' ' + away_team.get('commonName', {}).get('default', '')).strip() or
                'Unknown'
            )
        
        if home_team_id is None or away_team_id is None:
            print(f"  Warning: Missing team IDs for game {game_id}")
            return None
        
        if home_team_name == 'Unknown' or away_team_name == 'Unknown':
            print(f"  Warning: Could not extract team names for game {game_id}")
            print(f"    Home team structure: {list(home_team.keys()) if home_team else 'None'}")
            print(f"    Away team structure: {list(away_team.keys()) if away_team else 'None'}")
        
        return {
            'game_id': game_id,
            'date': game_date,
            'home_team_id': home_team_id,
            'home_team_name': home_team_name,
            'away_team_id': away_team_id,
            'away_team_name': away_team_name,
            'home_wins': None,  # Not yet played
            'home_score': None,
            'away_score': None,
            'season': f"{game_date.year}_{game_date.year+1}" if game_date.month >= 10 else f"{game_date.year-1}_{game_date.year}"
        }
    except Exception as e:
        print(f"  Warning: Could not parse game {game.get('gamePk', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_features_for_game(games_df, game, feature_columns):
    """Create features for a single upcoming game."""
    game_date = game['date']
    home_team_id = game['home_team_id']
    away_team_id = game['away_team_id']
    home_team_name = game['home_team_name']
    away_team_name = game['away_team_name']
    
    # Ensure game_date is a date object for comparison
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()
    elif not isinstance(game_date, date):
        print(f"    Debug: Invalid game_date type: {type(game_date)}, value: {game_date}")
        return None
    
    # Check if date is valid (NaT check)
    try:
        if pd.isna(game_date) or game_date is None:
            print(f"    Debug: game_date is NaT or None for {home_team_name} vs {away_team_name}")
            return None
    except (TypeError, ValueError):
        # If pd.isna fails, the date is probably valid
        pass
    
    # Get the season from the game date (for season-only stats)
    game_season = get_season_from_date(game_date)
    
    # Ensure games_df date column is date type
    if games_df['date'].dtype != 'object' or isinstance(games_df['date'].iloc[0], str):
        games_df = games_df.copy()
        games_df['date'] = pd.to_datetime(games_df['date']).dt.date
    
    # Convert team IDs to same type (might be int vs float)
    home_team_id = int(home_team_id) if home_team_id is not None else None
    away_team_id = int(away_team_id) if away_team_id is not None else None
    
    # Ensure games_df team IDs are int
    if games_df['home_team_id'].dtype != 'int64':
        games_df = games_df.copy()
        games_df['home_team_id'] = games_df['home_team_id'].astype('Int64')
        games_df['away_team_id'] = games_df['away_team_id'].astype('Int64')
    
    # Debug: Check if team IDs exist in historical data for this season
    home_games = games_df[
        ((games_df['home_team_id'] == home_team_id) | (games_df['away_team_id'] == home_team_id)) &
        (games_df['date'] < game_date) &
        (games_df['season'] == game_season) &
        (games_df['home_wins'].notna())
    ]
    away_games = games_df[
        ((games_df['home_team_id'] == away_team_id) | (games_df['away_team_id'] == away_team_id)) &
        (games_df['date'] < game_date) &
        (games_df['season'] == game_season) &
        (games_df['home_wins'].notna())
    ]
    
    # Debug output for first game
    if len(home_games) == 0 and home_team_id is not None:
        # Check if team ID exists at all (regardless of date)
        all_home_games = games_df[
            ((games_df['home_team_id'] == home_team_id) | (games_df['away_team_id'] == home_team_id))
        ]
        if len(all_home_games) == 0:
            print(f"    Debug: Home team ID {home_team_id} ({home_team_name}) not found in historical data at all")
            print(f"    Debug: Sample historical team IDs: {list(games_df['home_team_id'].unique()[:5])}")
        else:
            print(f"    Debug: Home team ID {home_team_id} has {len(all_home_games)} total games, but 0 in season {game_season} before {game_date}")
    
    # Get team stats up to this date (current season only)
    home_stats = calculate_team_stats_up_to_date(games_df, home_team_id, game_date, season=game_season)
    away_stats = calculate_team_stats_up_to_date(games_df, away_team_id, game_date, season=game_season)
    
    # Skip if we don't have enough data
    if not home_stats or not away_stats:
        # Provide more detailed error message
        home_count = len(home_games)
        away_count = len(away_games)
        if home_count < 5:
            print(f"  ⚠ Skipping {away_team_name} @ {home_team_name} (home team has only {home_count} historical games, need 5+)")
        elif away_count < 5:
            print(f"  ⚠ Skipping {away_team_name} @ {home_team_name} (away team has only {away_count} historical games, need 5+)")
        else:
            print(f"  ⚠ Skipping {away_team_name} @ {home_team_name} (home: {home_count} games, away: {away_count} games, but stats calculation failed)")
        return None
    
    # Get recent form
    home_recent = calculate_recent_form(games_df, home_team_id, game_date)
    away_recent = calculate_recent_form(games_df, away_team_id, game_date)
    
    # Get head-to-head statistics (cross-season, as matchup history persists)
    h2h_stats = calculate_head_to_head(games_df, home_team_id, away_team_id, game_date)
    
    # Get Natural Stat Trick advanced stats (current season only)
    home_nst_name = match_team_name_to_nst(home_team_name)
    away_nst_name = match_team_name_to_nst(away_team_name)
    home_nst = calculate_nst_stats_up_to_date(home_nst_name, game_date, season=game_season) if home_nst_name else None
    away_nst = calculate_nst_stats_up_to_date(away_nst_name, game_date, season=game_season) if away_nst_name else None
    
    # Create feature vector (same format as training)
    # NOTE: Reduced feature set - redundant features removed:
    # - home_games_played, away_games_played (just sample size)
    # - win_pct_diff, goal_differential_diff (derived)
    # - h2h_games, days_since_last_meeting, h2h_recent_home_win_pct (weak predictors)
    # - hdcf_pct, scf_pct (correlated with xgf_pct)
    # - xgf_pct_diff, cf_pct_diff (derived)
    features = {
        # Home team features (current season stats)
        'home_win_pct': home_stats['win_pct'],
        'home_goals_for_avg': home_stats['goals_for_avg'],
        'home_goals_against_avg': home_stats['goals_against_avg'],
        'home_goal_differential_avg': home_stats['goal_differential_avg'],
        'home_home_win_pct': home_stats['home_win_pct'],
        
        # Away team features (current season stats)
        'away_win_pct': away_stats['win_pct'],
        'away_goals_for_avg': away_stats['goals_for_avg'],
        'away_goals_against_avg': away_stats['goals_against_avg'],
        'away_goal_differential_avg': away_stats['goal_differential_avg'],
        'away_away_win_pct': away_stats['away_win_pct'],
        
        # Recent form (last 10 games)
        'home_recent_win_pct': home_recent['recent_win_pct'] if home_recent else 0.5,
        'home_recent_goal_differential': home_recent['recent_goal_differential_avg'] if home_recent else 0,
        'away_recent_win_pct': away_recent['recent_win_pct'] if away_recent else 0.5,
        'away_recent_goal_differential': away_recent['recent_goal_differential_avg'] if away_recent else 0,
        
        # Head-to-head features (cross-season)
        'h2h_home_win_pct': h2h_stats['h2h_home_team_win_pct'] if h2h_stats else 0.5,
        'h2h_goal_differential': h2h_stats['h2h_goal_differential'] if h2h_stats else 0,
        'last_meeting_home_won': h2h_stats['last_meeting_home_won'] if h2h_stats else 0.5,
        
        # Natural Stat Trick advanced stats (current season)
        # Use neutral defaults for missing values: 50% for percentages, 1.0 for PDO
        'home_xgf_pct_avg': home_nst['xgf_pct_avg'] if home_nst and home_nst.get('xgf_pct_avg') is not None else 50.0,
        'home_cf_pct_avg': home_nst['cf_pct_avg'] if home_nst and home_nst.get('cf_pct_avg') is not None else 50.0,
        'home_pdo_avg': home_nst['pdo_avg'] if home_nst and home_nst.get('pdo_avg') is not None else 100.0,
        'away_xgf_pct_avg': away_nst['xgf_pct_avg'] if away_nst and away_nst.get('xgf_pct_avg') is not None else 50.0,
        'away_cf_pct_avg': away_nst['cf_pct_avg'] if away_nst and away_nst.get('cf_pct_avg') is not None else 50.0,
        'away_pdo_avg': away_nst['pdo_avg'] if away_nst and away_nst.get('pdo_avg') is not None else 100.0,
    }
    
    # Create DataFrame with features in correct order
    features_df = pd.DataFrame([features])
    
    # If we have feature columns from the model, ensure correct order
    if feature_columns:
        # Add missing columns with default values
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        # Reorder to match training data
        features_df = features_df[feature_columns]
    
    return features_df


def load_historical_games():
    """Load historical games for feature calculation."""
    print("\nLoading historical game data...")
    
    # Use the same loading function from create_features.py
    from create_features import load_games
    
    games = load_games('data')
    
    if not games:
        raise ValueError("No historical games found. Please run src/fetch_historical_seasons.py first.")
    
    print(f"  Loaded {len(games)} historical games")
    
    # Convert to DataFrame using the same extraction function
    from create_features import extract_game_info
    
    games_list = []
    for game in games:
        game_info = extract_game_info(game)
        if game_info:
            games_list.append(game_info)
    
    games_df = pd.DataFrame(games_list)
    
    if len(games_df) == 0:
        raise ValueError("No valid games found in historical data.")
    
    # Parse dates
    games_df['date'] = pd.to_datetime(games_df['date']).dt.date
    
    return games_df


def predict_games(model, model_type, upcoming_games, games_df, feature_columns):
    """Make predictions for upcoming games."""
    print(f"\nCreating features and making predictions...")
    
    # Debug: Check what team IDs exist in historical data
    unique_team_ids = set(games_df['home_team_id'].unique()) | set(games_df['away_team_id'].unique())
    print(f"  Historical data contains {len(unique_team_ids)} unique team IDs")
    
    predictions = []
    
    for game in upcoming_games:
        # Parse game data
        game_data = parse_game_data(game)
        if not game_data:
            continue
        
        # Debug: Check if team IDs are in historical data
        home_id = game_data['home_team_id']
        away_id = game_data['away_team_id']
        if home_id not in unique_team_ids:
            print(f"  ⚠ Skipping {game_data['away_team_name']} @ {game_data['home_team_name']} (home team ID {home_id} not in historical data)")
            continue
        if away_id not in unique_team_ids:
            print(f"  ⚠ Skipping {game_data['away_team_name']} @ {game_data['home_team_name']} (away team ID {away_id} not in historical data)")
            continue
        
        # Create features
        features_df = create_features_for_game(games_df, game_data, feature_columns)
        
        if features_df is None:
            # Error message already printed in create_features_for_game
            continue
        
        # Make prediction
        if model_type == 'pipeline':
            # Pipeline model (StandardScaler + LogisticRegression)
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
        else:
            # Random Forest or other models
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
        
        home_win_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        away_win_prob = probabilities[0] if len(probabilities) > 1 else (1 - probabilities[0])
        
        # Convert to odds (American format)
        if home_win_prob > 0:
            home_odds = (1 / home_win_prob - 1) * 100
            if home_odds < 0:
                home_odds = abs(home_odds)
        else:
            home_odds = float('inf')
        
        if away_win_prob > 0:
            away_odds = (1 / away_win_prob - 1) * 100
            if away_odds < 0:
                away_odds = abs(away_odds)
        else:
            away_odds = float('inf')
        
        predictions.append({
            'date': game_data['date'],
            'away_team': game_data['away_team_name'],
            'home_team': game_data['home_team_name'],
            'prediction': 'Home Win' if prediction == 1 else 'Away Win',
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'confidence': max(home_win_prob, away_win_prob)
        })
    
    return predictions


def display_predictions(predictions):
    """Display predictions in a readable format."""
    print("\n" + "=" * 80)
    print("UPCOMING GAME PREDICTIONS")
    print("=" * 80)
    
    if not predictions:
        print("\nNo predictions available (insufficient historical data for upcoming games)")
        return
    
    # Sort by date
    predictions.sort(key=lambda x: x['date'])
    
    current_date = None
    for pred in predictions:
        # Print date header if new date
        if pred['date'] != current_date:
            current_date = pred['date']
            print(f"\n📅 {current_date.strftime('%A, %B %d, %Y')}")
            print("-" * 80)
        
        # Print prediction
        print(f"\n  {pred['away_team']} @ {pred['home_team']}")
        print(f"  Prediction: {pred['prediction']} ({pred['confidence']:.1%} confidence)")
        print(f"  Probabilities:")
        print(f"    {pred['home_team']}: {pred['home_win_prob']:.1%} (odds: {pred['home_odds']:.0f})")
        print(f"    {pred['away_team']}: {pred['away_win_prob']:.1%} (odds: {pred['away_odds']:.0f})")
    
    print("\n" + "=" * 80)


def main():
    """Main function to predict upcoming games."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict upcoming NHL games')
    parser.add_argument(
        '--model',
        type=str,
        default='models/random_forest_optimized.pkl',
        help='Path to model file (default: models/random_forest_optimized.pkl)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days ahead to fetch games (default: 7)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NHL Game Prediction - Upcoming Games")
    print("=" * 80)
    
    # Load model
    model, feature_columns = load_model(args.model)
    
    # If feature_columns not in model, get them from training data
    if feature_columns is None:
        print("\nFeature columns not found in model. Loading from training data...")
        try:
            df = pd.read_csv('data/game_features.csv')
            exclude_cols = [
                'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
                'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score'
            ]
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            print(f"  Found {len(feature_columns)} feature columns from training data")
        except Exception as e:
            print(f"  Warning: Could not load feature columns: {e}")
            feature_columns = None
    
    # Determine model type
    model_type = 'pipeline' if hasattr(model, 'steps') else 'standard'
    
    # Load historical games
    games_df = load_historical_games()
    
    # Initialize NHL client
    print("\nInitializing NHL API client...")
    client = NHLClient()
    
    # Fetch upcoming games
    upcoming_games = fetch_upcoming_games(client, days_ahead=args.days)
    
    if not upcoming_games:
        print("\nNo upcoming games found in the specified time period.")
        return
    
    # Make predictions
    predictions = predict_games(model, model_type, upcoming_games, games_df, feature_columns)
    
    # Display results
    display_predictions(predictions)
    
    print("\n✓ Predictions complete!")


if __name__ == '__main__':
    main()

