"""
Predict outcomes for upcoming NHL games.

Uses the copied hockey_moneyline feature builder and its frozen logistic
regression. Upcoming games are fetched from the NHL API.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from nhlpy import NHLClient

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_features import (
    FROZEN_FEATURE_COLUMNS,
    feature_frame,
    features_for_game,
    load_source,
    season_id,
)
from hockey_moneyline.errors import SchemaError
from hockey_moneyline.normalize import canonical_moneypuck_team_code

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hockey_moneyline",
    "artifacts",
    "frozen_prospective_fitted_artifact.json",
)


def load_model(model_path):
    """Load a saved model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if str(model_path).endswith(".json"):
        from hockey_moneyline.prospective_artifact import (
            assert_canonical_frozen_artifact,
            load_frozen_artifact_from_path,
        )

        print(f"Loading frozen moneyline model from {model_path}...")
        artifact = assert_canonical_frozen_artifact(load_frozen_artifact_from_path(model_path))
        model = artifact.pipeline()
        model.named_steps["scaler"].feature_names_in_ = np.array(
            artifact.feature_columns, dtype=object
        )
        print("✓ Frozen prospective model loaded")
        return model, list(artifact.feature_columns)
    
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

                game_state = str(game.get('gameState') or '').upper()
                if game_state in {'OFF', 'FINAL', 'LIVE', 'CRIT', 'OVER'}:
                    continue

                game_status = game.get('status') or {}
                abstract_state = game_status.get('abstractGameState') or game.get('abstractGameState', '')
                if game_state in {'FUT', 'PRE'} or abstract_state not in {'Final', 'Live', 'Official'}:
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
        game_id = game.get('id') or game.get('gamePk')
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
        
        home_abbrev = None
        away_abbrev = None
        if home_team:
            home_abbrev = home_team.get("abbrev") or home_team.get("triCode") or home_team.get("abbreviation")
        if away_team:
            away_abbrev = away_team.get("abbrev") or away_team.get("triCode") or away_team.get("abbreviation")
        if not home_abbrev or not away_abbrev:
            print(f"  Warning: Missing team abbreviations for game {game_id}")
            return None
        try:
            home_code = canonical_moneypuck_team_code(home_abbrev, season_id(game_date))
            away_code = canonical_moneypuck_team_code(away_abbrev, season_id(game_date))
        except SchemaError as exc:
            print(f"  Warning: Could not map teams for game {game_id}: {exc}")
            return None
        home_team_id = home_code
        away_team_id = away_code

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
            'season': season_id(game_date),
            'home_team_abbrev': home_code,
            'away_team_abbrev': away_code,
            'start_time': game.get('startTimeUTC') or '',
        }
    except Exception as e:
        print(f"  Warning: Could not parse game {game.get('gamePk', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_features_for_game(games_df, game, feature_columns):
    """Create the frozen 25 pregame features for one game."""
    columns = list(feature_columns or FROZEN_FEATURE_COLUMNS)
    game_id = game.get("game_id")
    if (
        game_id is not None
        and "game_id" in games_df.columns
        and "home_prior_games" in games_df.columns
    ):
        match = games_df[games_df["game_id"].astype(str) == str(game_id)]
        if len(match) == 1:
            row = match.iloc[0]
            return pd.DataFrame([{column: float(row[column]) for column in columns}])

    features = features_for_game(game)
    if features is None:
        print(
            f"  ⚠ Skipping {game.get('away_team_name', '?')} @ {game.get('home_team_name', '?')} "
            "(fewer than 5 prior games, or expected-goal history is missing)"
        )
        return None
    return pd.DataFrame([{column: float(features[column]) for column in columns}])


def load_historical_games():
    """Load MoneyPuck history and the frozen 25 pregame features."""
    print("\nLoading MoneyPuck history for the frozen feature set...")
    games_df = feature_frame(load_source())
    if games_df.empty:
        raise ValueError("No eligible games were built from the MoneyPuck history.")
    print(f"  Loaded {len(games_df)} games with frozen features")
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


def save_predictions_csv(predictions, output_path):
    """Save predictions to a CSV file."""
    if not predictions:
        print(f"\nNo predictions to save to {output_path}")
        return

    rows = []
    for pred in sorted(predictions, key=lambda x: x['date']):
        predicted_winner = pred['home_team'] if pred['prediction'] == 'Home Win' else pred['away_team']
        rows.append({
            'date': pred['date'].isoformat() if hasattr(pred['date'], 'isoformat') else pred['date'],
            'away_team': pred['away_team'],
            'home_team': pred['home_team'],
            'prediction': pred['prediction'],
            'predicted_winner': predicted_winner,
            'confidence': round(pred['confidence'], 4),
            'home_win_prob': round(pred['home_win_prob'], 4),
            'away_win_prob': round(pred['away_win_prob'], 4),
            'home_odds': round(pred['home_odds'], 1) if pred['home_odds'] != float('inf') else '',
            'away_odds': round(pred['away_odds'], 1) if pred['away_odds'] != float('inf') else '',
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(rows)} predictions to {output_path}")


def main():
    """Main function to predict upcoming games."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict upcoming NHL games')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help='Path to model file (default: frozen hockey_moneyline artifact)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days ahead to fetch games (default: 7)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save predictions as CSV'
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

    if args.output:
        save_predictions_csv(predictions, args.output)
    
    print("\n✓ Predictions complete!")


if __name__ == '__main__':
    main()

