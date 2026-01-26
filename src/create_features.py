"""
Create features for machine learning model from historical game data.

This script processes historical games and creates features for each game
based on team performance up to that point in the CURRENT season only.
"""

import os
import json
import pandas as pd
from datetime import datetime, date
from collections import defaultdict


def get_season_from_date(game_date):
    """
    Determine the NHL season from a game date.
    
    NHL seasons run from October to June. Games from Oct-Dec belong to the
    season that starts that year. Games from Jan-June belong to the season
    that started the previous year.
    
    Args:
        game_date: date object or string
        
    Returns:
        Season as integer like 20242025 or None
    """
    if game_date is None:
        return None
    
    # Convert to date object if needed
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()
    
    year = game_date.year
    month = game_date.month
    
    # Oct-Dec: season starts this year
    # Jan-Sep: season started previous year
    if month >= 10:
        start_year = year
    else:
        start_year = year - 1
    
    end_year = start_year + 1
    return int(f"{start_year}{end_year}")

# Try to load NST data (optional)
NST_DATA = None
try:
    nst_file = os.path.join('data', 'nst_processed.csv')
    if os.path.exists(nst_file):
        NST_DATA = pd.read_csv(nst_file)
        NST_DATA['date'] = pd.to_datetime(NST_DATA['date']).dt.date
        print(f"✓ Loaded Natural Stat Trick data: {len(NST_DATA)} records")
except Exception as e:
    print(f"Note: Natural Stat Trick data not available: {e}")

# Try to load Kaggle odds data (optional)
ODDS_DATA = None
ODDS_BY_TEAMS = None  # Lookup dictionary by (home_team, away_team) pairs
_ODDS_LOOKUP_BUILT = False  # Flag to track if lookup has been built

def normalize_team_name(name):
    """Normalize team name for matching (remove accents, periods, etc.)."""
    if name is None:
        return None
    import unicodedata
    # Remove accents (Montréal -> Montreal)
    name = unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode('ascii')
    # Remove periods (St. Louis -> St Louis)
    name = name.replace('.', '')
    # Lowercase and strip
    return name.lower().strip()

def _build_odds_lookup():
    """Build a lookup dictionary by team pairs for efficient matching."""
    global ODDS_BY_TEAMS, _ODDS_LOOKUP_BUILT
    
    if ODDS_DATA is None or len(ODDS_DATA) == 0:
        ODDS_BY_TEAMS = {}
        _ODDS_LOOKUP_BUILT = True
        return {}
    
    lookup = {}
    for idx, row in ODDS_DATA.iterrows():
        try:
            home_norm = normalize_team_name(row['home_team'])
            away_norm = normalize_team_name(row['away_team'])
            if home_norm is None or away_norm is None:
                continue
            key = (home_norm, away_norm)
            
            if key not in lookup:
                lookup[key] = []
            lookup[key].append({
                'date': row['date'],
                'home_moneyline': row.get('home_moneyline')
            })
        except Exception as e:
            # Skip rows with errors
            continue
    
    ODDS_BY_TEAMS = lookup
    _ODDS_LOOKUP_BUILT = True
    return lookup

try:
    odds_file = os.path.join('data', 'kaggle_odds.csv')
    if os.path.exists(odds_file):
        ODDS_DATA = pd.read_csv(odds_file)
        ODDS_DATA['date'] = pd.to_datetime(ODDS_DATA['date']).dt.date
        print(f"✓ Loaded Kaggle odds data: {len(ODDS_DATA)} records")
        print(f"  Columns: {list(ODDS_DATA.columns)}")
        _build_odds_lookup()
        if ODDS_BY_TEAMS:
            print(f"✓ Built odds lookup by team pairs: {len(ODDS_BY_TEAMS)} unique matchups")
            # Show a sample of keys
            sample_keys = list(ODDS_BY_TEAMS.keys())[:3]
            print(f"  Sample matchups: {sample_keys}")
        else:
            print(f"⚠️  Warning: Odds lookup is empty!")
    else:
        print(f"Note: Kaggle odds file not found: {odds_file}")
except Exception as e:
    print(f"Note: Kaggle odds data not available: {e}")
    import traceback
    traceback.print_exc()


def load_games(data_dir):
    """Load all games from JSON files."""
    games = []
    
    # Try to load combined file first
    combined_file = os.path.join(data_dir, "games_all_seasons.json")
    if os.path.exists(combined_file):
        print(f"Loading games from {combined_file}...")
        with open(combined_file, 'r') as f:
            games = json.load(f)
        print(f"  Loaded {len(games)} games")
        return games
    
    # Otherwise load individual season files
    season_files = [
        "games_2023_2024.json",
        "games_2024_2025.json",
        "games_2022_2023.json",
        "games_2021_2022.json"
    ]
    
    for filename in season_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading games from {filename}...")
            with open(filepath, 'r') as f:
                season_games = json.load(f)
                games.extend(season_games)
                print(f"  Loaded {len(season_games)} games")
    
    return games


def extract_game_info(game):
    """Extract basic information from a game."""
    try:
        # Get game date
        game_date = game.get('gameDate') or game.get('startTimeUTC', '')[:10]
        
        # Get teams
        away_team = game.get('awayTeam', {})
        home_team = game.get('homeTeam', {})
        
        away_team_id = away_team.get('id')
        home_team_id = home_team.get('id')
        
        away_team_name = away_team.get('placeName', {}).get('default', '') + ' ' + away_team.get('commonName', {}).get('default', '')
        home_team_name = home_team.get('placeName', {}).get('default', '') + ' ' + home_team.get('commonName', {}).get('default', '')
        
        # Get scores
        away_score = away_team.get('score')
        home_score = home_team.get('score')
        
        # Determine winner (1 if home wins, 0 if away wins)
        if away_score is not None and home_score is not None:
            home_wins = 1 if home_score > away_score else 0
        else:
            home_wins = None  # Game not played yet
        
        # Get game state
        game_state = game.get('gameState', '')
        
        return {
            'game_id': game.get('id'),
            'date': game_date,
            'season': game.get('season'),
            'away_team_id': away_team_id,
            'away_team_name': away_team_name.strip(),
            'home_team_id': home_team_id,
            'home_team_name': home_team_name.strip(),
            'away_score': away_score,
            'home_score': home_score,
            'home_wins': home_wins,
            'game_state': game_state
        }
    except Exception as e:
        print(f"Error extracting game info: {e}")
        return None


def get_previous_season(season):
    """Get the previous season (e.g., 20242025 -> 20232024)."""
    if season is None:
        return None
    
    # Convert to string if integer
    season_str = str(season)
    
    if len(season_str) != 8:
        return None
    
    try:
        start_year = int(season_str[:4])
        prev_season = f"{start_year - 1}{start_year}"
        # Return same type as input
        return int(prev_season) if isinstance(season, int) else prev_season
    except ValueError:
        return None


def calculate_team_stats_up_to_date(games_df, team_id, game_date, season=None, min_games=5):
    """
    Calculate team statistics up to (but not including) a given date.
    Prefers current season data, but falls back to previous season for early-season games.
    
    Args:
        games_df: DataFrame with all games
        team_id: Team ID to calculate stats for
        game_date: Date to calculate stats up to (exclusive)
        season: Season string like "20242025" (if None, derived from game_date)
        min_games: Minimum games required to return stats
    
    Returns:
        Dictionary with team statistics
    """
    # Determine the season if not provided
    if season is None:
        season = get_season_from_date(game_date)
    
    # Get all games for this team before the given date IN THE SAME SEASON
    team_games = games_df[
        (games_df['date'] < game_date) &
        ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
        (games_df['home_wins'].notna()) &  # Only completed games
        (games_df['season'] == season)  # Same season only
    ].copy()
    
    # If not enough current season games, fall back to previous season
    if len(team_games) < min_games:
        prev_season = get_previous_season(season)
        if prev_season:
            prev_season_games = games_df[
                ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
                (games_df['home_wins'].notna()) &
                (games_df['season'] == prev_season)
            ].copy()
            
            if len(prev_season_games) >= min_games:
                # Use previous season data as fallback
                team_games = prev_season_games
    
    if len(team_games) < min_games:
        return None
    
    # Calculate statistics
    wins = 0
    losses = 0
    ot_losses = 0
    goals_for = 0
    goals_against = 0
    home_wins = 0
    home_games = 0
    away_wins = 0
    away_games = 0
    
    for _, game in team_games.iterrows():
        is_home = game['home_team_id'] == team_id
        
        if is_home:
            team_score = game['home_score']
            opp_score = game['away_score']
            home_games += 1
            if game['home_wins'] == 1:
                wins += 1
                home_wins += 1
            else:
                losses += 1
        else:
            team_score = game['away_score']
            opp_score = game['home_score']
            away_games += 1
            if game['home_wins'] == 0:
                wins += 1
                away_wins += 1
            else:
                losses += 1
        
        goals_for += team_score if pd.notna(team_score) else 0
        goals_against += opp_score if pd.notna(opp_score) else 0
    
    total_games = len(team_games)
    
    if total_games == 0:
        return None
    
    stats = {
        'games_played': total_games,
        'wins': wins,
        'losses': losses,
        'win_pct': wins / total_games if total_games > 0 else 0,
        'goals_for_avg': goals_for / total_games if total_games > 0 else 0,
        'goals_against_avg': goals_against / total_games if total_games > 0 else 0,
        'goal_differential_avg': (goals_for - goals_against) / total_games if total_games > 0 else 0,
        'home_win_pct': home_wins / home_games if home_games > 0 else 0.5,
        'away_win_pct': away_wins / away_games if away_games > 0 else 0.5,
        'home_games': home_games,
        'away_games': away_games
    }
    
    return stats


def calculate_nst_stats_up_to_date(team_name, game_date, season=None, min_games=5):
    """
    Calculate Natural Stat Trick advanced stats for a team up to a given date.
    Prefers current season data, but falls back to previous season for early-season games.
    
    Args:
        team_name: Team name (must match NST data)
        game_date: Date to calculate stats up to (exclusive)
        season: Season string like "20242025" (if None, derived from game_date)
        min_games: Minimum games required
    
    Returns:
        Dictionary with NST statistics or None
    """
    if NST_DATA is None or len(NST_DATA) == 0:
        return None
    
    # Ensure date is a date object for comparison
    if isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()
    elif isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()
    
    # Determine the season if not provided
    if season is None:
        season = get_season_from_date(game_date)
    
    # Filter to this team's games before the date IN THE SAME SEASON
    # NST data doesn't have a season column, so we derive it from the date
    team_games = NST_DATA[
        (NST_DATA['team_name'] == team_name) &
        (NST_DATA['date'] < game_date) &
        (NST_DATA['date'].apply(get_season_from_date) == season)
    ].copy()
    
    # If not enough current season games, fall back to previous season
    if len(team_games) < min_games:
        prev_season = get_previous_season(season)
        if prev_season:
            prev_season_games = NST_DATA[
                (NST_DATA['team_name'] == team_name) &
                (NST_DATA['date'].apply(get_season_from_date) == prev_season)
            ].copy()
            
            if len(prev_season_games) >= min_games:
                # Use previous season data as fallback
                team_games = prev_season_games
    
    if len(team_games) < min_games:
        return None
    
    # Calculate averages (handle None values)
    def safe_mean(series):
        valid = series.dropna()
        return valid.mean() if len(valid) > 0 else None
    
    stats = {
        'xgf_pct_avg': safe_mean(team_games['xGF%']) if 'xGF%' in team_games.columns else None,
        'cf_pct_avg': safe_mean(team_games['CF%']) if 'CF%' in team_games.columns else None,
        'ff_pct_avg': safe_mean(team_games['FF%']) if 'FF%' in team_games.columns else None,
        'sf_pct_avg': safe_mean(team_games['SF%']) if 'SF%' in team_games.columns else None,
        'hdcf_pct_avg': safe_mean(team_games['HDCF%']) if 'HDCF%' in team_games.columns else None,
        'scf_pct_avg': safe_mean(team_games['SCF%']) if 'SCF%' in team_games.columns else None,
        'pdo_avg': safe_mean(team_games['PDO']) if 'PDO' in team_games.columns else None,
        'xgf_avg': safe_mean(team_games['xGF']) if 'xGF' in team_games.columns else None,
        'xga_avg': safe_mean(team_games['xGA']) if 'xGA' in team_games.columns else None,
    }
    
    return stats




def match_team_name_to_nst(our_team_name):
    """
    Match our team name format to NST team name format.
    
    Args:
        our_team_name: Team name from our data
    
    Returns:
        NST team name or None
    """
    if NST_DATA is None:
        return None
    
    our_normalized = normalize_team_name(our_team_name)
    nst_teams = NST_DATA['team_name'].unique()
    
    # Try exact match (normalized)
    for nst_team in nst_teams:
        if normalize_team_name(nst_team) == our_normalized:
            return nst_team
    
    # Try partial match (normalized)
    for nst_team in nst_teams:
        nst_normalized = normalize_team_name(nst_team)
        # Check if key words match
        our_words = set(our_normalized.split())
        nst_words = set(nst_normalized.split())
        if our_words & nst_words:  # If there's any overlap
            return nst_team
    
    # Special cases for teams that don't exist in NST data
    # (e.g., Utah teams that relocated after NST data ends)
    if 'utah' in our_normalized:
        return None
    
    return None


def get_odds_for_game(game_id, game_date, home_team_name, away_team_name):
    """
    Get odds for a specific game from Kaggle data.
    
    Uses team-first matching: matches by (home_team, away_team) pair first,
    then checks dates with ±1 day flexibility to handle date offset issues.
    
    Args:
        game_id: Game ID (not used, kept for compatibility)
        game_date: Game date (can be string, date, or datetime)
        home_team_name: Home team name
        away_team_name: Away team name
    
    Returns:
        Dictionary with odds or None
    
    Note: The Kaggle data has:
    - home_moneyline: Odds for the home team (not the favorite, just the home team)
    - Some dates may be off by ±1 day, so we check both the exact date and date±1
    """
    if ODDS_BY_TEAMS is None or len(ODDS_BY_TEAMS) == 0:
        return None
    
    # Convert date to date object
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()
    
    # Normalize team names for matching
    home_normalized = normalize_team_name(home_team_name)
    away_normalized = normalize_team_name(away_team_name)
    
    # Look up by team pair (this is the primary key)
    team_key = (home_normalized, away_normalized)
    
    if team_key not in ODDS_BY_TEAMS:
        return None
    
    # Get all games between these teams
    candidate_games = ODDS_BY_TEAMS[team_key]
    
    # Find the best date match (check exact date, date-1, and date+1)
    from datetime import timedelta
    
    best_match = None
    best_date_diff = None
    
    for game in candidate_games:
        odds_date = game['date']
        
        # Calculate date difference
        date_diff = abs((game_date - odds_date).days)
        
        # Accept matches within ±1 day
        if date_diff <= 1:
            # Prefer exact date match, then date-1, then date+1
            if best_match is None or date_diff < best_date_diff:
                best_match = game
                best_date_diff = date_diff
    
    if best_match is None:
        return None
    
    # Return only home_moneyline (most informative for win/loss prediction)
    return {
        'home_moneyline': best_match.get('home_moneyline')  # Odds for home team
    }


def calculate_head_to_head(games_df, home_team_id, away_team_id, date, lookback_games=10):
    """
    Calculate head-to-head statistics between two teams.
    
    Args:
        games_df: DataFrame with all games
        home_team_id: Home team ID
        away_team_id: Away team ID
        date: Date to calculate stats up to
        lookback_games: Number of recent H2H games to consider
    
    Returns:
        Dictionary with head-to-head statistics
    """
    # Get all games between these two teams before the given date
    h2h_games = games_df[
        (games_df['date'] < date) &
        (games_df['home_wins'].notna()) &  # Only completed games
        (
            ((games_df['home_team_id'] == home_team_id) & (games_df['away_team_id'] == away_team_id)) |
            ((games_df['home_team_id'] == away_team_id) & (games_df['away_team_id'] == home_team_id))
        )
    ].sort_values('date', ascending=False)
    
    if len(h2h_games) == 0:
        return None
    
    # Calculate overall H2H stats
    home_team_wins = 0
    away_team_wins = 0
    home_team_goals = 0
    away_team_goals = 0
    
    for _, game in h2h_games.iterrows():
        is_home = game['home_team_id'] == home_team_id
        
        if is_home:
            home_team_score = game['home_score']
            away_team_score = game['away_score']
            if game['home_wins'] == 1:
                home_team_wins += 1
            else:
                away_team_wins += 1
        else:
            home_team_score = game['away_score']  # home_team_id was away in this game
            away_team_score = game['home_score']
            if game['home_wins'] == 0:
                home_team_wins += 1
            else:
                away_team_wins += 1
        
        home_team_goals += home_team_score if pd.notna(home_team_score) else 0
        away_team_goals += away_team_score if pd.notna(away_team_score) else 0
    
    total_h2h_games = len(h2h_games)
    
    # Get recent H2H games (last N games)
    recent_h2h = h2h_games.head(lookback_games)
    recent_home_wins = 0
    recent_away_wins = 0
    
    for _, game in recent_h2h.iterrows():
        is_home = game['home_team_id'] == home_team_id
        if is_home:
            if game['home_wins'] == 1:
                recent_home_wins += 1
            else:
                recent_away_wins += 1
        else:
            if game['home_wins'] == 0:
                recent_home_wins += 1
            else:
                recent_away_wins += 1
    
    # Get last meeting info
    last_game = h2h_games.iloc[0] if len(h2h_games) > 0 else None
    last_meeting_home_won = None
    days_since_last_meeting = None
    
    if last_game is not None:
        is_home = last_game['home_team_id'] == home_team_id
        last_meeting_home_won = 1 if (is_home and last_game['home_wins'] == 1) or (not is_home and last_game['home_wins'] == 0) else 0
        
        # Calculate days since last meeting
        last_date = pd.to_datetime(last_game['date'])
        current_date = pd.to_datetime(date)
        days_since_last_meeting = (current_date - last_date).days
    
    return {
        'h2h_games': total_h2h_games,
        'h2h_home_team_win_pct': home_team_wins / total_h2h_games if total_h2h_games > 0 else 0.5,
        'h2h_home_team_goals_avg': home_team_goals / total_h2h_games if total_h2h_games > 0 else 0,
        'h2h_away_team_goals_avg': away_team_goals / total_h2h_games if total_h2h_games > 0 else 0,
        'h2h_goal_differential': (home_team_goals - away_team_goals) / total_h2h_games if total_h2h_games > 0 else 0,
        'h2h_recent_games': len(recent_h2h),
        'h2h_recent_home_win_pct': recent_home_wins / len(recent_h2h) if len(recent_h2h) > 0 else 0.5,
        'last_meeting_home_won': last_meeting_home_won if last_meeting_home_won is not None else 0.5,
        'days_since_last_meeting': days_since_last_meeting if days_since_last_meeting is not None else 999
    }


def calculate_recent_form(games_df, team_id, date, lookback_games=10):
    """
    Calculate recent form (last N games) for a team.
    
    Args:
        games_df: DataFrame with all games
        team_id: Team ID
        date: Date to calculate form up to
        lookback_games: Number of recent games to consider
    
    Returns:
        Dictionary with recent form statistics
    """
    # Get recent games
    team_games = games_df[
        (games_df['date'] < date) &
        ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
        (games_df['home_wins'].notna())
    ].sort_values('date', ascending=False).head(lookback_games)
    
    if len(team_games) == 0:
        return None
    
    wins = 0
    goals_for = 0
    goals_against = 0
    
    for _, game in team_games.iterrows():
        is_home = game['home_team_id'] == team_id
        
        if is_home:
            if game['home_wins'] == 1:
                wins += 1
            team_score = game['home_score']
            opp_score = game['away_score']
        else:
            if game['home_wins'] == 0:
                wins += 1
            team_score = game['away_score']
            opp_score = game['home_score']
        
        goals_for += team_score if pd.notna(team_score) else 0
        goals_against += opp_score if pd.notna(opp_score) else 0
    
    return {
        'recent_games': len(team_games),
        'recent_wins': wins,
        'recent_win_pct': wins / len(team_games) if len(team_games) > 0 else 0,
        'recent_goals_for_avg': goals_for / len(team_games) if len(team_games) > 0 else 0,
        'recent_goals_against_avg': goals_against / len(team_games) if len(team_games) > 0 else 0,
        'recent_goal_differential_avg': (goals_for - goals_against) / len(team_games) if len(team_games) > 0 else 0
    }


def create_features(games_df):
    """
    Create features for each game based on team performance up to that date.
    
    Args:
        games_df: DataFrame with all games
    
    Returns:
        DataFrame with features for each game
    """
    print("\nCreating features for each game...")
    
    features_list = []
    total_games = len(games_df)
    
    for idx, game in games_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing game {idx+1}/{total_games}...")
        
        # Skip games without scores (future games)
        if pd.isna(game['home_wins']):
            continue
        
        game_date = game['date']
        game_season = game['season']
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        
        # Get team stats up to this date (current season only)
        home_stats = calculate_team_stats_up_to_date(games_df, home_team_id, game_date, season=game_season)
        away_stats = calculate_team_stats_up_to_date(games_df, away_team_id, game_date, season=game_season)
        
        # Get recent form (still uses all recent games, not season-filtered)
        home_recent = calculate_recent_form(games_df, home_team_id, game_date)
        away_recent = calculate_recent_form(games_df, away_team_id, game_date)
        
        # Get head-to-head statistics (cross-season, as matchup history persists)
        h2h_stats = calculate_head_to_head(games_df, home_team_id, away_team_id, game_date)
        
        # Get Natural Stat Trick advanced stats (current season only)
        home_nst_name = match_team_name_to_nst(game['home_team_name'])
        away_nst_name = match_team_name_to_nst(game['away_team_name'])
        home_nst = calculate_nst_stats_up_to_date(home_nst_name, game_date, season=game_season) if home_nst_name else None
        away_nst = calculate_nst_stats_up_to_date(away_nst_name, game_date, season=game_season) if away_nst_name else None
        
        # Get odds data
        odds = get_odds_for_game(game['game_id'], game_date, game['home_team_name'], game['away_team_name'])
        
        # Skip if we don't have enough data
        if not home_stats or not away_stats:
            continue
        
        # Create feature vector
        # NOTE: Redundant features removed:
        # - home_games_played, away_games_played (just sample size, not team quality)
        # - win_pct_diff, goal_differential_diff (derived from other features)
        # - h2h_games, days_since_last_meeting (weak predictors)
        # - hdcf_pct, scf_pct (correlated with xgf_pct)
        # - xgf_pct_diff, cf_pct_diff (derived from individual team stats)
        features = {
            'game_id': game['game_id'],
            'date': game_date,
            'season': game['season'],
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': game['home_team_name'],
            'away_team_name': game['away_team_name'],
            
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
            
            # Natural Stat Trick advanced stats (current season, if available)
            # Keeping xGF% (best predictor), CF% (possession), and PDO (luck indicator)
            'home_xgf_pct_avg': home_nst['xgf_pct_avg'] if home_nst and home_nst.get('xgf_pct_avg') is not None else None,
            'home_cf_pct_avg': home_nst['cf_pct_avg'] if home_nst and home_nst.get('cf_pct_avg') is not None else None,
            'home_pdo_avg': home_nst['pdo_avg'] if home_nst and home_nst.get('pdo_avg') is not None else None,
            'away_xgf_pct_avg': away_nst['xgf_pct_avg'] if away_nst and away_nst.get('xgf_pct_avg') is not None else None,
            'away_cf_pct_avg': away_nst['cf_pct_avg'] if away_nst and away_nst.get('cf_pct_avg') is not None else None,
            'away_pdo_avg': away_nst['pdo_avg'] if away_nst and away_nst.get('pdo_avg') is not None else None,
            
            # Target variable
            'home_wins': game['home_wins'],
            'home_score': game['home_score'],
            'away_score': game['away_score']
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"\n✓ Created features for {len(features_df)} games")
    
    return features_df


def main():
    """Main function to create features."""
    print("=" * 60)
    print("NHL Game Features Creator")
    print("=" * 60)
    
    data_dir = "data"
    
    # Load games
    games = load_games(data_dir)
    
    if len(games) == 0:
        print("\n✗ No games found. Please run fetch_historical_seasons.py first.")
        return
    
    # Convert to DataFrame
    print("\nProcessing games...")
    game_info = [extract_game_info(game) for game in games]
    game_info = [g for g in game_info if g is not None]
    
    games_df = pd.DataFrame(game_info)
    games_df['date'] = pd.to_datetime(games_df['date'])
    games_df = games_df.sort_values('date')
    
    print(f"  Total games: {len(games_df)}")
    print(f"  Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    print(f"  Completed games: {games_df['home_wins'].notna().sum()}")
    
    # Create features
    features_df = create_features(games_df)
    
    # Drop home_moneyline column
    if 'home_moneyline' in features_df.columns:
        features_df = features_df.drop(columns=['home_moneyline'])
        print(f"\n✓ Dropped home_moneyline column")
    
    # Save features
    output_file = os.path.join(data_dir, "game_features.csv")
    features_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved features to {output_file}")
    
    # Show summary
    print("\n" + "=" * 60)
    print("Feature Summary")
    print("=" * 60)
    print(f"Total games with features: {len(features_df)}")
    print(f"Home team win rate: {features_df['home_wins'].mean():.2%}")
    print("\nFeature columns:")
    feature_cols = [col for col in features_df.columns if col not in ['game_id', 'date', 'season', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name', 'home_wins', 'home_score', 'away_score']]
    for col in feature_cols:
        print(f"  - {col}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

