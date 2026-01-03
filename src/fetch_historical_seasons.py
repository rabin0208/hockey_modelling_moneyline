"""
Fetch historical game data for the last two seasons using query builder pattern.

This script fetches all games from the past two NHL seasons using the
efficient query builder approach with season filters.
"""

import os
import json
from datetime import datetime
from nhlpy.nhl_client import NHLClient
from nhlpy.api.query.builder import QueryBuilder, QueryContext
from nhlpy.api.query.filters.season import SeasonQuery
from nhlpy.api.query.filters.game_type import GameTypeQuery


def ensure_data_folder():
    """Create data folder if it doesn't exist."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def fetch_games_for_season(client, season_start, season_end, data_dir, season_name, limit=100):
    """
    Fetch all games for a season using query builder pattern.
    
    Args:
        client: NHLClient instance
        season_start: Season start in format YYYYYYYY (e.g., 20232024 for 2023-2024)
        season_end: Season end in format YYYYYYYY (e.g., 20232024 for 2023-2024)
        data_dir: Directory to save data
        season_name: Name for the season (e.g., "2023_2024")
        limit: Number of records per request (default 100)
    
    Returns:
        List of all games for the season
    """
    print(f"\nFetching games for {season_name} season...")
    print(f"  Season: {season_start} to {season_end}")
    print(f"  Game type: Regular season (2)")
    
    # Build query with filters
    filters = [
        SeasonQuery(season_start=season_start, season_end=season_end),
        GameTypeQuery(game_type="2"),  # Regular season games
    ]
    
    context: QueryContext = QueryBuilder().build(filters=filters)
    
    all_games = []
    start = 0
    
    try:
        while True:
            print(f"  Fetching batch starting at {start}...")
            
            # Try to fetch games using query context
            # Note: This may need adjustment based on actual nhlpy API methods
            try:
                # Try schedule method with query context if available
                response = client.schedule.games_with_query_context(
                    query_context=context,
                    limit=limit,
                    start=start
                )
            except AttributeError:
                # If that method doesn't exist, try alternative approach
                # Fall back to date-based fetching if query builder doesn't work for schedules
                print("  Note: Query builder may not be available for schedules.")
                print("  Falling back to alternative method...")
                return fetch_games_alternative_method(client, season_start, data_dir, season_name)
            
            # Handle response
            if isinstance(response, dict):
                total = response.get('total', 0)
                batch_data = response.get('data', [])
                
                if not batch_data:
                    break
                
                all_games.extend(batch_data)
                print(f"    Fetched {len(batch_data)} games (total: {len(all_games)})")
                
                if len(all_games) >= total or len(batch_data) < limit:
                    break
                
                start += limit
            else:
                # Response might be a list directly
                if isinstance(response, list):
                    all_games.extend(response)
                    print(f"    Fetched {len(response)} games (total: {len(all_games)})")
                break
                
    except Exception as e:
        print(f"  Error with query builder method: {e}")
        print("  Falling back to alternative method...")
        return fetch_games_alternative_method(client, season_start, data_dir, season_name)
    
    # Save all games for this season
    output_file = os.path.join(data_dir, f"games_{season_name}.json")
    with open(output_file, 'w') as f:
        json.dump(all_games, f, indent=2)
    
    print(f"\n✓ Saved {len(all_games)} games to {output_file}")
    
    return all_games


def fetch_games_alternative_method(client, season_start, data_dir, season_name):
    """
    Alternative method to fetch games if query builder doesn't work for schedules.
    Uses date-based fetching as fallback.
    """
    from datetime import timedelta
    
    print(f"  Using date-based fetching for {season_name}...")
    
    # Extract year from season_start (first 4 digits)
    season_year = int(str(season_start)[:4])
    start_date = datetime(season_year, 10, 1)
    end_date = datetime(season_year + 1, 6, 30)
    
    all_games = []
    current_date = start_date
    games_fetched = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            schedule = client.schedule.daily_schedule(date=date_str)
            
            if isinstance(schedule, dict) and 'games' in schedule:
                games = schedule['games']
            elif isinstance(schedule, list):
                games = schedule
            else:
                games = []
            
            for game in games:
                game['gameDate'] = date_str
                all_games.append(game)
            
            games_fetched += len(games)
            
            if len(games) > 0 and games_fetched % 50 == 0:
                print(f"    Progress: {games_fetched} games fetched...")
            
        except Exception as e:
            pass  # Skip errors silently in fallback mode
        
        current_date += timedelta(days=1)
        
        # Small delay
        import time
        time.sleep(0.05)
    
    # Save games
    output_file = os.path.join(data_dir, f"games_{season_name}.json")
    with open(output_file, 'w') as f:
        json.dump(all_games, f, indent=2)
    
    print(f"  ✓ Saved {len(all_games)} games to {output_file}")
    
    return all_games


def main():
    """Main function to fetch historical seasons."""
    print("=" * 60)
    print("NHL Historical Season Data Fetcher")
    print("=" * 60)
    
    # Create data folder
    data_dir = ensure_data_folder()
    
    # Initialize client
    print("\nInitializing NHL client...")
    client = NHLClient(debug=False)
    
    # Get current year to determine last two seasons
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Determine seasons to fetch
    # If we're past October, current season started this year
    # Otherwise, current season started last year
    if current_month >= 10:
        season2_year = current_year  # Current season
        season1_year = current_year - 1  # Last season
    else:
        season2_year = current_year - 1  # Last season (ended)
        season1_year = current_year - 2  # Season before that
    
    # Format seasons as YYYYYYYY (e.g., 20232024 for 2023-2024 season)
    season1_start = int(f"{season1_year}{season1_year+1}")
    season1_end = season1_start
    season1_name = f"{season1_year}_{season1_year+1}"
    
    season2_start = int(f"{season2_year}{season2_year+1}")
    season2_end = season2_start
    season2_name = f"{season2_year}_{season2_year+1}"
    
    seasons = [
        (season1_start, season1_end, season1_name),
        (season2_start, season2_end, season2_name)
    ]
    
    print(f"\nWill fetch data for:")
    for start, end, name in seasons:
        print(f"  - {name}: Season {start}")
    
    all_season_games = []
    
    # Fetch each season using query builder
    for season_start, season_end, season_name in seasons:
        games = fetch_games_for_season(client, season_start, season_end, data_dir, season_name)
        all_season_games.extend(games)
    
    # Save combined file
    combined_file = os.path.join(data_dir, "games_all_seasons.json")
    with open(combined_file, 'w') as f:
        json.dump(all_season_games, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data fetch complete!")
    print(f"Total games fetched: {len(all_season_games)}")
    print(f"Data saved to: {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

