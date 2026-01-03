"""
Fetch historical game data for the last five seasons.

This script fetches all games from the past five NHL seasons using
date-based fetching (day-by-day).
"""

import os
import json
from datetime import datetime, timedelta
from nhlpy import NHLClient


def ensure_data_folder():
    """Create data folder if it doesn't exist."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def fetch_games_for_season(client, season_start, data_dir, season_name):
    """
    Fetch all games for a season using date-based fetching.
    
    Args:
        client: NHLClient instance
        season_start: Season start in format YYYYYYYY (e.g., 20232024 for 2023-2024)
        data_dir: Directory to save data
        season_name: Name for the season (e.g., "2023_2024")
    
    Returns:
        List of all games for the season
    """
    print(f"\nFetching games for {season_name} season...")
    
    # Extract year from season_start (first 4 digits)
    season_year = int(str(season_start)[:4])
    start_date = datetime(season_year, 10, 1)
    end_date = datetime(season_year + 1, 6, 30)
    
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_games = []
    current_date = start_date
    games_fetched = 0
    errors = 0
    
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
            
            if len(games) > 0:
                print(f"  {date_str}: {len(games)} games")
            elif games_fetched > 0 and games_fetched % 50 == 0:
                print(f"  Progress: {games_fetched} games fetched...")
            
        except Exception as e:
            errors += 1
            if errors < 5:  # Only print first few errors
                print(f"  Warning: Error fetching {date_str}: {e}")
        
        current_date += timedelta(days=1)
        
        # Small delay to avoid rate limiting
        import time
        time.sleep(0.1)
    
    # Save all games for this season
    output_file = os.path.join(data_dir, f"games_{season_name}.json")
    with open(output_file, 'w') as f:
        json.dump(all_games, f, indent=2)
    
    print(f"\n✓ Saved {len(all_games)} games to {output_file}")
    print(f"  Total games fetched: {games_fetched}")
    if errors > 0:
        print(f"  Errors encountered: {errors}")
    
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
    
    # Get current year to determine last five seasons
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Determine seasons to fetch
    # If we're past October, current season started this year
    # Otherwise, current season started last year
    if current_month >= 10:
        current_season_year = current_year  # Current season
        last_completed_year = current_year - 1  # Last completed season
    else:
        current_season_year = current_year - 1  # Last season (ended)
        last_completed_year = current_year - 2  # Season before that
    
    # Create list of 5 seasons (current + 4 previous)
    seasons = []
    for i in range(5):
        season_year = current_season_year - i
        season_start = int(f"{season_year}{season_year+1}")
        season_name = f"{season_year}_{season_year+1}"
        seasons.append((season_start, season_name))
    
    print(f"\nWill fetch data for:")
    for start, name in seasons:
        season_year = int(str(start)[:4])
        start_date = datetime(season_year, 10, 1)
        end_date = datetime(season_year + 1, 6, 30)
        print(f"  - {name}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_season_games = []
    
    # Fetch each season using date-based method
    for season_start, season_name in seasons:
        games = fetch_games_for_season(client, season_start, data_dir, season_name)
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

