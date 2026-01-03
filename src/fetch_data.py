"""
Fetch data from NHL API and save to data folder.

This script fetches various NHL data and saves it locally.
"""

import os
import json
from datetime import datetime
from nhlpy import NHLClient


def ensure_data_folder():
    """Create data folder if it doesn't exist."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir}/ folder")
    return data_dir


def fetch_teams(client, data_dir):
    """Fetch all NHL teams and save to JSON."""
    print("Fetching teams...")
    try:
        teams = client.teams.teams()
        
        # Save as JSON
        output_file = os.path.join(data_dir, "teams.json")
        with open(output_file, 'w') as f:
            json.dump(teams, f, indent=2)
        
        print(f"✓ Saved {len(teams)} teams to {output_file}")
        return teams
    except Exception as e:
        print(f"✗ Error fetching teams: {e}")
        return None


def fetch_standings(client, data_dir):
    """Fetch current standings and save to JSON."""
    print("Fetching standings...")
    try:
        standings = client.standings.league_standings()
        
        # Save as JSON
        output_file = os.path.join(data_dir, "standings.json")
        with open(output_file, 'w') as f:
            json.dump(standings, f, indent=2)
        
        print(f"✓ Saved standings to {output_file}")
        return standings
    except Exception as e:
        print(f"✗ Error fetching standings: {e}")
        return None


def fetch_schedule(client, data_dir, days=7):
    """Fetch recent game schedule and save to JSON."""
    print(f"Fetching schedule for last {days} days...")
    try:
        # Get today's date
        today = datetime.now()
        
        # Fetch today's schedule
        schedule = client.schedule.daily_schedule()
        
        # Save as JSON
        output_file = os.path.join(data_dir, f"schedule_{today.strftime('%Y%m%d')}.json")
        with open(output_file, 'w') as f:
            json.dump(schedule, f, indent=2)
        
        num_games = len(schedule) if isinstance(schedule, list) else 0
        print(f"✓ Saved schedule to {output_file}")
        return schedule
    except Exception as e:
        print(f"✗ Error fetching schedule: {e}")
        return None


def main():
    """Main function to fetch and save NHL data."""
    print("=" * 60)
    print("NHL Data Fetcher")
    print("=" * 60)
    print()
    
    # Create data folder
    data_dir = ensure_data_folder()
    
    # Initialize client
    print("Initializing NHL client...")
    client = NHLClient()
    print()
    
    # Fetch data
    teams = fetch_teams(client, data_dir)
    print()
    
    standings = fetch_standings(client, data_dir)
    print()
    
    schedule = fetch_schedule(client, data_dir)
    print()
    
    print("=" * 60)
    print("Data fetch complete!")
    print(f"Data saved to: {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

