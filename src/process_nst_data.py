"""
Process Natural Stat Trick game data and prepare for integration.

This script:
1. Loads the NST games.csv file
2. Parses game dates and team names
3. Matches to our game data
4. Calculates rolling averages of advanced stats
"""

import os
import pandas as pd
from datetime import datetime
import re


def parse_game_string(game_str):
    """
    Parse game string like " 2021-10-12 - Penguins 6, Lightning 2"
    
    Returns:
        date, home_team, away_team (or None if can't parse)
    """
    try:
        # Remove leading/trailing spaces
        game_str = game_str.strip()
        
        # Extract date (format: YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', game_str)
        if not date_match:
            return None, None, None
        
        date_str = date_match.group(1)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Extract teams and score (format: "Team1 Score1, Team2 Score2")
        # Pattern: " - Team1 Score1, Team2 Score2"
        game_part = game_str.split(' - ', 1)
        if len(game_part) < 2:
            return None, None, None
        
        teams_part = game_part[1]
        # Match pattern like "Penguins 6, Lightning 2"
        match = re.match(r'(.+?)\s+(\d+),\s+(.+?)\s+(\d+)', teams_part)
        if match:
            team1 = match.group(1).strip()
            score1 = int(match.group(2))
            team2 = match.group(3).strip()
            score2 = int(match.group(4))
            
            # Determine home/away (usually second team is home, but not always)
            # For now, we'll match by team name regardless
            return date_obj, team1, team2
        
        return None, None, None
    except Exception as e:
        return None, None, None


def load_nst_data(csv_file):
    """
    Load Natural Stat Trick data from CSV.
    
    Returns:
        DataFrame with parsed game data
    """
    print(f"Loading Natural Stat Trick data from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")
    
    # Parse game strings
    print("  Parsing game dates and teams...")
    parsed_data = []
    for idx, row in df.iterrows():
        date, team1, team2 = parse_game_string(row['Game'])
        if date:
            # Determine which team this row is for
            current_team = row['Team']
            
            # Helper function to safely convert to float
            def safe_float(value):
                if pd.isna(value) or value == '-' or value == '':
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Add both teams' data
            parsed_data.append({
                'date': date,
                'team_name': current_team,
                'opponent': team2 if current_team == team1 else team1,
                'xGF': safe_float(row['xGF']),
                'xGA': safe_float(row['xGA']),
                'xGF%': safe_float(row['xGF%']),
                'CF%': safe_float(row['CF%']),
                'FF%': safe_float(row['FF%']),
                'SF%': safe_float(row['SF%']),
                'HDCF%': safe_float(row['HDCF%']),
                'SCF%': safe_float(row['SCF%']),
                'PDO': safe_float(row['PDO']),
            })
    
    nst_df = pd.DataFrame(parsed_data)
    print(f"  Parsed {len(nst_df)} team-game records")
    print(f"  Date range: {nst_df['date'].min()} to {nst_df['date'].max()}")
    print(f"  Unique teams: {nst_df['team_name'].nunique()}")
    
    return nst_df


def match_team_names(nst_team_name, our_team_names):
    """
    Match NST team name to our team names.
    
    Returns:
        Matched team name or None
    """
    nst_lower = nst_team_name.lower()
    
    # Try exact match first
    for our_name in our_team_names:
        if our_name.lower() == nst_lower:
            return our_name
    
    # Try partial match
    for our_name in our_team_names:
        if nst_lower in our_name.lower() or our_name.lower() in nst_lower:
            return our_name
    
    return None


def main():
    """Main function to process NST data."""
    print("=" * 60)
    print("Natural Stat Trick Data Processor")
    print("=" * 60)
    
    nst_file = os.path.join('data', 'games.csv')
    
    if not os.path.exists(nst_file):
        print(f"✗ File not found: {nst_file}")
        return
    
    # Load NST data
    nst_df = load_nst_data(nst_file)
    
    # Show sample
    print("\nSample data:")
    print(nst_df.head(10))
    
    # Show unique teams
    print(f"\nUnique teams in NST data ({nst_df['team_name'].nunique()}):")
    for team in sorted(nst_df['team_name'].unique()):
        game_count = len(nst_df[nst_df['team_name'] == team])
        print(f"  - {team}: {game_count} games")
    
    # Save processed data
    output_file = os.path.join('data', 'nst_processed.csv')
    nst_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved processed data to {output_file}")
    print(f"  Ready for integration into feature creation")


if __name__ == '__main__':
    main()

