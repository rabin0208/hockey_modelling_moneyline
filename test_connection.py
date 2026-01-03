"""
Test connection to NHL API using nhl-api-py package.

This script tests if we can connect to the NHL API using the nhl-api-py package.
Reference: https://pypi.org/project/nhl-api-py/
"""

try:
    from nhlpy import NHLClient
    NHL_API_AVAILABLE = True
except ImportError:
    NHL_API_AVAILABLE = False
    print("⚠️  nhl-api-py package not installed.")
    print("   Install it with: pip install nhl-api-py")
    print("   Or: conda env update -f environment.yml")
    print()


def test_connection():
    """Test connection to NHL API using nhl-api-py package."""
    print("Testing connection to NHL API using nhl-api-py...")
    print("-" * 50)
    
    if not NHL_API_AVAILABLE:
        print("✗ FAILED: nhl-api-py package is not installed")
        return False
    
    try:
        print("Initializing NHL client...")
        client = NHLClient()
        
        print("Fetching teams...")
        teams = client.teams.teams()
        
        if teams:
            print("✓ SUCCESS! Connected to NHL API")
            print(f"  Found {len(teams)} teams")
            
            # Show first few teams
            print("\n  Sample teams:")
            for team in teams[:5]:
                # Handle different possible data structures
                if isinstance(team, dict):
                    team_name = team.get('name', team.get('teamName', 'Unknown'))
                    team_abbr = team.get('abbreviation', team.get('abbrev', 'N/A'))
                    team_id = team.get('id', 'N/A')
                else:
                    team_name = str(team)
                    team_abbr = 'N/A'
                    team_id = 'N/A'
                print(f"    - {team_name} ({team_abbr}) [ID: {team_id}]")
            
            return True
        else:
            print("✗ FAILED: No teams returned")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print("✗ FAILED: Cannot connect to NHL API")
        print(f"  Error: {error_msg}")
        
        # Check if it's a DNS/connection error
        if "resolve" in error_msg.lower() or "nodename" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n  This appears to be a DNS/network issue.")
            print("  Troubleshooting:")
            print("    1. Check your internet connection")
            print("    2. Try accessing in your browser:")
            print("       https://statsapi.web.nhl.com/api/v1/teams")
            print("    3. Change DNS servers to 8.8.8.8 (Google DNS)")
            print("    4. Check firewall/antivirus settings")
            print("    5. Try a different network (mobile hotspot)")
        
        return False


if __name__ == "__main__":
    test_connection()

