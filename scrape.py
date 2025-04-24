import pandas as pd
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashteamstats
import time
import os

# Dictionary of NBA champions by season end year
NBA_CHAMPIONS = {
    2001: "Los Angeles Lakers",
    2002: "Los Angeles Lakers",
    2003: "San Antonio Spurs",
    2004: "Detroit Pistons",
    2005: "San Antonio Spurs",
    2006: "Miami Heat",
    2007: "San Antonio Spurs",
    2008: "Boston Celtics",
    2009: "Los Angeles Lakers",
    2010: "Los Angeles Lakers",
    2011: "Dallas Mavericks",
    2012: "Miami Heat",
    2013: "Miami Heat",
    2014: "San Antonio Spurs",
    2015: "Golden State Warriors",
    2016: "Cleveland Cavaliers",
    2017: "Golden State Warriors",
    2018: "Golden State Warriors",
    2019: "Toronto Raptors",
    2020: "Los Angeles Lakers",
    2021: "Milwaukee Bucks",
    2022: "Golden State Warriors",
    2023: "Denver Nuggets",
    2024: "Boston Celtics"
}

def fetch_season_metrics(season_end_year: int) -> pd.DataFrame:
    """
    Fetch regular-season advanced metrics from nba_api,
    and SRS and win percentage from Basketball-Reference, for a given season ending in season_end_year.
    """
    # Format season string for nba_api (e.g., "2000-01")
    season_str = f"{season_end_year-1}-{str(season_end_year)[-2:]}"

    # Initialize champion DataFrame early to avoid reference errors
    df_champion = pd.DataFrame(columns=['team', 'is_champion'])

    # 1. Regular-season advanced stats
    reg = leaguedashteamstats.LeagueDashTeamStats(
        season=season_str,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    
    # Print columns for debugging
    print(f"Available columns for {season_end_year}: {reg.columns.tolist()}")
    
    # Get available columns that we need - using the correct column names from the API
    available_columns = ['TEAM_ID', 'TEAM_NAME', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'EFG_PCT', 'TS_PCT', 'TM_TOV_PCT', 'OREB_PCT', 'DREB_PCT']
    
    # Select only columns that exist
    reg = reg[available_columns]
    
    # Rename columns to our standardized names
    column_mapping = {
        'TEAM_ID': 'team_id',
        'TEAM_NAME': 'team',
        'W_PCT': 'win_pct',
        'OFF_RATING': 'off_rtg',
        'DEF_RATING': 'def_rtg',
        'NET_RATING': 'net_rtg',
        'PACE': 'pace',
        'EFG_PCT': 'efg_pct',
        'TS_PCT': 'ts_pct',
        'TM_TOV_PCT': 'tov_pct',
        'OREB_PCT': 'orb_pct',
        'DREB_PCT': 'drb_pct'
    }
    reg = reg.rename(columns=column_mapping)
    
    # 2. Fetch SRS from Basketball-Reference
    br_url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}.html"
    
    try:
        response = requests.get(br_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the team stats table
        table = soup.find('table', {'id': 'misc_stats'})  # SRS is in the misc_stats table
        if not table:
            table = soup.find('table', {'id': 'advanced-team'})  # Try alternate table
        
        if table:
            # Parse the table into a DataFrame
            df_adv = pd.read_html(str(table))[0]
            
            # Handle multi-level columns if present
            if isinstance(df_adv.columns, pd.MultiIndex):
                df_adv.columns = [' '.join(col).strip() for col in df_adv.columns.values]
            
            # Clean up the team names (remove * and trailing spaces)
            df_adv['Team'] = df_adv['Team'].str.replace('*', '', regex=False).str.strip()
            
            # Extract only the columns we need
            if 'SRS' in df_adv.columns:
                df_adv = df_adv[['Team', 'SRS']]
                df_adv = df_adv.rename(columns={'Team': 'team', 'SRS': 'srs'})
            else:
                df_adv = pd.DataFrame(columns=['team', 'srs'])
                print(f"SRS column not found for {season_end_year}")
        else:
            print(f"Advanced stats table not found for {season_end_year}")
            df_adv = pd.DataFrame(columns=['team', 'srs'])
    except Exception as e:
        print(f"Error fetching Basketball-Reference data for {season_end_year}: {e}")
        df_adv = pd.DataFrame(columns=['team', 'srs'])
    
    # 3. Add champion information
    if season_end_year in NBA_CHAMPIONS:
        champion_team = NBA_CHAMPIONS[season_end_year]
        df_champion = pd.DataFrame({'team': [champion_team], 'is_champion': [True]})
    
    # Merge all data
    df_season = reg.copy()
    
    if not df_adv.empty:
        # Clean team names for better matching
        df_adv['team'] = df_adv['team'].str.replace('*', '', regex=False)
        df_season = df_season.merge(df_adv, on='team', how='left')
    
    # Add champion information
    if not df_champion.empty and 'is_champion' in df_champion.columns:
        df_season = df_season.merge(df_champion, on='team', how='left')
        df_season['is_champion'] = df_season['is_champion'].fillna(False)
    else:
        df_season['is_champion'] = False
    
    df_season['season_end'] = season_end_year
    return df_season

# Fetch and combine all seasons from 2001 to 2025
all_data = []
for year in range(2001, 2026):
    print(f"Fetching metrics for {year} season...")
    try:
        year_df = fetch_season_metrics(year)
        if not year_df.empty:
            all_data.append(year_df)
            print(f"  ✓ Successfully added data for {year} season")
    except Exception as e:
        print(f"  → Error for {year}: {e}")
    time.sleep(1)  # NBA API rate‐limit courtesy

if all_data:
    master_df = pd.concat(all_data, ignore_index=True)
    print(f"Final dataset shape: {master_df.shape}")
    
    # Save to CSV for downstream modeling - handle permission error
    output_file = 'nba_team_season_metrics_2001_2025.csv'
    
    # Try to save the file, handling permission errors
    try:
        # Check if file exists and try to remove it first
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Removed existing file: {output_file}")
            except PermissionError:
                # If can't remove, try a different filename
                output_file = 'nba_team_season_metrics_new.csv'
                print(f"Could not remove existing file, using new filename: {output_file}")
        
        # Save the data
        master_df.to_csv(output_file, index=False)
        print(f"All seasons saved to '{output_file}'")
    except PermissionError:
        # If still getting permission error, try saving to a different location
        alt_output_file = os.path.join(os.path.expanduser("~"), "Desktop", "nba_team_season_metrics.csv")
        try:
            master_df.to_csv(alt_output_file, index=False)
            print(f"Permission denied on original file. Data saved to: {alt_output_file}")
        except Exception as e:
            print(f"Could not save data: {e}")
            # As a last resort, display the first few rows
            print("\nFirst few rows of the data:")
            print(master_df.head())
else:
    print("No data was collected. Check the errors above.")
