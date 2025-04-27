import pandas as pd

# Load the files
playoff_series = pd.read_csv("NBA_2025_Playoff_Series.csv")
season_metrics = pd.read_csv("nba_team_season_metrics_2001_2025.csv")

# Filter metrics to only include 2025 season
season_metrics_2025 = season_metrics[season_metrics['season_end'] == 2025].copy()

# Create dictionary to map team names to their metrics
team_metrics = {}
for _, row in season_metrics_2025.iterrows():
    team_name = row['team']
    team_metrics[team_name] = {
        'win_pct': row['win_pct'],
        'off_rtg': row['off_rtg'],
        'def_rtg': row['def_rtg'],
        'net_rtg': row['net_rtg'],
        'pace': row['pace'],
        'efg_pct': row['efg_pct'],
        'ts_pct': row['ts_pct'],
        'tov_pct': row['tov_pct'],
        'orb_pct': row['orb_pct'],
        'drb_pct': row['drb_pct']
    }

# Populate metrics for both visitor and home teams
for i, row in playoff_series.iterrows():
    # Populate visitor team metrics
    visitor_team = row['Visitor']
    if visitor_team in team_metrics:
        metrics = team_metrics[visitor_team]
        playoff_series.at[i, 'Visitor win_pct'] = metrics['win_pct']
        playoff_series.at[i, 'Visitor off_rtg'] = metrics['off_rtg']
        playoff_series.at[i, 'Visitor def_rtg'] = metrics['def_rtg']
        playoff_series.at[i, 'Visitor net_rtg'] = metrics['net_rtg']
        playoff_series.at[i, 'Visitor pace'] = metrics['pace']
        playoff_series.at[i, 'Visitor efg_pct'] = metrics['efg_pct']
        playoff_series.at[i, 'Visitor ts_pct'] = metrics['ts_pct']
        playoff_series.at[i, 'Visitor tov_pct'] = metrics['tov_pct']
        playoff_series.at[i, 'Visitor orb_pct'] = metrics['orb_pct']
        playoff_series.at[i, 'Visitor drb_pct'] = metrics['drb_pct']
    
    # Populate home team metrics
    home_team = row['Home']
    if home_team in team_metrics:
        metrics = team_metrics[home_team]
        playoff_series.at[i, 'Home win_pct'] = metrics['win_pct']
        playoff_series.at[i, 'Home off_rtg'] = metrics['off_rtg']
        playoff_series.at[i, 'Home def_rtg'] = metrics['def_rtg']
        playoff_series.at[i, 'Home net_rtg'] = metrics['net_rtg']
        playoff_series.at[i, 'Home pace'] = metrics['pace']
        playoff_series.at[i, 'Home efg_pct'] = metrics['efg_pct']
        playoff_series.at[i, 'Home ts_pct'] = metrics['ts_pct']
        playoff_series.at[i, 'Home tov_pct'] = metrics['tov_pct']
        playoff_series.at[i, 'Home orb_pct'] = metrics['orb_pct']
        playoff_series.at[i, 'Home drb_pct'] = metrics['drb_pct']

# Save the updated playoff series data
playoff_series.to_csv("NBA_2025_Playoff_Series_with_metrics.csv", index=False)

print("Playoff series data has been updated with team metrics.")
