import pandas as pd

# 1. Load data
season_metrics = pd.read_csv('nba_team_season_metrics_2001_2025.csv')
df_stats = pd.read_csv('NBA_2001_2005_All team stats.csv')

# 2. First merge Visitor stats
df_filled = df_stats.merge(
    season_metrics,
    left_on=['Year', 'Visitor/Neutral'],    
    right_on=['season_end', 'team'],            
    how='left'
).rename(columns={
    # rename each metric from CSV to match your "Visitor X" naming
    'win_pct': 'Visitor win_pct',
    'off_rtg': 'Visitor off_rtg',
    'def_rtg': 'Visitor def_rtg',
    'net_rtg': 'Visitor net_rtg',
    'pace':    'Visitor pace',
    'efg_pct': 'Visitor efg_pct',
    'ts_pct':  'Visitor ts_pct',
    'tov_pct': 'Visitor tov_pct',
    'orb_pct': 'Visitor orb_pct',
    'drb_pct': 'Visitor drb_pct'
})

# 3. Then merge Home stats
df_filled = df_filled.merge(
    season_metrics,
    left_on=['Year', 'Home/Neutral'],
    right_on=['season_end', 'team'],
    how='left'
).rename(columns={
    'win_pct': 'Home win_pct',
    'off_rtg': 'Home off_rtg',
    'def_rtg': 'Home def_rtg',
    'net_rtg': 'Home net_rtg',
    'pace':    'Home pace',
    'efg_pct': 'Home efg_pct',
    'ts_pct':  'Home ts_pct',
    'tov_pct': 'Home tov_pct',
    'orb_pct': 'Home orb_pct',
    'drb_pct': 'Home drb_pct'
})

# 4. Drop extra CSV columns
df_filled = df_filled.drop(columns=['season_end_x','team_x','season_end_y','team_y'])

# 5. Save to a new csv file
df_filled.to_csv(
    'NBA_2001_2005_All team stats_filled.csv',
    index=False
)
print("Finished filling.  Saved to NBA_2001_2005_All team stats_filled.csv")
