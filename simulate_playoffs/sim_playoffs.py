import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
import random
import matplotlib.pyplot as plt
from collections import Counter


games = pd.read_csv('NBA_2001_2025_All_team_stats_filled.csv')
bracket = pd.read_csv('NBA_2025_Playoff_Series.csv')

X = games[[
  'Visitor Seed', 'Home Seed',
  'Visitor win_pct', 'Home win_pct',
  'Visitor off_rtg', 'Home off_rtg',
  'Visitor def_rtg', 'Home def_rtg',
  'Visitor net_rtg', 'Home net_rtg',
  'Visitor pace', 'Home pace',
  'Visitor efg_pct', 'Home efg_pct',
  'Visitor ts_pct', 'Home ts_pct',
  'Visitor tov_pct', 'Home tov_pct',
  'Visitor orb_pct', 'Home orb_pct',
  'Visitor drb_pct', 'Home drb_pct',
]]
y = games['Win']

# split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=games['Year'], random_state=42
)


# set up RF with some basic tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0, 0.01, 0.1]
}

clf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
clf.fit(X_train, y_train)

print("Val AUC:", roc_auc_score(y_val, clf.predict_proba(X_val)[:,1]))
print("Best params:", clf.best_params_)
model = clf.best_estimator_

def game_win_prob(visitor_stats, home_stats):
    '''Given two 1D-arrays of stats (visitor vs home), return P(visitor wins).'''
    Xg = pd.DataFrame([{
      'Visitor Seed': visitor_stats['seed'], 
      'Home Seed':    home_stats['seed'],
      'Visitor win_pct': visitor_stats['win_pct'],
      'Home win_pct':    home_stats['win_pct'],
      'Visitor off_rtg': visitor_stats['off_rtg'],
      'Home off_rtg':    home_stats['off_rtg'],
      'Visitor def_rtg': visitor_stats['def_rtg'],
      'Home def_rtg':    home_stats['def_rtg'],
      'Visitor net_rtg': visitor_stats['net_rtg'],
      'Home net_rtg':    home_stats['net_rtg'],
      'Visitor pace': visitor_stats['pace'],
      'Home pace':    home_stats['pace'],
      'Visitor efg_pct': visitor_stats['efg_pct'],
      'Home efg_pct':    home_stats['efg_pct'],
      'Visitor ts_pct': visitor_stats['ts_pct'],
      'Home ts_pct':    home_stats['ts_pct'],
      'Visitor tov_pct': visitor_stats['tov_pct'],
      'Home tov_pct':    home_stats['tov_pct'],
      'Visitor orb_pct': visitor_stats['orb_pct'],
      'Home orb_pct':    home_stats['orb_pct'],
      'Visitor drb_pct': visitor_stats['drb_pct'],
      'Home drb_pct':    home_stats['drb_pct'],
    }])
    # Ensure columns are in the same order as during training
    Xg = Xg[X.columns]
    return model.predict_proba(Xg)[:,1][0]  # P(visitor wins)

def simulate_series(team_high, team_low, stats_df):
    '''
    team_high/low: team names (high seed has home‐court)
    stats_df: 2025 stats loader (or bracket lookup)
    Returns winner name.
    '''
    # pull each team's season stats + seed
    h = stats_df[stats_df['Home/Neutral'] == team_high].iloc[0]
    l = stats_df[stats_df['Visitor/Neutral'] == team_low].iloc[0]
    high = {'seed': h['Home Seed'],
            'win_pct': h['Home win_pct'],
            'net_rtg': h['Home net_rtg'],
            'pace': h['Home pace'],
            'off_rtg': h['Home off_rtg'],
            'def_rtg': h['Home def_rtg'],
            'efg_pct': h['Home efg_pct'],
            'ts_pct': h['Home ts_pct'],
            'tov_pct': h['Home tov_pct'],
            'orb_pct': h['Home orb_pct'],
            'drb_pct': h['Home drb_pct'],
            }
    low  = {'seed': l['Visitor Seed'],
            'win_pct': l['Visitor win_pct'],
            'net_rtg': l['Visitor net_rtg'],
            'pace': l['Visitor pace'],
            'off_rtg': l['Visitor off_rtg'],
            'def_rtg': l['Visitor def_rtg'],
            'efg_pct': l['Visitor efg_pct'],
            'ts_pct': l['Visitor ts_pct'],
            'tov_pct': l['Visitor tov_pct'],
            'orb_pct': l['Visitor orb_pct'],
            'drb_pct': l['Visitor drb_pct'],
            }

    # NBA home/away pattern
    schedule = [
      ('home', 'home'),  # G1  H
      ('home', 'home'),  # G2  H
      ('away', 'away'),  # G3  L
      ('away', 'away'),  # G4  L
      ('home', 'home'),  # G5  H
      ('away', 'away'),  # G6  L
      ('home', 'home')   # G7  H
    ]

    wins = {team_high: 0, team_low: 0}
    for H, _ in schedule:
        if wins[team_high] == 4 or wins[team_low] == 4:
            break
        if H == 'home':
            p_vis = game_win_prob(low, high)
            # "visitor" here is the lower seed on the road
            visitor, home = team_low, team_high
        else:  
            p_vis = game_win_prob(high, low)
            visitor, home = team_high, team_low

        # sample outcome
        if random.random() < p_vis:
            wins[visitor] += 1
        else:
            wins[home] += 1

    return team_high if wins[team_high] > wins[team_low] else team_low

def simulate_2025_bracket(bracket_df, stats_df):
    winners = {}

    # Round 1 — West
    rd1_w = bracket_df[bracket_df['Series']=='rd1_west']
    w0 = simulate_series(rd1_w.iloc[0]['Home'], rd1_w.iloc[0]['Visitor'], stats_df)
    w1 = simulate_series(rd1_w.iloc[1]['Home'], rd1_w.iloc[1]['Visitor'], stats_df)
    w2 = simulate_series(rd1_w.iloc[2]['Home'], rd1_w.iloc[2]['Visitor'], stats_df)
    w3 = simulate_series(rd1_w.iloc[3]['Home'], rd1_w.iloc[3]['Visitor'], stats_df)
    winners['west_r2_0'] = simulate_series(w0, w1, stats_df)
    winners['west_r2_1'] = simulate_series(w2, w3, stats_df)
    winners['west_cf']  = simulate_series(winners['west_r2_0'],
                                          winners['west_r2_1'], stats_df)

    # Round 1 — East (same pattern)
    rd1_e = bracket_df[bracket_df['Series']=='rd1_east']
    e0 = simulate_series(rd1_e.iloc[0]['Home'], rd1_e.iloc[0]['Visitor'], stats_df)
    e1 = simulate_series(rd1_e.iloc[1]['Home'], rd1_e.iloc[1]['Visitor'], stats_df)
    e2 = simulate_series(rd1_e.iloc[2]['Home'], rd1_e.iloc[2]['Visitor'], stats_df)
    e3 = simulate_series(rd1_e.iloc[3]['Home'], rd1_e.iloc[3]['Visitor'], stats_df)
    winners['east_r2_0'] = simulate_series(e0, e1, stats_df)
    winners['east_r2_1'] = simulate_series(e2, e3, stats_df)
    winners['east_cf']  = simulate_series(winners['east_r2_0'],
                                          winners['east_r2_1'], stats_df)

    # NBA Finals
    champion = simulate_series(winners['west_cf'], winners['east_cf'], stats_df)
    return champion

def estimate_champion_probs(n_sims=1_000):
    """Run n_sims brackets and return a dict of P(team wins title)."""
    counts = Counter()
    
    # Performance optimization - run fewer simulations if needed
    n_actual = min(n_sims, 100)  # Start with 100 sims for quicker results
    
    for _ in range(n_actual):
        counts[simulate_2025_bracket(bracket, games)] += 1
    
    # Convert counts to probabilities
    return {team: count / n_actual for team, count in counts.items()}

if __name__ == '__main__':
    # 1) Run the sims
    probs = estimate_champion_probs(n_sims=100)

    # 2) Sort teams by probability descending
    teams, values = zip(*sorted(probs.items(), key=lambda x: x[1], reverse=True))

    # 3) Plot
    plt.figure(figsize=(12, 6))
    plt.bar(teams, values)
    plt.xticks(rotation=90)
    plt.ylabel('Probability')
    plt.title('Estimated Probability of 2025 NBA Championship by Team')
    plt.tight_layout()
    plt.show()