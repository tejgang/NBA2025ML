import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
df = pd.read_csv("NBA_2025_Playoff_Series_with_metrics.csv")

# 2. Prepare training data: use all seasons before 2025 to avoid leakage
df_train = df[df['Year'] < 2025].copy()

# Define the performance‐metric columns
metrics = [
    'win_pct', 'off_rtg', 'def_rtg', 'net_rtg', 
    'pace', 'efg_pct', 'ts_pct', 'tov_pct', 'orb_pct', 'drb_pct'
]

# 3. Create feature differences (Visitor metric minus Home metric) for training
for m in metrics:
    df_train[f'diff_{m}'] = df_train[f'Visitor {m}'] - df_train[f'Home {m}']

diff_features = [f'diff_{m}' for m in metrics]

# Target: “Win” indicates whether the Visitor team won (1) or not (0)
y_train = df_train['Win']
X_train = df_train[diff_features]

# 4. Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Helper function to extract a team’s 2025 metrics from the DataFrame
def get_team_metrics(team_name):
    df_2025 = df[df['Year'] == 2025]
    # Check if team appears as Visitor in any 2025 game
    row_vis = df_2025[df_2025['Visitor/Neutral'] == team_name]
    if not row_vis.empty:
        row = row_vis.iloc[0]
        prefix = 'Visitor'
    else:
        row_home = df_2025[df_2025['Home/Neutral'] == team_name]
        row = row_home.iloc[0]
        prefix = 'Home'
    return {m: row[f'{prefix} {m}'] for m in metrics}

# Get the 2025 regular‐season metrics for OKC and Indiana
indiana_metrics = get_team_metrics('Indiana Pacers')
okc_metrics = get_team_metrics('Oklahoma City Thunder')

# 6. Function to compute home‐team win probability given two metric dicts
def home_win_probability(visitor_metrics, home_metrics):
    diff_array = np.array([visitor_metrics[m] - home_metrics[m] for m in metrics]).reshape(1, -1)
    prob_visitor = rf_model.predict_proba(diff_array)[0][1]
    return 1 - prob_visitor  # this is P(home team wins)

# 7. NBA Finals home‐court pattern: 2–2–1–1–1 for the higher seed (OKC)
home_schedule_okc = [True, True, False, False, True, False, True]

prob_home_wins = []
for is_home_okc in home_schedule_okc:
    if is_home_okc:
        # OKC is home, Indiana visitor
        p_home = home_win_probability(indiana_metrics, okc_metrics)
    else:
        # Indiana is home, OKC visitor
        p_home = home_win_probability(okc_metrics, indiana_metrics)
    prob_home_wins.append(p_home)

# 8. Simulate the best‐of‐7 series many times
n_simulations = 10_000
okc_series_wins = 0
rng = np.random.default_rng(42)

for _ in range(n_simulations):
    okc_wins = 0
    ind_wins = 0
    for i in range(7):
        is_home_okc = home_schedule_okc[i]
        p_home = prob_home_wins[i]
        if rng.random() < p_home:
            # Home team wins
            if is_home_okc:
                okc_wins += 1
            else:
                ind_wins += 1
        else:
            # Visitor team wins
            if is_home_okc:
                ind_wins += 1
            else:
                okc_wins += 1
        # Stop if one team reaches 4 wins
        if okc_wins == 4 or ind_wins == 4:
            break
    if okc_wins > ind_wins:
        okc_series_wins += 1

okc_probability = okc_series_wins / n_simulations
indiana_probability = 1 - okc_probability

# 9. Print the final result
print(f"Oklahoma City Thunder win probability: {okc_probability:.4f}")
print(f"Indiana Pacers win probability: {indiana_probability:.4f}")
