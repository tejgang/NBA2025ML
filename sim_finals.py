import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("NBA Finals 2025 Simulation: OKC Thunder vs Indiana Pacers")
print("="*60)

# 1. LOAD AND PREPARE DATA
print("\n1. LOADING DATA...")
df = pd.read_csv("NBA_2025_Playoff_Series_with_metrics.csv")

# Define metrics we'll use as features
metrics = ['Seed', 'win_pct', 'off_rtg', 'def_rtg', 'net_rtg', 'pace', 'efg_pct', 'ts_pct', 'tov_pct', 'orb_pct', 'drb_pct']

# Create feature differences (Visitor - Home)
for metric in metrics:
    df[f'diff_{metric}'] = df[f'Visitor {metric}'] - df[f'Home {metric}']

# Prepare features and target
feature_columns = [f'diff_{metric}' for metric in metrics]
X = df[feature_columns].dropna()
y = df.loc[X.index, 'Win']  # Visitor team wins

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 2. EXPLORATORY DATA ANALYSIS
print("\n2. EXPLORATORY DATA ANALYSIS...")

def plot_eda():
    """Create EDA visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 2a. Correlation heatmap
    feature_names = [name.replace('diff_', '').replace('_', ' ').title() for name in feature_columns]
    corr_data = X.copy()
    corr_data.columns = feature_names
    corr_data['Win'] = y
    
    corr_matrix = corr_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', ax=axes[0,0])
    axes[0,0].set_title('Feature Correlation Matrix')
    
    # 2b. Feature importance vs target
    correlations = corr_data.corr()['Win'].drop('Win').sort_values(key=abs, ascending=False)
    colors = ['red' if x < 0 else 'blue' for x in correlations.values]
    
    axes[0,1].barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
    axes[0,1].set_yticks(range(len(correlations)))
    axes[0,1].set_yticklabels(correlations.index, fontsize=10)
    axes[0,1].set_xlabel('Correlation with Win')
    axes[0,1].set_title('Feature Correlations with Target')
    axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 2c. Distribution of key features by outcome
    key_features = ['diff_net_rtg', 'diff_win_pct']
    for i, feature in enumerate(key_features):
        if feature in X.columns:
            ax = axes[1, i]
            for outcome in [0, 1]:
                data = X[y == outcome][feature]
                ax.hist(data, alpha=0.7, label=f'{"Loss" if outcome == 0 else "Win"}', bins=20)
            ax.set_xlabel(feature.replace('diff_', '').replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature.replace("diff_", "").replace("_", " ").title()}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_eda()

# 3. MODEL TRAINING WITH HYPERPARAMETER TUNING
print("\n3. MODEL TRAINING...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0, 0.01, 0.1]
}

# Grid search
print("Running hyperparameter tuning...")
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Feature importance
def plot_feature_importance():
    """Plot feature importance"""
    importances = best_model.feature_importances_
    feature_names = [name.replace('diff_', '').replace('_', ' ').title() for name in feature_columns]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', alpha=0.8)
    
    # Add value labels
    for bar, importance in zip(bars, importance_df['Importance']):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=9)
    
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_feature_importance()

# 4. SIMULATION SETUP
print("\n4. SETTING UP FINALS SIMULATION...")

def get_team_stats(team_name):
    """Extract 2025 season stats for a team from the dataset"""
    df_2025 = df[df['Year'] == 2025]
    
    # Find team in visitor column
    visitor_row = df_2025[df_2025['Visitor/Neutral'] == team_name]
    if not visitor_row.empty:
        row = visitor_row.iloc[0]
        return {metric: row[f'Visitor {metric}'] for metric in metrics}
    
    # Find team in home column
    home_row = df_2025[df_2025['Home/Neutral'] == team_name]
    if not home_row.empty:
        row = home_row.iloc[0]
        return {metric: row[f'Home {metric}'] for metric in metrics}

# Get team stats
thunder_stats = get_team_stats('Oklahoma City Thunder')
pacers_stats = get_team_stats('Indiana Pacers')

print(f"Thunder: {thunder_stats}")
print(f"Pacers: {pacers_stats}")

def calculate_game_probability(visitor_stats, home_stats):
    """Calculate probability that visitor wins"""
    differences = [visitor_stats[metric] - home_stats[metric] for metric in metrics]
    return best_model.predict_proba([differences])[0][1]

# Calculate game probabilities for 2-2-1-1-1 format
home_schedule = ['OKC', 'OKC', 'IND', 'IND', 'OKC', 'IND', 'OKC']
game_probs = []

for i, home_team in enumerate(home_schedule):
    if home_team == 'OKC':
        # Thunder home, Pacers visitor
        prob_pacers_win = calculate_game_probability(pacers_stats, thunder_stats)
        prob_thunder_win = 1 - prob_pacers_win
    else:
        # Pacers home, Thunder visitor  
        prob_thunder_win = calculate_game_probability(thunder_stats, pacers_stats)
        prob_pacers_win = 1 - prob_thunder_win
    
    game_probs.append(prob_thunder_win)
    print(f"Game {i+1}: {'IND' if home_team == 'OKC' else 'OKC'} @ {home_team} - Thunder win prob: {prob_thunder_win:.3f}")

# 5. MONTE CARLO SIMULATION
print("\n5. RUNNING MONTE CARLO SIMULATION...")

n_simulations = 10000
thunder_series_wins = 0
simulation_results = []
probability_tracking = []

np.random.seed(42)

for sim in range(n_simulations):
    thunder_wins = 0
    pacers_wins = 0
    
    for game in range(7):
        if np.random.random() < game_probs[game]:
            thunder_wins += 1
        else:
            pacers_wins += 1
        
        if thunder_wins == 4 or pacers_wins == 4:
            break
    
    if thunder_wins > pacers_wins:
        thunder_series_wins += 1
    
    # Track progress every 100 simulations
    if (sim + 1) % 100 == 0:
        current_prob = thunder_series_wins / (sim + 1)
        probability_tracking.append({
            'simulation': sim + 1,
            'thunder_prob': current_prob,
            'pacers_prob': 1 - current_prob
        })

final_thunder_prob = thunder_series_wins / n_simulations
final_pacers_prob = 1 - final_thunder_prob

# 6. RESULTS AND VISUALIZATION
print("\n6. SIMULATION RESULTS...")

def plot_simulation_results():
    """Plot simulation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 6a. Monte Carlo convergence
    tracking_df = pd.DataFrame(probability_tracking)
    ax1.plot(tracking_df['simulation'], tracking_df['thunder_prob'], 'b-', linewidth=2, label='Thunder')
    ax1.plot(tracking_df['simulation'], tracking_df['pacers_prob'], 'gold', linewidth=2, label='Pacers')
    ax1.axhline(y=final_thunder_prob, color='blue', linestyle='--', alpha=0.7)
    ax1.axhline(y=final_pacers_prob, color='orange', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Win Probability')
    ax1.set_title('Monte Carlo Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 6b. Final probabilities
    teams = ['OKC Thunder', 'Indiana Pacers']
    probabilities = [final_thunder_prob, final_pacers_prob]
    colors = ['#007AC1', '#FDBB30']
    
    bars = ax2.bar(teams, probabilities, color=colors, alpha=0.8, edgecolor='black')
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_ylabel('Series Win Probability')
    ax2.set_title('2025 NBA Finals Prediction')
    ax2.set_ylim(0, 1)
    
    # 6c. Game-by-game probabilities
    games = [f'Game {i+1}' for i in range(7)]
    colors = ['lightblue' if team == 'OKC' else 'lightcoral' for team in home_schedule]
    
    bars = ax3.bar(games, game_probs, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Thunder Win Probability')
    ax3.set_title('Game-by-Game Win Probabilities\n(Blue = Thunder Home, Red = Pacers Home)')
    ax3.set_ylim(0, 1)
    
    for bar, prob in zip(bars, game_probs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6d. Probability distribution histogram
    all_game_probs = game_probs * (n_simulations // 7)  # Approximate distribution
    ax4.hist(all_game_probs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(x=np.mean(game_probs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(game_probs):.3f}')
    ax4.set_xlabel('Thunder Win Probability per Game')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Game Win Probabilities')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_simulation_results()

# FINAL RESULTS
print("\n" + "="*60)
print("FINAL PREDICTION")
print("="*60)
print(f"Oklahoma City Thunder: {final_thunder_prob:.1%}")
print(f"Indiana Pacers: {final_pacers_prob:.1%}")

if final_thunder_prob > final_pacers_prob:
    favorite = "Oklahoma City Thunder"
    advantage = final_thunder_prob - final_pacers_prob
else:
    favorite = "Indiana Pacers"
    advantage = final_pacers_prob - final_thunder_prob

print(f"\n WINNER: {favorite}")
print(f" Advantage: {advantage:.1%}")
print(f" Based on {n_simulations:,} Monte Carlo simulations")

print("\n Analysis complete! Graphs saved:")
print("   eda_analysis.png - Exploratory Data Analysis")
print("   feature_importance.png - Model Feature Importance")  
print("   simulation_results.png - Simulation Results")
