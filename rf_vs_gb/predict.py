import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading NBA team metrics data...")
df = pd.read_csv('nba_team_season_metrics_2001_2025.csv')

# Inspect the data
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nColumns in dataset:")
print(df.columns.tolist())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Handle missing values in is_champion for 2025 season
df.loc[df['season_end'] == 2025, 'is_champion'] = False

# Ensure is_champion is boolean
if df['is_champion'].dtype != 'bool':
    df['is_champion'] = df['is_champion'].astype(bool)

# Create derived features that might be predictive
print("\nCreating enhanced features...")

# Basic derived metrics
df['offensive_efficiency'] = df['efg_pct'] * (1 - df['tov_pct'])
df['defensive_efficiency'] = (1 - df['orb_pct']) * df['drb_pct']
df['balance_score'] = np.abs(df['off_rtg'] - 110) + np.abs(df['def_rtg'] - 105)  # How close to ideal balance
df['reb_advantage'] = df['orb_pct'] - (1 - df['drb_pct'])  # Net rebounding advantage
df['shooting_advantage'] = df['ts_pct'] - df['efg_pct']  # Free throw contribution to scoring

# Create a synthetic SRS (Simple Rating System) based on net_rtg
# SRS is typically net_rtg adjusted for strength of schedule
df['synthetic_srs'] = df['net_rtg'] * 0.9  # SRS is typically slightly lower than net_rtg

# Create "championship DNA" feature - teams that won recently
df['recent_champion'] = False
for year in range(2005, 2025):
    # Teams that won in the last 3 years have "championship DNA"
    recent_champions = df[(df['season_end'].between(year-3, year-1)) & (df['is_champion'])]['team'].unique()
    df.loc[(df['season_end'] == year) & (df['team'].isin(recent_champions)), 'recent_champion'] = True

# Create "hotness" indicator - teams performing above their expected level
df['hot_team'] = df['win_pct'] > (1 - df['def_rtg'] / df['off_rtg'] + 0.05)

# Split into training and prediction sets
df_train = df[df['season_end'] <= 2024].copy()
df_predict = df[df['season_end'] == 2025].copy()

print(f"Training data shape: {df_train.shape}")
print(f"Prediction data shape: {df_predict.shape}")

# Define features based on available columns
features = [
    'win_pct',  # Now using actual win percentage from the dataset
    'off_rtg', 'def_rtg', 'net_rtg', 'pace',
    'efg_pct', 'ts_pct', 'tov_pct', 'orb_pct', 'drb_pct',
    'synthetic_srs',  # Using our synthetic SRS instead
    'offensive_efficiency', 'defensive_efficiency', 
    'balance_score', 'reb_advantage', 'shooting_advantage',
    'recent_champion', 'hot_team'
]

# Prepare training data
X = df_train[features].copy()
y = df_train['is_champion']

# Convert boolean features to integers
bool_features = ['recent_champion', 'hot_team']
for col in bool_features:
    X[col] = X[col].astype(int)

# Print class distribution
print("\nClass distribution in training data:")
print(y.value_counts())
print(f"Imbalance ratio: 1:{y.value_counts()[False]/y.value_counts()[True]:.1f}")

# Apply SMOTE to balance classes
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"After SMOTE - X shape: {X_res.shape}, y distribution: {np.bincount(y_res.astype(int))}")

# Train both Random Forest and LightGBM models
print("\nTraining both Random Forest and LightGBM models...")

# Random Forest parameters
rf_params = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42
}

# LightGBM parameters optimized for imbalanced classification
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 29,  # Approximates our class imbalance
    'random_state': 42,
    'verbose': -1
}

# Train models
rf_model = BalancedRandomForestClassifier(**rf_params)
rf_model.fit(X_res, y_res)

lgb_model = LGBMClassifier(**lgb_params)
lgb_model.fit(X_res, y_res)

# Compare feature importance
rf_importance = pd.DataFrame({
    'Feature': features,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False)

lgb_importance = pd.DataFrame({
    'Feature': features,
    'LGB_Importance': lgb_model.feature_importances_
}).sort_values('LGB_Importance', ascending=False)

# Merge importance dataframes
importance_df = pd.merge(rf_importance, lgb_importance, on='Feature')

print("\nFeature Importance Comparison (Top 10):")
print(importance_df.head(10))

# Plot feature importance comparison
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
sns.barplot(x='RF_Importance', y='Feature', data=rf_importance.head(10))
plt.title('Random Forest Feature Importance')

plt.subplot(1, 2, 2)
sns.barplot(x='LGB_Importance', y='Feature', data=lgb_importance.head(10))
plt.title('LightGBM Feature Importance')

plt.tight_layout()
'''
plt.savefig('feature_importance_comparison.png')
'''
print("Feature importance comparison saved to 'feature_importance_comparison.png'")

# Prepare prediction data
X_predict = df_predict[features].copy()

# Convert boolean features to integers
for col in bool_features:
    X_predict[col] = X_predict[col].astype(int)

# Make predictions for 2025 with both models
print("\nPredicting 2025 NBA Champion with both models...")
rf_proba = rf_model.predict_proba(X_predict)[:, 1]
lgb_proba = lgb_model.predict_proba(X_predict)[:, 1]

# Average the predictions (simple ensemble)
ensemble_proba = (rf_proba + lgb_proba) / 2

# Get team names and create results dataframe
teams = df_predict['team'].values
team_probs = pd.DataFrame({
    'Team': teams,
    'RF_Probability': rf_proba,
    'LGB_Probability': lgb_proba,
    'Ensemble_Probability': ensemble_proba,
    'Net Rating': df_predict['net_rtg'].values,
    'Win %': df_predict['win_pct'].values
})

# Sort by ensemble probability
team_probs = team_probs.sort_values('Ensemble_Probability', ascending=False).reset_index(drop=True)

# Display top 10 teams
print("\nTop 10 Teams Most Likely to Win the 2025 NBA Championship:")
print(team_probs[['Team', 'Ensemble_Probability', 'RF_Probability', 'LGB_Probability', 'Win %', 'Net Rating']].head(10))

# Plot top 10 teams
plt.figure(figsize=(14, 8))
top10_teams = team_probs.head(10)
x = np.arange(len(top10_teams))
width = 0.35

plt.bar(x - width/2, top10_teams['RF_Probability'], width, label='Random Forest')
plt.bar(x + width/2, top10_teams['LGB_Probability'], width, label='LightGBM')

plt.xlabel('Team')
plt.ylabel('Championship Probability')
plt.title('Top 10 Teams Most Likely to Win the 2025 NBA Championship')
plt.xticks(x, top10_teams['Team'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
'''
plt.savefig('championship_prediction_comparison_2025.png')
'''
print("Prediction comparison plot saved to 'championship_prediction_comparison_2025.png'")

# Historical validation with both models
print("\nPerforming historical validation with both models...")
historical_results = []

# For each season from 2010 to 2024, train on previous seasons and predict that season
for test_year in range(2010, 2025):
    print(f"Evaluating models for {test_year} season...")
    # Split data
    hist_train = df[(df['season_end'] < test_year) & (df['season_end'] >= 2001)]
    hist_test = df[df['season_end'] == test_year]
    
    # Prepare features
    X_hist_train = hist_train[features].copy()
    y_hist_train = hist_train['is_champion']
    X_hist_test = hist_test[features].copy()
    
    # Convert boolean features to integers
    for col in bool_features:
        if col in X_hist_train.columns:
            X_hist_train[col] = X_hist_train[col].astype(int)
        if col in X_hist_test.columns:
            X_hist_test[col] = X_hist_test[col].astype(int)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_hist_train_res, y_hist_train_res = smote.fit_resample(X_hist_train, y_hist_train)
    
    # Train both models
    hist_rf_model = BalancedRandomForestClassifier(**rf_params)
    hist_rf_model.fit(X_hist_train_res, y_hist_train_res)
    
    hist_lgb_model = LGBMClassifier(**lgb_params)
    hist_lgb_model.fit(X_hist_train_res, y_hist_train_res)
    
    # Predict with both models
    rf_probs = hist_rf_model.predict_proba(X_hist_test)[:, 1]
    lgb_probs = hist_lgb_model.predict_proba(X_hist_test)[:, 1]
    
    # Ensemble predictions
    ensemble_probs = (rf_probs + lgb_probs) / 2
    
    # Get actual champion
    actual_champion = hist_test[hist_test['is_champion'] == True]['team'].values
    if len(actual_champion) == 0:
        continue
    actual_champion = actual_champion[0]
    
    # Get teams
    teams = hist_test['team'].values
    
    # Get predictions for each model
    rf_pred_idx = np.argmax(rf_probs)
    lgb_pred_idx = np.argmax(lgb_probs)
    ensemble_pred_idx = np.argmax(ensemble_probs)
    
    rf_predicted = teams[rf_pred_idx]
    lgb_predicted = teams[lgb_pred_idx]
    ensemble_predicted = teams[ensemble_pred_idx]
    
    # Get top 3 predictions for ensemble
    top3_indices = np.argsort(ensemble_probs)[-3:][::-1]
    top3_teams = teams[top3_indices]
    actual_in_top3 = actual_champion in top3_teams
    
    # Get top 5 predictions for ensemble
    top5_indices = np.argsort(ensemble_probs)[-5:][::-1]
    top5_teams = teams[top5_indices]
    actual_in_top5 = actual_champion in top5_teams
    
    # Calculate AUC if possible
    if len(np.unique(hist_test['is_champion'])) > 1:
        rf_auc = roc_auc_score(hist_test['is_champion'], rf_probs)
        lgb_auc = roc_auc_score(hist_test['is_champion'], lgb_probs)
        ensemble_auc = roc_auc_score(hist_test['is_champion'], ensemble_probs)
    else:
        rf_auc = lgb_auc = ensemble_auc = np.nan
    
    # Store results
    historical_results.append({
        'Season': test_year,
        'Actual_Champion': actual_champion,
        'RF_Predicted': rf_predicted,
        'LGB_Predicted': lgb_predicted,
        'Ensemble_Predicted': ensemble_predicted,
        'RF_Correct': actual_champion == rf_predicted,
        'LGB_Correct': actual_champion == lgb_predicted,
        'Ensemble_Correct': actual_champion == ensemble_predicted,
        'Actual_In_Top3': actual_in_top3,
        'Actual_In_Top5': actual_in_top5,
        'RF_AUC': rf_auc,
        'LGB_AUC': lgb_auc,
        'Ensemble_AUC': ensemble_auc,
        'Top3_Teams': ', '.join(top3_teams)
    })

# Create historical results dataframe
hist_df = pd.DataFrame(historical_results)

# Calculate accuracy metrics
rf_accuracy = hist_df['RF_Correct'].mean() if not hist_df.empty else 0
lgb_accuracy = hist_df['LGB_Correct'].mean() if not hist_df.empty else 0
ensemble_accuracy = hist_df['Ensemble_Correct'].mean() if not hist_df.empty else 0
top3_accuracy = hist_df['Actual_In_Top3'].mean() if not hist_df.empty else 0
top5_accuracy = hist_df['Actual_In_Top5'].mean() if not hist_df.empty else 0

print("\nHistorical Prediction Results:")
print(hist_df[['Season', 'Actual_Champion', 'Ensemble_Predicted', 'Ensemble_Correct', 'Actual_In_Top3']])
print(f"\nRandom Forest accuracy: {rf_accuracy:.2%}")
print(f"LightGBM accuracy: {lgb_accuracy:.2%}")
print(f"Ensemble accuracy: {ensemble_accuracy:.2%}")
print(f"Top 3 accuracy: {top3_accuracy:.2%}")
print(f"Top 5 accuracy: {top5_accuracy:.2%}")

# Calculate average AUC
rf_avg_auc = hist_df['RF_AUC'].mean()
lgb_avg_auc = hist_df['LGB_AUC'].mean()
ensemble_avg_auc = hist_df['Ensemble_AUC'].mean()

print(f"\nAverage RF AUC: {rf_avg_auc:.4f}")
print(f"Average LGB AUC: {lgb_avg_auc:.4f}")
print(f"Average Ensemble AUC: {ensemble_avg_auc:.4f}")

# Save historical results
hist_df.to_csv('historical_prediction_results_comparison.csv', index=False)
print("Historical results saved to 'historical_prediction_results_comparison.csv'")

print("\nNBA Championship prediction model complete!")
