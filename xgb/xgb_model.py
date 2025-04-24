import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading NBA team metrics data...")
df = pd.read_csv('nba_team_season_metrics_2001_2025.csv')

# Inspect the data
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Handle missing values in is_champion for 2025 season
df.loc[df['season_end'] == 2025, 'is_champion'] = False

# Ensure is_champion is boolean
if df['is_champion'].dtype != 'bool':
    df['is_champion'] = df['is_champion'].astype(bool)

# Create derived features
print("\nCreating enhanced features...")
df['offensive_efficiency'] = df['efg_pct'] * (1 - df['tov_pct'])
df['defensive_efficiency'] = (1 - df['orb_pct']) * df['drb_pct']
df['balance_score'] = np.abs(df['off_rtg'] - 110) + np.abs(df['def_rtg'] - 105)
df['reb_advantage'] = df['orb_pct'] - (1 - df['drb_pct'])
df['shooting_advantage'] = df['ts_pct'] - df['efg_pct']

# Create "championship DNA" feature - teams that won recently
df['recent_champion'] = False
for year in range(2005, 2025):
    recent_champions = df[(df['season_end'].between(year-3, year-1)) & (df['is_champion'])]['team'].unique()
    df.loc[(df['season_end'] == year) & (df['team'].isin(recent_champions)), 'recent_champion'] = True

# Create "hotness" indicator
df['hot_team'] = df['win_pct'] > (1 - df['def_rtg'] / df['off_rtg'] + 0.05)

# Split into training and prediction sets
df_train = df[df['season_end'] <= 2024].copy()
df_predict = df[df['season_end'] == 2025].copy()

print(f"Training data shape: {df_train.shape}")
print(f"Prediction data shape: {df_predict.shape}")

# Define features
features = [
    'win_pct',
    'off_rtg', 'def_rtg', 'net_rtg', 'pace',
    'efg_pct', 'ts_pct', 'tov_pct', 'orb_pct', 'drb_pct',
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

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 5, 10]  # To handle class imbalance
}

# Define time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform grid search with time series cross-validation
print("\nPerforming grid search for hyperparameter tuning...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid_search = GridSearchCV(
        xgb.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss'),  # Removed use_label_encoder
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_res, y_res)

# Print best parameters
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best accuracy score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    best_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',  # Removed use_label_encoder
        **grid_search.best_params_
    )
    best_model.fit(X_res, y_res)

# Evaluate model on training data
y_train_pred = best_model.predict(X)
train_accuracy = accuracy_score(y, y_train_pred)
print(f"\nTraining accuracy: {train_accuracy:.4f}")

# Perform historical validation
print("\nPerforming historical validation...")
historical_results = []

# For each season from 2010 to 2024, train on previous seasons and predict that season
for test_year in range(2010, 2025):
    print(f"Evaluating model for {test_year} season...")
    
    # Split data
    hist_train = df[(df['season_end'] < test_year) & (df['season_end'] >= 2001)]
    hist_test = df[df['season_end'] == test_year]
    
    # Skip if no champion in test year
    if not hist_test['is_champion'].any():
        print(f"No champion in {test_year}, skipping...")
        continue
    
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
    
    # Train model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',  # Removed use_label_encoder
            **grid_search.best_params_
        )
        hist_model.fit(X_hist_train_res, y_hist_train_res)
    
    # Predict
    probs = hist_model.predict_proba(X_hist_test)[:, 1]
    
    # Get actual champion
    actual_champion = hist_test[hist_test['is_champion'] == True]['team'].values[0]
    
    # Get predicted champion
    teams = hist_test['team'].values
    pred_idx = np.argmax(probs)
    predicted_champion = teams[pred_idx]
    
    # Calculate metrics
    champion_correct = actual_champion == predicted_champion
    
    # Get top 3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_teams = teams[top3_indices]
    actual_in_top3 = actual_champion in top3_teams
    
    # Get top 5 predictions
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_teams = teams[top5_indices]
    actual_in_top5 = actual_champion in top5_teams
    
    # Store results
    historical_results.append({
        'Season': test_year,
        'Actual_Champion': actual_champion,
        'Predicted_Champion': predicted_champion,
        'Correct': champion_correct,
        'Actual_In_Top3': actual_in_top3,
        'Actual_In_Top5': actual_in_top5,
        'Probability': probs[pred_idx],
        'Top3_Teams': ', '.join(top3_teams)
    })

# Create historical results dataframe
hist_df = pd.DataFrame(historical_results)
champion_accuracy = hist_df['Correct'].mean() if not hist_df.empty else 0
top3_accuracy = hist_df['Actual_In_Top3'].mean() if not hist_df.empty else 0
top5_accuracy = hist_df['Actual_In_Top5'].mean() if not hist_df.empty else 0

print("\nHistorical Prediction Results:")
print(hist_df[['Season', 'Actual_Champion', 'Predicted_Champion', 'Correct', 'Actual_In_Top3', 'Probability']])
print(f"\nChampion prediction accuracy: {champion_accuracy:.2%}")
print(f"Top 3 accuracy: {top3_accuracy:.2%}")
print(f"Top 5 accuracy: {top5_accuracy:.2%}")

# Save historical results
hist_df.to_csv('historical_prediction_results_xgb.csv', index=False)
print("Historical results saved to 'historical_prediction_results_xgb.csv'")

# Prepare prediction data
X_predict = df_predict[features].copy()
for col in bool_features:
    X_predict[col] = X_predict[col].astype(int)

# Make predictions for 2025
y_pred_proba = best_model.predict_proba(X_predict)[:, 1]
teams = df_predict['team'].values

# Create DataFrame with predictions
team_probs = pd.DataFrame({
    'Team': teams,
    'Championship Probability': y_pred_proba,
    'Win %': df_predict['win_pct'].values,
    'Net Rating': df_predict['net_rtg'].values
})
team_probs = team_probs.sort_values('Championship Probability', ascending=False).reset_index(drop=True)

# Display top 10 teams
print("\nTop 10 Teams Most Likely to Win the 2025 NBA Championship:")
print(team_probs.head(10))

# Plot top 10 teams
plt.figure(figsize=(12, 6))
sns.barplot(x='Championship Probability', y='Team', data=team_probs.head(10))
plt.title('Top 10 Teams Most Likely to Win the 2025 NBA Championship (XGBoost)')
plt.xlim(0, max(team_probs['Championship Probability']) * 1.1)
plt.tight_layout()
plt.savefig('championship_prediction_2025_xgb.png')
print("Prediction plot saved to 'championship_prediction_2025_xgb.png'")

# Feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(best_model, importance_type='weight', max_num_features=20)
plt.title('Feature Importance for NBA Championship Prediction (XGBoost)')
plt.tight_layout()
plt.savefig('feature_importance_xgb.png')
print("Feature importance plot saved to 'feature_importance_xgb.png'")

# Save predictions to CSV
team_probs.to_csv('championship_predictions_2025_xgb.csv', index=False)
print("Predictions saved to 'championship_predictions_2025_xgb.csv'")

print("\nXGBoost NBA Championship prediction model complete!")