
'''
This Random Forrest Model file will use the predict_proba package (i think its a package)
'''
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

def train_model(tune=False):
    nba_df = pd.read_csv("NBA_2001_2005_All team stats_filled.csv")

    print("Original column names:", nba_df.columns.tolist())
    print("Here are the first few rows of data ya bish")
    print(nba_df.head())

    missing_values = nba_df.isnull().sum()
    print("Missing values:")
    print(missing_values[missing_values > 0])

    nba_df.drop(['PTS', 'PTS.1'], axis=1, inplace=True)
    nba_df["Vis_Win"] = nba_df["Win"]
    nba_df["Home_Win"] = nba_df["Win.1"]
    nba_df.drop(['Win','Win.1',"Vis_Win","Visitor/Neutral","Home/Neutral", 'Visitor_id','Year','Home_id'], axis=1, inplace=True)

    if nba_df["Home_Win"].dtype != "bool":
        nba_df["Home_Win"] = nba_df["Home_Win"].astype(bool)
    nba_df["Home_Win"] = nba_df["Home_Win"].map({True: 1, False: 0})

    print(nba_df[pd.isnull(nba_df["Home_Win"])])
    print(nba_df.head())

    last_game = nba_df.iloc[-1]
    nba_df = nba_df.iloc[:-2]

    X = nba_df.drop(columns=['Home_Win'])
    y = nba_df['Home_Win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    smote = SMOTE(random_state=43)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train basic model first
    base_model = RandomForestClassifier(random_state=43)
    base_model.fit(X_train_resampled, y_train_resampled)

    print("🔵 Base Model Train Accuracy:", base_model.score(X_train_resampled, y_train_resampled))
    base_pred_probs = base_model.predict_proba(X_test)
    base_pred_labels = np.argmax(base_pred_probs, axis=1)
    print(f"🔵 Base Model Test Accuracy: {accuracy_score(y_test, base_pred_labels):.5f}")

    # Now optionally tune - flip the swtich to true or false
    if tune:
        best_params = tune_model(X_train_resampled, y_train_resampled)
        tuned_model = RandomForestClassifier(**best_params, random_state=43)
        tuned_model.fit(X_train_resampled, y_train_resampled)

        print("🟢 Tuned Model Train Accuracy:", tuned_model.score(X_train_resampled, y_train_resampled))
        tuned_pred = tuned_model.predict_proba(X_test)
        print(f"🟢 Tuned Model Test Accuracy: {accuracy_score(y_test, tuned_pred):.5f}")

        return tuned_model, last_game
    else:
        return base_model, last_game


def predict_game(model, game_features):
    game_df = pd.DataFrame([game_features])
    prob_visitor, prob_home = model.predict_proba(game_df)[0]
    return prob_home, prob_visitor

def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': [0, 0.01, 0.1]
    }
    
    base_model = RandomForestClassifier(random_state=43)

    print("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("✅ Best Parameters:", grid_search.best_params_)
    print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.5f}")

    return grid_search.best_params_

if __name__ == '__main__':
    model, last_game = train_model(tune=False)  # <--- toggle True or False here
    
    dummy_game = {
        'Visitor Seed': 6,
        'Home Seed': 3,
        'Visitor win_pct': 0.70,
        'Visitor off_rtg': 113.2,
        'Visitor def_rtg': 110.5,
        'Visitor net_rtg': 2.7,
        'Visitor pace': 100.4,
        'Visitor efg_pct': 0.540,
        'Visitor ts_pct': 0.575,
        'Visitor tov_pct': 13.4,
        'Visitor orb_pct': 25.8,
        'Visitor drb_pct': 74.3,
        'Home win_pct': 0.640,
        'Home off_rtg': 115.5,
        'Home def_rtg': 109.0,
        'Home net_rtg': 6.5,
        'Home pace': 100.8,
        'Home efg_pct': 0.550,
        'Home ts_pct': 0.580,
        'Home tov_pct': 12.9,
        'Home orb_pct': 27.0,
        'Home drb_pct': 75.5,
    }

    home_prob, visitor_prob = predict_game(model, dummy_game)
    print(f"Home win chance: {home_prob:.2%} | Visitor win chance: {visitor_prob:.2%}")


