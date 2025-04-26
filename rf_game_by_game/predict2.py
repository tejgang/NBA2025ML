import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# set random seed
np.random.seed(43)
print("loading that dih...")

# load data with limited number of columns to avoid the trailing commas
nba_df = pd.read_csv("NBA_2001_2005_All team stats_filled.csv")

# Print column names to verify
print("Original column names:", nba_df.columns.tolist())

# print first few rows of data
print("here are the first few rows of data ya bish")
print(nba_df.head())

# print missing values
missing_values = nba_df.isnull().sum()
print("Missing values:")
print(missing_values[missing_values > 0])

# drop columns with missing values
nba_df.drop(['PTS', 'PTS.1'], axis=1, inplace=True)

# rename columns
nba_df["Vis_Win"] = nba_df["Win"]
nba_df["Home_Win"] = nba_df["Win.1"]

# drop columns we don't need
nba_df.drop(['Win','Win.1',"Vis_Win","Visitor/Neutral","Home/Neutral", 'Visitor_id','Year','Home_id'],axis=1,inplace=True)
if nba_df["Home_Win"].dtype != "bool": #make home_win boolean
    nba_df["Home_Win"] = nba_df["Home_Win"].astype(bool)
nba_df["Home_Win"] = nba_df["Home_Win"].map({True: 1, False: 0}) #make home win mapped to 1 for True, 0 for false

# Show any unmapped values
print(nba_df[pd.isnull(nba_df["Home_Win"])])
print(nba_df.head())

# select last row for game we want to predict
last_game = nba_df.iloc[-20] #select last row for game we want to predict
nba_df = nba_df.iloc[0:-21] #select everything else for training and testing
X = nba_df.drop(columns=['Home_Win'])
y = nba_df['Home_Win']

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
smote = SMOTE(random_state=43) #adds random for x and y for balance
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# train model
model = RandomForestClassifier(random_state=43)
model.fit(X_train, y_train)
print("Train Accuracy:", model.score(X_train, y_train))
y_pred = model.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.5f}')


# Make Prediction
last_game = last_game.drop(['Home_Win']).to_frame().T
predicted_outcome = model.predict(last_game)
print(predicted_outcome)
print("Predicted Home Win:" if predicted_outcome[0] == 1 else "Predicted Home Loss 😢")

# hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0, 0.01, 0.1]
}

# perform hyperparameter tuning
print("hyperparameter tuning")
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# cross validation
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_scores.mean())

# fine tun model
model2 = RandomForestClassifier(
    n_estimators=75, 
    max_depth=7, 
    min_samples_split=6, 
    min_samples_leaf=7,
    ccp_alpha=0,
    max_features='sqrt',
    random_state=43)

model2.fit(X_train_resampled, y_train_resampled)
print("Train Accuracy:", model2.score(X_train_resampled, y_train_resampled))
y_pred2 = model2.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred2):.5f}')

# make prediction
predicted_outcome2 = model2.predict(last_game)
print(predicted_outcome2)
print("Predicted Home Win:" if predicted_outcome2[0] == 1 else "Predicted Home Loss 😢")
