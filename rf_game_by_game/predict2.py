import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

np.random.seed(43)
print("loading that dih...")
nba_df = pd.read_csv("NBA 2001 Every Playoff Game.csv")

print("here are the first few rows of data ya bish")
print(nba_df.head())

missing_values = nba_df.isnull().sum()
print(missing_values[missing_values > 0])
nba_df.drop(['PTS', 'PTS.1', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'], axis=1, inplace=True)



nba_df["Vis_Win"] = nba_df["Win"]
nba_df["Home_Win"] = nba_df["Win.1"]
nba_df.drop(['Win','Win.1',"Vis_Win"],axis=1,inplace=True)
print(nba_df.head())

features = [
    'Home_Win','Visitor Seed','Home Seed']

df_train = nba_df[nba_df['season_end'] <= 2001].copy()
df_predict = nba_df[nba_df['season_end'] == 2025].copy()

X = df_train[features].copy()
y = df_train['Home_Win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = BalancedRandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Train Accuracy:", model.score(X_train, y_train))
y_pred = model.predict(X_test)
print(f'Test Accuracy: {roc_auc_score(y_test, y_pred):.5f}')

