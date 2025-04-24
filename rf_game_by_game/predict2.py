import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings

np.random.seed(43)
print("loading that dih...")
nba_df = pd.read_csv("NBA 2001 Every Playoff Game.csv")

print("here are the first few rows of data ya bish")
print(nba_df.head())

missing_values = nba_df.isnull().sum()
print(missing_values[missing_values > 0])


features = [
    'Visitor/Neutral','PTS','Win','Home/Neutral','PTS','Win','Visitor Seed','Home Seed','Year'
]