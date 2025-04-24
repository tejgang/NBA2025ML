import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

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