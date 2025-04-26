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
df = pd.read_csv('NBA_2001_2005_All team stats_filled.csv')

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# print first few rows of data
print("here are the first few rows of data ya bish")
print(df.head())

# drop unneeded columns
df.drop(['PTS','PTS.1'],axis=1,inplace=True)
df.drop(['Visitor_id','Home_id'],axis=1,inplace=True)








