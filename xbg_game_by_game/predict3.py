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

# Handle missing values in is_champion for 2025 season
df.loc[df['season_end'] == 2025, 'is_champion'] = False

# Ensure is_champion is boolean
if df['is_champion'].dtype != 'bool':
    df['is_champion'] = df['is_champion'].astype(bool)

# Drop columns that are not needed
data = df.drop(columns=['points'])







