import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

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

# rename columns
df["Vis_Win"] = df["Win"]
df["Home_Win"] = df["Win.1"]

# drop columns we don't need
df.drop(['PTS','PTS.1','Win','Win.1',"Vis_Win","Visitor/Neutral","Home/Neutral", 'Visitor_id','Year','Home_id'],axis=1,inplace=True)
print(df.head())
if df["Home_Win"].dtype != "bool": #make home_win boolean
    df["Home_Win"] = df["Home_Win"].astype(bool)
df["Home_Win"] = df["Home_Win"].map({True: 1, False: 0}) #make home win mapped to 1 for True, 0 for false

# Show any unmapped values
print(df[pd.isnull(df["Home_Win"])])
print(df.head())

# select last row for game we want to predict
last_game = df.iloc[-1] #select last row for game we want to predict
df = df.iloc[0:-2] #select everything else for training and testing
X = df.drop(columns=['Home_Win'])
y = df['Home_Win']

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
smote = SMOTE(random_state=43) #adds random for x and y for balance
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=43
)

# Train the model
model.fit(X_train, y_train)
print("Train Accuracy:", model.score(X_train, y_train))
y_pred = model.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.5f}')


# Make Prediction
last_game = last_game.drop(['Home_Win']).to_frame().T
predicted_outcome = model.predict(last_game)
print(predicted_outcome)
print("Predicted Home Win:" if predicted_outcome[0] == 1 else "Predicted Home Loss 😢")






