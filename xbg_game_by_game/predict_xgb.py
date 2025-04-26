import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
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

# Hyperparameter tuning
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Perform hyperparameter tuning
grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=43),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Cross validation
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_scores.mean())

# Create fine-tuned model with best parameters
model2 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=43,
    **grid_search.best_params_
)

model2.fit(X_train_resampled, y_train_resampled)
print("Train Accuracy with tuned model:", model2.score(X_train_resampled, y_train_resampled))
y_pred2 = model2.predict(X_test)
print(f'Test Accuracy with tuned model: {accuracy_score(y_test, y_pred2):.5f}')

# Make prediction with tuned model
predicted_outcome2 = model2.predict(last_game)
print(predicted_outcome2)
print("Predicted Home Win (tuned model):" if predicted_outcome2[0] == 1 else "Predicted Home Loss 😢 (tuned model)")

'''
# Feature importance
print("\nFeature Importance:")
feature_importance = model2.feature_importances_
feature_names = X.columns
sorted_idx = feature_importance.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.savefig('xgb_feature_importance.png')
print("Feature importance chart saved as 'xgb_feature_importance.png'")
'''





