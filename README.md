# NBA Finals Prediction Simulator

This project uses a Random Forest model and Monte Carlo simulation to estimate the probability of whether the **Oklahoma City Thunder** or the **Indiana Pacers** will win the **2025 NBA Finals**. It relies on team-level season and playoff performance metrics and simulates a best-of-7 series based on learned win probabilities.

## Overview

### 1. **Data Preparation**
- We scraped Basketball Reference and NBA.com using `nba_api`, and merged all relevant statistics and metrics into a single csv: `NBA_2025_Playoff_Series_with_metrics.csv`
- Some of the metrics we scraped included: seed, net rating, win percentage, offensive rating, defensive rating, pace, etc.
- After loading the data into the notebook, we confirmed missing values, and created feature differences between Home and Visitor to train the model on.

### 2. **Exploratory Data Analysis**
- Visulized distributions between features and target using confusion matrices and histograms.
- Analyzed the correlation between features and target to identify relationships in the data.

### 3. **Model Training**
- We utilized a Random Forest model to train on whether a team wins based on the supplied feature differences.
- Splits the data into train/test sets and evaluates model performance via accuracy.
- Initial training and testing accuracy seen was 79.6% and 58.6% respectively, indicating overfitting.

### 4. **Model Tuning and Evaluation**
- Performed hyperparamter tuning using `GridSearchCV` to optimize model performance.
- Used cross-validation to assess the model's stability and generalizability.
- Evaluated the model's accuracy on both training and testing sets to be 71.1% and 62% respectively.

### 5. **Finals Matchup Setup**
- Inputed team statistics for the 2025 Finals matchup: OKC vs. Indiana.
- Computed the feature differentials between the two teams to serve as input for the trained model.

### 6. **Game-by-Game Probability Estimation**
- Predicts the win probability for OKC for each of the 7 possible Finals games using `model.predict_proba`.
- These predictions assume static team metrics and game independence.

### 7. **Monte Carlo Simulation**
- Simulates 10,000 best-of-7 Finals series using the game-by-game win probabilities.
- For each simulation, randomly determines outcomes of individual games.
- Outputs the overall probability that OKC wins the series.

### Results
Our model outputs that the Oklahoma City Thunder have a series win probability of 75.5%, while the Indiana Pacers have a win probability of 24.5%.
