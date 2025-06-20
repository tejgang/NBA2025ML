# 🏀 NBA Finals Prediction Simulator

This project uses a machine learning model and Monte Carlo simulation to estimate the probability that the **Oklahoma City Thunder** will win the **2025 NBA Finals** against the **Indiana Pacers**. It relies on team-level playoff performance metrics and simulates a best-of-7 series based on learned win probabilities.

## 📁 Notebook Overview

### 1. **Data Preparation**
- Loads a CSV containing team-level performance statistics from each playoff series.
- Filters and organizes data to focus on relevant features such as offensive rating, defensive rating, net rating, win percentage, and pace.

### 2. **Model Training**
- Trains a `RandomForestClassifier` using historical Finals series data.
- Uses performance differentials (e.g. difference in net rating, win %, etc.) as input features.
- Splits the data into train/test sets and evaluates model performance via accuracy.

### 3. **Finals Matchup Setup**
- Manually inputs team statistics for the 2025 Finals matchup: OKC vs. Indiana.
- Computes the feature differentials between the two teams to serve as input for the trained model.

### 4. **Game-by-Game Probability Estimation**
- Predicts the win probability for OKC for each of the 7 possible Finals games using `model.predict_proba`.
- These predictions assume static team metrics and game independence.

### 5. **Monte Carlo Simulation**
- Simulates 10,000 best-of-7 Finals series using the game-by-game win probabilities.
- For each simulation, randomly determines outcomes of individual games.
- Outputs the overall probability that OKC wins the series.

## ✅ Key Result
> The final output is a single estimated probability representing the Thunder’s chance of winning the 2025 NBA Finals.