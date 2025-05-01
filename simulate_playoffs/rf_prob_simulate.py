import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import sys

def load_and_prepare_data():
    """
    Load historical playoff game data and prepare features
    """
    try:
        print("Loading historical playoff data...")
        nba_df = pd.read_csv("NBA_2001_2025_All_team_stats_filled.csv")
        
        # Print column names for debugging
        print("Available columns:", nba_df.columns.tolist())
        
        # Drop unused columns and rename for consistency
        if 'PTS' in nba_df.columns and 'PTS.1' in nba_df.columns:
            nba_df.drop(['PTS', 'PTS.1'], axis=1, inplace=True)
            
        # Process win columns
        if 'Win' in nba_df.columns and 'Win.1' in nba_df.columns:
            nba_df["HomeWin"] = nba_df["Win.1"].astype(int)
            nba_df.drop(['Win', 'Win.1'], axis=1, inplace=True)
        
        # Drop team identification columns that aren't needed for modeling
        cols_to_drop = [col for col in ['Visitor/Neutral', 'Home/Neutral', 'Visitor_id', 'Home_id', 'Year'] 
                        if col in nba_df.columns]
        nba_df.drop(cols_to_drop, axis=1, inplace=True)
        
        print("Data loaded successfully!")
        print(f"Dataset shape: {nba_df.shape}")
        
        return nba_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_model(df):
    """
    Train a Random Forest model on historical playoff data
    """
    try:
        print("\nTraining model on historical playoff data...")
        
        # Define features for prediction
        if 'HomeWin' not in df.columns:
            raise ValueError("HomeWin column missing from dataset")
            
        # Select all available features except the target
        X = df.drop(columns=['HomeWin'])
        y = df['HomeWin']
        
        print(f"Features: {X.columns.tolist()}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
        
        # Fit a Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=43)
        model.fit(X_train, y_train)
        
        # Report accuracy
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def create_2025_playoff_series(csv_path="NBA_2025_Playoff_Series.csv"):
    """
    Load 2025 playoff series data from CSV
    """
    try:
        print(f"Loading playoff series data from {csv_path}...")
        series_df = pd.read_csv(csv_path)
        
        # Print column names for debugging
        print("Playoff series columns:", series_df.columns.tolist())
        
        # Extract team names and seeds into the format needed for simulation
        playoff_data = []
        
        # Process first round matchups (the ones with data)
        for _, row in series_df.iterrows():
            # Skip empty rows (later rounds that will be filled in)
            if pd.isna(row['Visitor']) or pd.isna(row['Home']):
                continue
                
            # Create series entry with basic data
            series_entry = {
                "SeriesID": row['Series'],
                "Round": 1 if row['Series'].startswith('rd1') else (
                         2 if row['Series'].startswith('rd2') else (
                         3 if row['Series'].startswith('rd3') else 4)),
                "Home": row['Home'],
                "Visitor": row['Visitor'],
                "HomeSeed": int(row['Home seed']),
                "VisitorSeed": int(row['Visitor seed']),
            }
            
            # Add all available metrics with correct naming for the simulation
            for col in row.index:
                # Skip non-metric columns
                if col in ['Series', 'Visitor', 'Home', 'Visitor seed', 'Home seed']:
                    continue
                
                # If it's a visitor metric, add it directly
                if col.startswith('Visitor '):
                    metric_name = col  # Keep the full original column name
                    if not pd.isna(row[col]):
                        series_entry[metric_name] = row[col]
                
                # If it's a home metric, add it directly
                elif col.startswith('Home '):
                    metric_name = col  # Keep the full original column name
                    if not pd.isna(row[col]):
                        series_entry[metric_name] = row[col]
            
            playoff_data.append(series_entry)
        
        print(f"Loaded {len(playoff_data)} playoff matchups")
        
        # If no data was found, return a message
        if len(playoff_data) == 0:
            print("Warning: No playoff data found in CSV!")
            return pd.DataFrame()
            
        return pd.DataFrame(playoff_data)
    except Exception as e:
        print(f"Error creating playoff series: {e}")
        print(f"Detailed error: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def simulate_series(row, model, feature_map=None):
    """
    Simulate a best-of-7 playoff series
    """
    wins = {"home": 0, "visitor": 0}
    
    # Default to using just the seed values
    feat_cols = ["HomeSeed", "VisitorSeed"]
    
    # If we have a feature map, use it to ensure we're selecting the right columns
    if feature_map:
        # Create feature list using the input columns
        extended_features = []
        for hist_col, series_col in feature_map.items():
            if series_col in row:
                # Train the transform from CSV names to model feature names 
                extended_features.append(hist_col)
        
        # Add these to the feature columns if available
        if extended_features:
            feat_cols = extended_features + feat_cols
    
    # Play games until someone gets 4 wins
    while wins["home"] < 4 and wins["visitor"] < 4:
        try:
            # Create the feature vector with appropriate mapping
            game_features = []
            for col in feat_cols:
                if col in row:
                    game_features.append(row[col])
                elif feature_map and col in feature_map:
                    # If we have a mapping, use it
                    series_col = feature_map[col]
                    if series_col in row:
                        game_features.append(row[series_col])
                    else:
                        # If we don't find the feature, use a default
                        if "Seed" in col:
                            game_features.append(8 if "Visitor" in col else 1)  # Default seeds
                        else:
                            game_features.append(0)  # Default for other metrics
                else:
                    # Default case
                    game_features.append(0)
            
            Xg = np.array(game_features).reshape(1, -1)
            if Xg.shape[1] != len(feat_cols):
                # If dimensions don't match, fall back to 50/50
                if np.random.rand() < 0.5:
                    wins["home"] += 1
                else:
                    wins["visitor"] += 1
                continue
                
            p_home = model.predict_proba(Xg)[0, 1]
            if np.random.rand() < p_home:
                wins["home"] += 1
            else:
                wins["visitor"] += 1
        except Exception as e:
            print(f"Error in simulate_series: {e}")
            # Fallback to 50/50 chance if prediction fails
            if np.random.rand() < 0.5:
                wins["home"] += 1
            else:
                wins["visitor"] += 1
    
    # Return a dict representing the series winner
    if wins["home"] > wins["visitor"]:
        side = "Home"
    else:
        side = "Visitor"
    
    result = {
        "Team": row[side],
        "Seed": row[f"{side}Seed"],
    }
    
    # Add metrics if available
    for col in row.keys():
        if col.startswith(f"{side} ") and col not in [side, f"{side}Seed"]:
            result[col] = row[col]
    
    return result

def build_next_round(winners):
    """
    Pair winners into the next round
    """
    next_rows = []
    # Winners is a list of dicts in bracket order
    for i in range(0, len(winners), 2):
        # Make sure we have a pair to match
        if i + 1 >= len(winners):
            break
            
        a, b = winners[i], winners[i + 1]
        # Lower seed number = better seed → home court
        if a["Seed"] < b["Seed"]:
            home, visitor = a, b
        else:
            home, visitor = b, a
            
        # Create the matchup for the next round
        matchup = {
            "HomeSeed": home["Seed"],
            "VisitorSeed": visitor["Seed"],
            "Home": home["Team"],
            "Visitor": visitor["Team"]
        }
        
        # Add metrics to the matchup if they exist
        for key in home:
            if key not in ["Team", "Seed"]:
                if key.startswith("Home "):
                    matchup[key] = home[key]
                elif key.startswith("Visitor "):
                    new_key = key.replace("Visitor ", "Home ")
                    matchup[new_key] = home[key]
                    
        for key in visitor:
            if key not in ["Team", "Seed"]:
                if key.startswith("Visitor "):
                    matchup[key] = visitor[key]
                elif key.startswith("Home "):
                    new_key = key.replace("Home ", "Visitor ")
                    matchup[new_key] = visitor[key]
        
        next_rows.append(matchup)
    
    return next_rows

def simulate_bracket(series_df, model, feature_map=None):
    """
    Simulate an entire playoff bracket
    """
    # Get first round matchups
    first_round = series_df.copy()
    
    # Dictionary to store results
    results = {
        "round1_winners": [],
        "round2_winners": [],
        "conference_finals_winners": [],
        "champion": None
    }
    
    # Simulate first round
    for _, row in first_round.iterrows():
        if pd.isna(row['Visitor']) or pd.isna(row['Home']):
            continue
        winner = simulate_series(row, model, feature_map)
        results["round1_winners"].append(winner)
    
    # Build and simulate second round
    r2_matchups = build_next_round(results["round1_winners"])
    for matchup in r2_matchups:
        winner = simulate_series(matchup, model, feature_map)
        results["round2_winners"].append(winner)
    
    # Build and simulate conference finals
    cf_matchups = build_next_round(results["round2_winners"])
    for matchup in cf_matchups:
        winner = simulate_series(matchup, model, feature_map)
        results["conference_finals_winners"].append(winner)
    
    # Simulate the Finals
    finals = build_next_round(results["conference_finals_winners"])
    if finals:
        results["champion"] = simulate_series(finals[0], model, feature_map)
    
    return results

def monte_carlo(model, series_df, feature_map=None, N=100):
    """
    Run multiple simulations and return probabilities
    """
    # Counter for champions
    champ_counter = Counter()
    
    # Run N simulations
    for i in range(N):
        if i % 10 == 0:  # Print progress every 10 iterations
            print(f"Simulation {i}/{N}", end="\r")
        
        # Run the bracket simulation
        results = simulate_bracket(series_df, model, feature_map)
        
        # Count the champion
        if results["champion"]:
            champ_counter[results["champion"]["Team"]] += 1
    
    # Calculate probabilities
    total = sum(champ_counter.values())
    probabilities = {team: count / total for team, count in champ_counter.items()}
    
    # Sort by probability (high to low)
    sorted_probs = {k: v for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)}
    
    return sorted_probs

def plot_championship_probabilities(probs):
    """
    Plot the championship probabilities
    """
    teams = list(probs.keys())
    probs_values = list(probs.values())
    
    # Sort by probability in descending order
    sorted_indices = np.argsort(probs_values)[::-1]
    teams = [teams[i] for i in sorted_indices]
    probs_values = [probs_values[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(teams, probs_values, color='navy')
    plt.xlabel('Championship Probability')
    plt.title('2025 NBA Championship Probabilities')
    plt.xlim(0, max(probs_values) * 1.1)  # Add some space on the right
    
    # Add probability labels
    for i, v in enumerate(probs_values):
        plt.text(v + 0.01, i, f'{v:.1%}', va='center')
    
    plt.tight_layout()
    plt.savefig('championship_probabilities.png')
    plt.close()
    
    print(f"Plot saved as 'championship_probabilities.png'")

def map_features(historical_df, series_df):
    """
    Create a mapping between historical feature names and 2025 series feature names
    """
    feature_map = {}
    
    # Get historical column names
    hist_cols = set(historical_df.columns)
    
    # Get series column names
    series_cols = set()
    for col in series_df.columns:
        series_cols.add(col)
    
    print("Series columns:", series_cols)
    
    # Map Home and Visitor metrics
    for hist_col in hist_cols:
        if hist_col.startswith('Home_'):
            # Change from Home_metric to Home metric (with space)
            new_col = 'Home ' + hist_col[5:]
            if new_col in series_cols:
                feature_map[hist_col] = new_col
        elif hist_col.startswith('Visitor_'):
            # Change from Visitor_metric to Visitor metric (with space)
            new_col = 'Visitor ' + hist_col[8:]
            if new_col in series_cols:
                feature_map[hist_col] = new_col
    
    # Add seed mappings
    if 'HomeSeed' in hist_cols and 'Home seed' in series_cols:
        feature_map['HomeSeed'] = 'Home seed'
    if 'VisitorSeed' in hist_cols and 'Visitor seed' in series_cols:
        feature_map['VisitorSeed'] = 'Visitor seed'
    
    print(f"Created feature map with {len(feature_map)} mappings")
    return feature_map

def main():
    # 1. Load and prepare historical playoff data
    historical_df = load_and_prepare_data()
    if historical_df is None:
        print("Failed to load historical data. Exiting.")
        sys.exit(1)
    
    # 2. Train model on historical data
    model = train_model(historical_df)
    if model is None:
        print("Failed to train model. Exiting.")
        sys.exit(1)
    
    # 3. Create 2025 playoff bracket
    series_df = create_2025_playoff_series()
    if series_df.empty:
        print("Failed to create playoff bracket. Exiting.")
        sys.exit(1)
    
    # 4. Create feature mapping from historical to 2025 data
    feature_map = map_features(historical_df, series_df)
    
    # 5. Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation (100 iterations)...")
    championship_probs = monte_carlo(model, series_df, feature_map, N=100)
    
    # 6. Print championship probabilities
    print("\nChampionship Probabilities:")
    for team, prob in championship_probs.items():
        print(f"{team}: {prob:.1%}")
    
    # 7. Plot results
    try:
        plot_championship_probabilities(championship_probs)
    except Exception as e:
        print(f"Error plotting results: {e}")

if __name__ == "__main__":
    main()
