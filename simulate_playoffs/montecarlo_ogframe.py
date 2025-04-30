import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from rf_game_by_game.randomforrest_gbg import train_model, predict_game

# === Series sim (best of 7) ===
def series_sim(team1, team2, team1_stats, team2_stats, model):
    team1_wins = 0
    team2_wins = 0

    while team1_wins < 4 and team2_wins < 4:
        winner = model(team1_stats, team2_stats)
        if winner == 1:
            team1_wins += 1
        else:
            team2_wins += 1
    return team1 if team1_wins == 4 else team2

# === Simulate 3 playoff rounds in a conference ===
def simulate_conference_playoffs(matchups, model):
    current_round = matchups

    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            t1, t2, s1, s2 = current_round[i]
            t3, t4, s3, s4 = current_round[i+1]

            winner1 = series_sim(t1, t2, s1, s2, model)
            winner2 = series_sim(t3, t4, s3, s4, model)

            stats1 = s1 if winner1 == t1 else s2
            stats2 = s3 if winner2 == t3 else s4

            next_round.append((winner1, winner2, stats1, stats2))

        current_round = next_round

    final_match = current_round[0]
    champ = series_sim(final_match[0], final_match[1], final_match[2], final_match[3], model)
    champ_stats = final_match[2] if champ == final_match[0] else final_match[3]
    return champ, champ_stats

# === Simulate full NBA playoffs ===
def simulate_full_playoffs(model, east_matchups, west_matchups):
    east_champ, east_stats = simulate_conference_playoffs(east_matchups, model)
    west_champ, west_stats = simulate_conference_playoffs(west_matchups, model)
    nba_champ = series_sim(east_champ, west_champ, east_stats, west_stats, model)
    return nba_champ

# === Monte Carlo Simulator ===
def monte_carlo_sim(num_trials, model, east_matchups, west_matchups):
    results = []
    for _ in range(num_trials):
        champ = simulate_full_playoffs(model, east_matchups, west_matchups)
        results.append(champ)
    return results

# === Wrapper for your RF model ===
def rf_model(team1_stats, team2_stats):
    game_features = {
        'Visitor Seed': team2_stats[0],
        'Home Seed': team1_stats[0],
        'Visitor win_pct': team2_stats[1],
        'Visitor off_rtg': team2_stats[2],
        'Visitor def_rtg': team2_stats[3],
        'Visitor net_rtg': team2_stats[4],
        'Visitor pace': team2_stats[5],
        'Visitor efg_pct': team2_stats[6],
        'Visitor ts_pct': team2_stats[7],
        'Visitor tov_pct': team2_stats[8],
        'Visitor orb_pct': team2_stats[9],
        'Visitor drb_pct': team2_stats[10],
        'Home win_pct': team1_stats[1],
        'Home off_rtg': team1_stats[2],
        'Home def_rtg': team1_stats[3],
        'Home net_rtg': team1_stats[4],
        'Home pace': team1_stats[5],
        'Home efg_pct': team1_stats[6],
        'Home ts_pct': team1_stats[7],
        'Home tov_pct': team1_stats[8],
        'Home orb_pct': team1_stats[9],
        'Home drb_pct': team1_stats[10],
    }
    return predict_game(rf_model.model, game_features)

if __name__ == '__main__':
    rf_model.model, _ = train_model(tune=False)
#need to fix these
    team_stats = {
        'BOS': [1, 0.75, 118.2, 110.3, 7.9, 98.6, 0.56, 0.59, 12.1, 28.5, 74.4],
        'MIA': [8, 0.52, 111.8, 112.5, -0.7, 96.3, 0.51, 0.55, 13.4, 26.7, 73.1],
        'CLE': [4, 0.63, 114.0, 108.2, 5.8, 97.0, 0.54, 0.57, 12.9, 27.8, 73.9],
        'ORL': [5, 0.60, 112.0, 110.0, 2.0, 96.5, 0.52, 0.56, 13.1, 26.0, 72.8],
        'MIL': [3, 0.68, 117.5, 109.5, 8.0, 99.1, 0.55, 0.58, 12.5, 27.2, 75.0],
        'IND': [6, 0.58, 113.2, 112.0, 1.2, 97.8, 0.53, 0.57, 13.0, 25.6, 72.5],
        'NYK': [2, 0.70, 116.5, 108.8, 7.7, 97.2, 0.54, 0.58, 12.7, 27.0, 74.0],
        'PHI': [7, 0.55, 111.0, 110.5, 0.5, 95.9, 0.51, 0.55, 13.3, 26.3, 73.4],
        'DEN': [1, 0.72, 117.8, 109.0, 8.8, 98.0, 0.56, 0.59, 12.4, 27.5, 75.2],
        'LAL': [8, 0.51, 111.2, 113.1, -1.9, 96.7, 0.50, 0.54, 13.6, 25.8, 72.6],
        'MIN': [4, 0.64, 114.5, 108.0, 6.5, 97.5, 0.53, 0.57, 13.0, 26.5, 74.2],
        'PHX': [5, 0.61, 112.9, 110.7, 2.2, 96.2, 0.52, 0.56, 13.2, 25.9, 73.0],
        'OKC': [2, 0.69, 116.9, 109.2, 7.7, 98.2, 0.55, 0.58, 12.8, 27.1, 74.6],
        'NOP': [7, 0.54, 110.7, 111.5, -0.8, 95.7, 0.50, 0.54, 13.5, 25.4, 72.0],
        'DAL': [3, 0.66, 115.0, 109.8, 5.2, 97.6, 0.53, 0.56, 13.1, 26.2, 73.8],
        'LAC': [6, 0.57, 113.0, 111.0, 2.0, 96.8, 0.52, 0.55, 13.3, 25.7, 73.2]
    }

    east_matchups = [
        ('BOS', 'MIA', team_stats['BOS'], team_stats['MIA']),
        ('CLE', 'ORL', team_stats['CLE'], team_stats['ORL']),
        ('MIL', 'IND', team_stats['MIL'], team_stats['IND']),
        ('NYK', 'PHI', team_stats['NYK'], team_stats['PHI']),
    ]

    west_matchups = [
        ('DEN', 'LAL', team_stats['DEN'], team_stats['LAL']),
        ('MIN', 'PHX', team_stats['MIN'], team_stats['PHX']),
        ('OKC', 'NOP', team_stats['OKC'], team_stats['NOP']),
        ('DAL', 'LAC', team_stats['DAL'], team_stats['LAC']),
    ]

    num_simulations = 1000
    results = monte_carlo_sim(num_simulations, rf_model, east_matchups, west_matchups)

    counts = Counter(results)
    print("Championship wins per team:")
    for team, count in counts.most_common():
        print(f"{team}: {count} wins ({count/num_simulations:.2%})")

    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=45)
    plt.title("NBA Championship Odds from Monte Carlo Simulation")
    plt.xlabel("Team")
    plt.ylabel("Championships Won")
    plt.tight_layout()
    plt.show()
