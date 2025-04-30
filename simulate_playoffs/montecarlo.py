import pandas as pd
import numpy as np
'''
from rf_game_by_game import randomforrest_gbg as rf
from xbg_game_by_game import predict_xgb as xbg
'''
import random 
import random
import matplotlib.pyplot as plt

'''
This will be a Monte Carlo Simulation for the rf game by game

'''
def Monte_Carlo_Sim(num_trials,input_variables,model):
    outputs = []
    for _ in range(num_trials):
        # Generate random inputs for this trial
        inputs = {}
        for var_name, var_range in input_variables.items():
            if isinstance(var_range, (list, tuple)):
                # If range is given, sample from uniform distribution
                inputs[var_name] = random.uniform(var_range[0], var_range[1])
            else:
                 # If single value given, use it directly
                inputs[var_name] = var_range
        # Run the simulation
        output = model(**inputs)
        outputs.append(output)
    return outputs

def round1_matchups():
    matchups = []
    return matchups


def playoff_sim():
    pass


if __name__ == '__main__':
    # Example: Simulate the sum of two dice rolls
    def dice_roll_model(num_dice=2):
        return sum(random.randint(1, 6) for _ in range(num_dice))

    input_vars = {'num_dice': 2}
    num_simulations = 10000

    results = Monte_Carlo_Sim(num_simulations, input_vars, dice_roll_model)

    # Analyze and visualize the results
    print(f"Mean sum: {np.mean(results):.2f}")
    print(f"Standard deviation: {np.std(results):.2f}")

    plt.hist(results, bins=11, range=(2,12), edgecolor='black')
    plt.title('Monte Carlo Simulation of Dice Rolls')
    plt.xlabel('Sum of Dice')
    plt.ylabel('Frequency')
    plt.show()

