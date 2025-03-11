# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_state(obs):
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    return (taxi_row, taxi_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

# Global variable to store the Q-table.
q_table = None

def load_q_table():
    global q_table
    if q_table is None:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    return q_table

def get_action(obs):
    """
    Given an observation, use the loaded Q-table to select an action.
    If the observation key is missing, choose a random action as a fallback.
    """
    # Load Q-table if not already loaded.
    qt = load_q_table()
    
    # Convert the observation to a unique state key.
    state_key = get_state(obs)
    
    # Check if the state key exists in the Q-table.
    if state_key in qt:
        # Exploit: select the action with the highest Q-value.
        action = int(np.argmax(np.array(q_table[state_key])))
    else:
        # Fallback: choose a random action if the state is not present.
        action = random.choice([0, 1, 2, 3, 4, 5])
    
    return action
