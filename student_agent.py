# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch

def get_action(obs):
    action = [0,1, 2, 3, 4, 5]
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    if obs[0]==0 or obs[10]==1:
        action.remove(1)  
    if obs[1]==0 or obs[12]==1:
        action.remove(3)  
    if obs[13]==1:
        action.remove(2)
    if obs[11]==1:
        action.remove(0)
    if obs[14]!=1:
        action.remove(4)
    if obs[15]!=1:
        action.remove(5)

    if action==[]:
        print(obs)
            
    return random.choice(action) # Choose a random action

    #return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

'''state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

 if action == 0 :  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1'''