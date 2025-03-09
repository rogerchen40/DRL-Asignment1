# custom_taxi_env.py
import gym
import numpy as np
import time
import random
# from IPython.display import clear_output

from xml.etree import ElementTree as ET
import importlib.util
import requests
import argparse

class DynamicTaxiEnv(gym.Wrapper):
    def __init__(self, grid_size=5, fuel_limit=50, randomize_passenger=True, randomize_destination=True):
        self.grid_size = grid_size
        env = gym.make("Taxi-v3", render_mode="ansi")
        super().__init__(env)
        
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.randomize_passenger = randomize_passenger
        self.randomize_destination = randomize_destination
        self.passenger_picked_up = False

        self.generate_random_map()

    def generate_random_map(self):
        """ 隨機產生站點與障礙物 """
        self.stations = random.sample([(x, y) for x in range(self.grid_size) for y in range(self.grid_size)], 4)
        self.obstacles = random.sample([(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.stations], int(self.grid_size**2 * 0.1))
        
        self.station_labels = ['R', 'G', 'Y', 'B']
        self.station_map = {self.station_labels[i]: self.stations[i] for i in range(4)}


    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_fuel = self.fuel_limit  # 重新初始化燃料
        self.passenger_picked_up = False
        self.generate_random_map()

        # 設定乘客位置
        if self.randomize_passenger:
            self.passenger_loc = random.choice(self.stations)
        else:
            self.passenger_loc = self.stations[0]

        # 設定目的地
        if self.randomize_destination:
            possible_destinations = [s for s in self.stations if s != self.passenger_loc]
            self.destination = random.choice(possible_destinations)
        else:
            self.destination = self.stations[1]

        # 取得 Agent 目前的位置
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        # 轉換 `pass_idx` 與 `dest_idx` 為 (x, y) 座標
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination

        # **加入四周障礙物資訊**
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        
        passenger_loc_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.passenger_loc)
        passenger_loc_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.passenger_loc)
        passenger_loc_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.passenger_loc)
        passenger_loc_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west
        
        # **回傳新的 state（不包含燃料）**
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1], destination_x, destination_y,
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look)

        return state, info

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(self.env.unwrapped.s)

        # **Determine next position BEFORE executing the move**
        next_row, next_col = taxi_row, taxi_col
        if action == 0 :  # Move South
            next_row += 1
        elif action == 1 :  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1

        # **Check if the move is invalid (hitting boundary or obstacle)**
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward = -5  # Penalty for hitting obstacle or boundary
                self.current_fuel -= 1
                if self.current_fuel <= 0:
                    return self.get_state(), reward -10, True, False, {}  # End game if out of fuel
                return self.get_state(), reward, False, False, {}

        # **✅ Update taxi position (Now taxi really moves)**
        taxi_row, taxi_col = next_row, next_col

        # **Execute the valid move**
        self.current_fuel -= 1  # Reduce fuel only for successful moves
        obs, reward, terminated, truncated, info = super().step(action)

        # **Modify reward system**
        if reward == 20:  # Successful `DROPOFF`
            reward = 50
        elif reward == -1:  # Regular movement penalty
            reward = -0.1
        elif reward == -10:  # Incorrect `PICKUP` or `DROPOFF`
            reward = -10

        # **🚀 Fix: Ensure `PICKUP` updates `self.passenger_loc`**
        if action == 4:  # PICKUP
            if pass_idx == 4:  # Passenger is inside the taxi
                self.passenger_picked_up = True  # Passenger is now inside the taxi
                self.passenger_loc = (taxi_row, taxi_col)  # ✅ Passenger moves with the taxi
            else:
                self.passenger_picked_up = False  # Pickup failed, Taxi-v3 already penalized it

        elif action == 5:  # DROPOFF
            if self.passenger_picked_up:  # **Only drop off if passenger was actually in the taxi**
                self.passenger_picked_up = False  # Passenger leaves the taxi
                self.passenger_loc = (taxi_row, taxi_col)

        # **✅ Ensure `self.passenger_loc` is always updated**
        if self.passenger_picked_up:
            self.passenger_loc = (taxi_row, taxi_col)  # ✅ Ensure passenger moves with taxi

        destination_x, destination_y = self.destination

        # **🚀 Ensure obstacle detection includes both borders & obstacles**
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        
        passenger_loc_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.passenger_loc)
        passenger_loc_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.passenger_loc)
        passenger_loc_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.passenger_loc)
        passenger_loc_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west
        
        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1], destination_x, destination_y,
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look)
        return state, reward, terminated, truncated, info





 

    def get_state(self):
        """ 取得當前 state (適用於不同 `n × n` 地圖) """
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(self.env.unwrapped.s)
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination
        # 計算障礙物資訊
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        passenger_loc_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.passenger_loc)
        passenger_loc_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.passenger_loc)
        passenger_loc_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.passenger_loc)
        passenger_loc_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west
        
        # **回傳新的 state（不包含燃料）**
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1], destination_x, destination_y,
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look)
        return state
    def render_env(self, taxi_pos):
        """ 顯示環境狀態，標示 Agent 位置 """
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        for label, pos in self.station_map.items():
            grid[pos[0]][pos[1]] = label
        
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        grid[self.passenger_loc[0]][self.passenger_loc[1]] = 'P'
        grid[self.destination[0]][self.destination[1]] = 'D'
        grid[taxi_pos[0]][taxi_pos[1]] = '🚖'

        for row in grid:
            print(" ".join(row))
        print("\n")

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = DynamicTaxiEnv(**env_config)
    obs, _ = env.reset()  
    total_reward = 0
    done = False
    step_count = 0

    while not done:

        if render:
            print(f"step={step_count}")
            # env.render_env((taxi_row, taxi_col))
            time.sleep(0.5)

        action = student_agent.get_action(obs)  # 確保 `get_action(obs)` 支援這個 state 格式
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

def parse_arguments():
    parser = argparse.ArgumentParser(description="HW1")

    parser.add_argument("--token", default="", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    # 讀取 meta.xml 內的 submission name
    xml_file_path = 'meta.xml'
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    sub_name = ""
    for book in root.findall('info'):
        sub_name = book.find('name').text

    ### 執行 10 次測試 ###
    num_trials = 50
    total_score = 0

    for i in range(num_trials):
        grid_size = random.randint(5, 10)  # 隨機選擇 5~10 的 grid_size
        env_config = {
            "grid_size": grid_size,
            "fuel_limit": 5000,
            "randomize_passenger": True,
            "randomize_destination": True
        }

        print(f"=== Running trial {i+1} with grid_size={grid_size} ===")
        agent_score = run_agent("student_agent.py", env_config, render=False)
        total_score += agent_score

    avg_score = total_score / num_trials  # 計算平均分數
    print(f"\nFinal Average Score over {num_trials} runs: {avg_score}")

    ### 提交結果 ###
    params = {
        'act': 'add',
        'name': sub_name,
        'score': str(avg_score),  # 提交平均分數
        'token': args.token
    }
    url = 'http://140.114.89.61/drl_hw1/action.php'

    response = requests.get(url, params=params)
    if response.ok:
        print('Success:', response.text)
    else:
        print('Error:', response.status_code)
