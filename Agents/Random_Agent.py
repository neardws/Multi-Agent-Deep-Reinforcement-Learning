# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning
@File    ：Random_Agent.py
@Author  ：Neardws
@Date    ：8/27/21 7:59 下午 
"""
from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


def random_np(data_size):
    random_np_array = np.random.rand(data_size)
    random_np_array = random_np_array / random_np_array.sum()
    return random_np_array


class Random_Agent(object):

    def __init__(self, environment: VehicularNetworkEnv):
        self.name = "Random_Agent"
        self.environment = environment
        self.reward = None
        self.action = None
        self.done = None
        self.total_episode_score_so_far = 0
        self.environment.reset()

    def step(self):
        with tqdm(total=300) as my_bar:
            while not self.done:
                self.pick_action()
                self.conduct_action()
                my_bar.update(n=1)

    def pick_action(self):

        priority = np.zeros(shape=(self.environment.config.vehicle_number, self.environment.config.data_types_number),
                            dtype=np.float)
        arrival_rate = np.zeros(
            shape=(self.environment.config.vehicle_number, self.environment.config.data_types_number), dtype=np.float)

        for sensor_node_index in range(self.environment.config.vehicle_number):

            sensor_node_action_of_priority = random_np(self.environment.config.data_types_number)
            sensor_node_action_of_arrival_rate = random_np(self.environment.config.data_types_number)

            for data_type_index in range(self.environment.config.data_types_number):
                if self.environment.state["data_types"][sensor_node_index][data_type_index] == 1:
                    priority[sensor_node_index][data_type_index] = sensor_node_action_of_priority[data_type_index]

                    arrival_rate[sensor_node_index][data_type_index] = \
                        float(sensor_node_action_of_arrival_rate[data_type_index]) / \
                        self.environment.config.mean_service_time_of_types[sensor_node_index][data_type_index]

        edge_nodes_bandwidth = random_np(self.environment.config.vehicle_number) * self.environment.config.bandwidth

        edge_nodes_bandwidth = edge_nodes_bandwidth[np.newaxis, :]
        self.action = {
            "priority": priority,
            "arrival_rate": arrival_rate,
            "bandwidth": edge_nodes_bandwidth
        }

    def conduct_action(self):
        _, _, _, self.reward, self.done = self.environment.step(self.action)
        self.total_episode_score_so_far += self.reward

    def run_n_episodes(self, num_episodes=None, temple_result_name=None):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.environment.config.episode_number

        try:
            result_data = pd.read_csv(temple_result_name, names=["Epoch index", "Total reward", "Time taken"], header=0)
        except FileNotFoundError:
            result_data = pd.DataFrame(data=None, columns={"Epoch index": "", "Total reward": "", "Time taken": ""}, index=[0])

        while self.environment.episode_index < num_episodes:
            print("*" * 64)
            start = time.time()
            self.reset_game()
            self.step()
            time_taken = time.time() - start
            print("Epoch index: ", self.environment.episode_index)
            print("Total reward: ", self.total_episode_score_so_far)
            print("Time taken: ", time_taken)
            new_line_in_result = pd.DataFrame({"Epoch index": str(self.environment.episode_index), "Total reward": str(self.total_episode_score_so_far), "Time taken": str(time_taken)}, index=["0"])
            result_data = result_data.append(new_line_in_result, ignore_index=True)
            if self.environment.episode_index % 10 == 0:
                print(result_data)
            if self.environment.episode_index == 1:
                result_data = result_data.drop(result_data.index[[0]])

            if self.environment.episode_index % 30 == 0:
                result_data.to_csv(temple_result_name)
                print("save result data successful")

    def reset_game(self):
        """float parameters"""
        self.reward = None
        self.done = None  # 1 or 0 indicate is episode finished
        self.action = None
        self.environment.reset()
        self.total_episode_score_so_far = 0

