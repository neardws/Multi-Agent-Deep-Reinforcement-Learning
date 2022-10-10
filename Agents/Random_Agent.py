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
        self.total_episode_view_required_number_so_far = 0
        self.total_episode_score_so_far = 0
        self.new_total_episode_score_so_far = 0
        self.total_episode_age_of_view_so_far = 0
        self.total_episode_timeliness_so_far = 0
        self.total_episode_consistence_so_far = 0
        self.total_episode_completeness_so_far = 0
        self.total_episode_intel_arrival_time = 0
        self.total_episode_queuing_time_so_far = 0
        self.total_episode_transmitting_time_so_far = 0
        self.total_episode_service_time_so_far = 0
        self.total_episode_service_rate = 0
        self.total_episode_received_data_number = 0
        self.total_episode_required_data_number = 0
        self.environment.reset()

    def config_environment(self, environment):
        self.environment = environment
        self.environment.reset()

    def step(self):
        with tqdm(total=300) as my_bar:
            while not self.done:
                self.pick_action()
                self.conduct_action()
                my_bar.update(n=1)

    def pick_action(self):

        priority = np.zeros(shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number),
                            dtype=np.float)
        arrival_rate = np.zeros(
            shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number), dtype=np.float)

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):

            sensor_node_action_of_priority = random_np(self.environment.experiment_config.data_types_number)
            sensor_node_action_of_arrival_rate = random_np(self.environment.experiment_config.data_types_number)

            for data_type_index in range(self.environment.experiment_config.data_types_number):
                if self.environment.state["data_types"][sensor_node_index][data_type_index] == 1:
                    priority[sensor_node_index][data_type_index] = sensor_node_action_of_priority[data_type_index]

                    arrival_rate[sensor_node_index][data_type_index] = \
                        float(sensor_node_action_of_arrival_rate[data_type_index]) / \
                        self.environment.experiment_config.mean_service_time_of_types[sensor_node_index][data_type_index]

        # print("mean_service_time_of_types: \n", self.environment.experiment_config.mean_service_time_of_types)

        edge_nodes_bandwidth = random_np(self.environment.experiment_config.vehicle_number)

        edge_nodes_bandwidth = edge_nodes_bandwidth[np.newaxis, :]
        self.action = {
            "priority": priority,
            "arrival_rate": arrival_rate,
            "bandwidth": edge_nodes_bandwidth[0]
        }

    def conduct_action(self):
        _, _, _, _, self.reward, view_required_number, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
        sum_intel_arrival_time, sum_queuing_time, sum_transmitting_time, sum_service_time, sum_service_rate, sum_received_data_number, \
        sum_required_data_number, new_reward = self.environment.step_with_difference_rewards(self.action)
        self.total_episode_score_so_far += self.reward
        self.total_episode_view_required_number_so_far += view_required_number
        self.new_total_episode_score_so_far += new_reward
        self.total_episode_age_of_view_so_far += sum_age_of_view
        self.total_episode_timeliness_so_far += sum_timeliness
        self.total_episode_consistence_so_far += sum_consistence
        self.total_episode_completeness_so_far += sum_completeness
        self.total_episode_intel_arrival_time += sum_intel_arrival_time
        self.total_episode_queuing_time_so_far += sum_queuing_time
        self.total_episode_transmitting_time_so_far += sum_transmitting_time
        self.total_episode_service_time_so_far += sum_service_time
        self.total_episode_service_rate += sum_service_rate / self.environment.max_episode_length
        self.total_episode_received_data_number += sum_received_data_number
        self.total_episode_required_data_number += sum_required_data_number

    def run_n_episodes_as_results(self, num_episodes, result_name):

        try:
            result_data = pd.read_csv(result_name, names=["Epoch index", "age_of_view", "new_age_of_view", "timeliness", "consistence", "completeness", "intel_arrival_time", "queuing_time", "transmitting_time", "service_time", "service_rate", "received_data", "required_data"], header=0)
        except FileNotFoundError:
            result_data = pd.DataFrame(data=None, columns={"Epoch index": "", "age_of_view": "", "new_age_of_view": "", "timeliness": "", "consistence": "", "completeness": "", "intel_arrival_time": "", "queuing_time": "", "transmitting_time": "", "service_time": "", "service_rate": "",  "received_data": "", "required_data": ""},
                                       index=[0])

        for i in range(num_episodes):
            print("*" * 64)
            self.reset_game()
            self.step()

            self.new_total_episode_score_so_far = self.total_episode_age_of_view_so_far
            self.total_episode_age_of_view_so_far /= self.total_episode_view_required_number_so_far
            print("Epoch index: ", i)
            print("age_of_view: ", self.total_episode_age_of_view_so_far)
            print("new_age_of_view: ", self.new_total_episode_score_so_far)
            
            self.total_episode_timeliness_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_consistence_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_completeness_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_intel_arrival_time /= self.total_episode_view_required_number_so_far
            self.total_episode_queuing_time_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_transmitting_time_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_service_time_so_far /= self.total_episode_view_required_number_so_far

            new_line_in_result = pd.DataFrame({
                "Epoch index": str(i),
                "age_of_view": str(self.total_episode_age_of_view_so_far),
                "new_age_of_view": str(self.new_total_episode_score_so_far),
                "timeliness": str(self.total_episode_timeliness_so_far),
                "consistence": str(self.total_episode_consistence_so_far),
                "completeness": str(self.total_episode_completeness_so_far),
                "intel_arrival_time": str(self.total_episode_intel_arrival_time),
                "queuing_time": str(self.total_episode_queuing_time_so_far),
                "transmitting_time": str(self.total_episode_transmitting_time_so_far),
                "service_time": str(self.total_episode_service_time_so_far),
                "service_rate": str(self.total_episode_service_rate),
                "received_data": str(self.total_episode_received_data_number),
                "required_data": str(self.total_episode_required_data_number)
            }, index=["0"])
            result_data = result_data.append(new_line_in_result, ignore_index=True)
            result_data.to_csv(result_name)
            print("save result data successful")

    def run_n_episodes(self, num_episodes=None, temple_result_name=None):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.environment.experiment_config.episode_number

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
        self.total_episode_view_required_number_so_far = 0
        self.total_episode_score_so_far = 0
        self.new_total_episode_score_so_far = 0
        self.total_episode_age_of_view_so_far = 0
        self.total_episode_timeliness_so_far = 0
        self.total_episode_consistence_so_far = 0
        self.total_episode_completeness_so_far = 0
        self.total_episode_intel_arrival_time = 0
        self.total_episode_queuing_time_so_far = 0
        self.total_episode_transmitting_time_so_far = 0
        self.total_episode_service_time_so_far = 0
        self.total_episode_service_rate = 0
        self.total_episode_received_data_number = 0
        self.total_episode_required_data_number = 0

