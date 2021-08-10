# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：VehicularNetworkEnv.py
@Author  ：Neardws
@Date    ：6/22/21 4:09 下午 
"""
import gym
import numpy as np
import torch
import pandas as pd
from torch import Tensor
from Config.ExperimentConfig import ExperimentConfig

"""
Workflow of VehicularNetworkEnv

No.1 New objective of VehicularNetworkEnv at experiment start
     call __init__(experiment_config) to initializations

No.2 VehicularNetworkEnv reset at each episode start
     call reset() to reset the environment
     
No.3 Do each step at each time slot
     call step() to get state, action, reward, next_state, done
     
"""


# noinspection PyPep8Naming
class VehicularNetworkEnv(gym.Env):
    """Vehicular Network Environment that follows gym interface"""
    action: dict
    metadata = {'render.modes': []}

    def __init__(self, experiment_config: ExperimentConfig):
        """
        first the environment via Experiment_Config
        :param experiment_config:
        """
        super(VehicularNetworkEnv, self).__init__()

        self.config = experiment_config
        assert self.config is not None

        self.device = "cuda" if self.config.use_gpu else "cpu"

        """Experiment Setup"""
        self.episode_number = self.config.episode_number
        self.max_episode_length = self.config.max_episode_length

        """Random generated of data size of all types"""
        np.random.seed(self.config.seed_data_size_of_types)
        self.data_size_of_types = np.random.uniform(low=self.config.data_size_low_bound,
                                                    high=self.config.data_size_up_bound,
                                                    size=self.config.data_types_number)

        """Random generated of data types in all vehicles"""
        np.random.seed(self.config.seed_data_types_in_vehicles)
        self.data_types_in_vehicles = np.random.rand(self.config.vehicle_number, self.config.data_types_number)
        for value in np.nditer(self.data_types_in_vehicles, op_flags=['readwrite']):
            if value <= self.config.threshold_data_types_in_vehicles:
                value[...] = 1
            else:
                value[...] = 0

        """Random generated of edge views requirement at each time-slot in one edge node"""
        np.random.seed(self.config.seed_edge_views_in_edge_node)
        self.edge_views_in_edge_node = np.random.rand(self.config.edge_views_number, self.config.time_slots_number)
        for value in np.nditer(self.edge_views_in_edge_node, op_flags=['readwrite']):
            if value <= self.config.threshold_edge_views_in_edge_node:
                value[...] = 1
            else:
                value[...] = 0
        for time_slot_index in range(self.config.edge_view_required_start_time):
            for edge_view_index in range(self.config.edge_views_number):
                self.edge_views_in_edge_node[edge_view_index][time_slot_index] = 0

        """Random generated of view required data"""
        np.random.seed(self.config.seed_view_required_data)
        self.view_required_data = np.random.rand(self.config.edge_views_number,
                                                 self.config.vehicle_number,
                                                 self.config.data_types_number)

        for edge_view_index in range(self.config.edge_views_number):
            for vehicle_index in range(self.config.vehicle_number):
                for data_types_index in range(self.config.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_types_index] == 1:
                        if self.view_required_data[edge_view_index][vehicle_index][data_types_index] \
                                < self.config.threshold_view_required_data:
                            self.view_required_data[edge_view_index][vehicle_index][data_types_index] = 1
                        else:
                            self.view_required_data[edge_view_index][vehicle_index][data_types_index] = 0
                    else:
                        self.view_required_data[edge_view_index][vehicle_index][data_types_index] = 0

        """Trajectories and data in edge node"""

        """
        --------------------------------------------------------------------------------------------
        -----------------------Parameters for Reinforcement Learning
        --------------------------------------------------------------------------------------------
        """

        """float parameters"""
        self.reward = None  # external reward
        self.done = None
        """dict() parameters"""
        self.state = dict()  # global state
        self.action = dict()  # global action
        """torch.Tensor parameters"""
        self.reward_observation = None
        self.sensor_nodes_observation = None  # individually observation state for sensor nodes
        self.edge_node_observation = None  # individually observation state for edge node
        """other parameters"""
        self.episode_index = 0   # in which episode of whole episode number
        self.episode_step = None    # in which step of whole one episode
        self.global_trajectories = None

        self.trajectories_file_name = "~/Documents/pyProject_hrl/CSV/vehicle.csv"
        self.trajectories = None

        self.init_experiences_global_trajectory()
        self.get_mean_and_second_moment_service_time_of_types()

        self.waiting_time_in_queue = None
        self.action_time_of_sensor_nodes = None
        self.next_action_time_of_sensor_nodes = None
        self.required_to_transmit_data_size_of_sensor_nodes = None
        self.data_in_edge_node = None

    def reset(self):

        """
        Reset the environment to an initial state
        :return:
        """

        """Parameters for Reinforcement Learning"""
        self.episode_step = 0
        self.trajectories = np.zeros(shape=(self.config.vehicle_number, self.config.trajectories_predicted_time),
                                     dtype=np.float)
        self.init_trajectory()

        self.waiting_time_in_queue = np.zeros(shape=(self.config.vehicle_number, self.config.data_types_number))
        self.action_time_of_sensor_nodes = np.zeros(shape=self.config.vehicle_number)
        self.next_action_time_of_sensor_nodes = np.zeros(shape=self.config.vehicle_number)
        self.required_to_transmit_data_size_of_sensor_nodes = np.zeros(shape=(self.config.vehicle_number,
                                                                              self.config.data_types_number))
        self.data_in_edge_node = np.zeros(shape=(self.config.vehicle_number, self.config.data_types_number))

        """Notice that time, action time, data in edge node, and trajectories varying with time"""
        self.state = {  # global state
            'time': self.episode_step,
            'action_time': self.action_time_of_sensor_nodes,
            'data_in_edge': self.data_in_edge_node,
            'trajectories': self.trajectories,
            'data_types': self.data_types_in_vehicles,
            'edge_view': self.edge_views_in_edge_node,
            'view': self.view_required_data
        }
        self.action = dict()
        self.reward = None  # external reward
        self.done = False

        """get Tensor type parameters"""
        self.sensor_nodes_observation = self.init_sensor_observation()  # individually observation state for sensor node
        self.edge_node_observation = self.init_edge_observation()
        self.reward_observation = self.init_reward_observation()

        return self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation

    def init_experiences_global_trajectory(self):
        self.global_trajectories = np.zeros(shape=(self.config.vehicle_number, self.config.time_slots_number),
                                            dtype=np.float)

        df = pd.read_csv(self.trajectories_file_name, names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)

        max_vehicle_id = df['vehicle_id'].max()

        random_vehicle_id = np.random.choice(int(max_vehicle_id), self.config.vehicle_number, replace=False)

        new_vehicle_id = 0
        for vehicle_id in random_vehicle_id:
            new_df = df[df['vehicle_id'] == vehicle_id]
            for row in new_df.itertuples():
                time = getattr(row, 'time')
                x = getattr(row, 'longitude')
                y = getattr(row, 'latitude')
                distance = np.sqrt((x - self.config.edge_node_x) ** 2 + (y - self.config.edge_node_y) ** 2)
                self.global_trajectories[new_vehicle_id][int(time)] = distance
            new_vehicle_id += 1

    def init_trajectory(self):
        for vehicle_index in range(self.config.vehicle_number):
            for time_slot_index in range(self.episode_step,
                                         self.episode_step + self.config.trajectories_predicted_time):
                self.trajectories[vehicle_index][time_slot_index] = self.global_trajectories[vehicle_index][
                    time_slot_index]

    def update_trajectories(self):
        if self.episode_step <= 290:
            for vehicle_index in range(self.config.vehicle_number):
                index = 0
                for time_slot_index in range(self.episode_step,
                                             self.episode_step + self.config.trajectories_predicted_time):
                    self.trajectories[vehicle_index][index] = self.global_trajectories[vehicle_index][
                        time_slot_index]
                    index += 1

    """
       /*________________________________________________________________
       NN Input and Output Dimensions

           Actor network of sensor node 
               Input: 
                   get_sensor_observation_size()
               Output:
                   get_sensor_action_size()
           Critic network of sensor node
               Input:
                   get_critic_size_for_sensor() = get_sensor_observation_size() + 
                   get_sensor_node_action_size() * all sensor nodes
               Output:
                   1

           Actor network of edge node
               Input:
                   get_actor_input_size_for_edge() = get_edge_observations_size() + 
                   get_sensor_action_size() * all sensor nodes
               Output:
                   get_edge_action_size()
           Critic network of edge node
               Input:
                   get_critic_size_for_edge() = get_actor_input_size_for_edge() + get_edge_action_size()
               Output:
                   1

           Actor network of reward function
               Input:
                   get_actor_input_size_for_reward() = get_global_state_size() + get_global_action_size()
               Output:
                   get_reward_action_size()
           Critic network of reward function
               Input:
                   get_critic_size_for_reward() = get_actor_input_size_for_reward() + get_reward_action_size()
               Output:
                   1
           ________________________________________________________________*/
       """

    def get_sensor_observation_size(self):
        """
        :return
            Observation state input to neural network
                    [
                        time
                        action_time
                        data_in_edge
                        data_types_in_vehicle
                        edge_view_in_edge_node
                        view_required_data
                    ]
        """
        return int(
            1  # time_slots_index, changeable with time
            + 1  # action_time_of_vehicle, changeable with action of vehicle
            + int(self.config.data_types_number)  # data_in_edge, changeable with action of vehicle
            + int(self.config.data_types_number)  # required data of the vehicle according to edge view in edge
            # + int(self.config.edge_views_number)  # edge_view_in_edge_node
            # + int(self.config.data_types_number)  # data_types_in_vehicle, unchangeable
            # + int(self.config.data_types_number * self.config.edge_views_number)  # view_required_data, unchangeable
        )

    def get_sensor_action_size(self):
        """
        :return
            Action output from neural network
                [
                    priority
                    arrival rate
                ]
        """
        return int(
            self.config.data_types_number  # priority of each data type
            + self.config.data_types_number  # arrival rate * mean service time of each data type
        )

    def get_critic_size_for_sensor(self):
        return self.get_sensor_observation_size() + self.get_sensor_action_size() * self.config.vehicle_number

    def get_edge_observation_size(self):
        """
        :return
            Observation state input to neural network
                [
                    time
                    data_in_edge
                    trajectories
                    data_types_of_all_vehicles
                    edge_view
                    view
                ]
        """
        return int(
            1  # time_slots_index
            + int(self.config.vehicle_number * self.config.data_types_number)  # owned data types of all vehicles in edge node
            + int(self.config.vehicle_number * self.config.trajectories_predicted_time)  # predicted trajectories of all vehicles
            + int(self.config.vehicle_number * self.config.data_types_number)  # required data in all vehicles according to edge view
            # + int(self.config.edge_views_number)  # required edge view in edge node
            # + int(self.config.vehicle_number * self.config.data_types_number)  # data types of all vehicles
            # + int(self.config.vehicle_number * self.config.data_types_number * self.config.edge_views_number)   # view required data
        )

    def get_actor_input_size_for_edge(self):
        """
        Edge observation plus sensor nodes` actions
        :return:
        """
        return self.get_edge_observation_size() + self.get_sensor_action_size() * self.config.vehicle_number

    def get_edge_action_size(self):
        """
        :return
             Action output from neural network
             [
                    bandwidth
             ]
        """
        return int(
            self.config.vehicle_number
        )

    def get_critic_size_for_edge(self):
        return self.get_actor_input_size_for_edge() + self.get_edge_action_size()

    def get_global_state_size(self):
        """
            :return
                Observation state input to neural network
                    [
                        time
                        action_time
                        data_in_edge
                        trajectories
                        data_types
                        edge_view
                        view
                    ]
        """
        return int(
            1  # time_slots_index
            + int(self.config.vehicle_number)  # action time of sensor nodes
            + int(self.config.vehicle_number * self.config.data_types_number)  # owned data types of all vehicles in edge node
            + int(self.config.vehicle_number * self.config.trajectories_predicted_time)  # predicted trajectories of all vehicles
            + int(self.config.vehicle_number * self.config.data_types_number)   # required data in all vehicles according to edge view
            # + int(self.config.edge_views_number)  # required edge view in edge node
            # + int(self.config.vehicle_number * self.config.data_types_number)  # data types of all vehicles
            # + int(self.config.vehicle_number * self.config.data_types_number * self.config.edge_views_number)   # view required data
        )

    def get_global_action_size(self):
        """
        :return
            sensor action of all vehicles
            edge action
        """
        return int(
            (self.config.data_types_number + self.config.data_types_number) * self.config.vehicle_number
            + self.config.vehicle_number
        )

    def get_actor_input_size_for_reward(self):
        return self.get_global_state_size() + self.get_global_action_size()

    def get_reward_action_size(self):
        """
        :return:
            internal reward for sensor nodes and edge node
        """
        return self.config.vehicle_number + 1

    def get_critic_size_for_reward(self):
        return self.get_actor_input_size_for_reward() + self.get_reward_action_size()

    """
    /*________________________________________________________________
    NN Input and Output Dimensions End 
    ________________________________________________________________*/
    """

    """
    /*——————————————————————————————————————————————————————————————
        Init and update NN input and output
    —————————————————————————————————————————————————————————————--*/
    """

    def init_sensor_observation(self):
        """
        Inputs of actor network of sensor nodes
        :return:
        """
        sensor_nodes_observation_list = []
        for vehicle_index in range(self.config.vehicle_number):
            observation = np.zeros(shape=self.get_sensor_observation_size(),
                                   dtype=np.float)
            index_start = 0
            observation[index_start] = float(self.state['time']) / self.config.time_slots_number

            index_start = 1
            observation[index_start] = self.state['action_time'][vehicle_index] / self.config.time_slots_number
            index_start += 1

            for data_type_index in range(self.config.data_types_number):
                observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.config.time_slots_number
                index_start += 1

            for data_type_index in range(self.config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:   # vehicle holds the data type
                    for edge_view_index in range(self.config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][0] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                observation[index_start] = 1
                index_start += 1

            observation = Tensor(observation)
            sensor_nodes_observation_list.append(observation)

        sensor_nodes_observation = torch.cat((sensor_nodes_observation_list[0].unsqueeze(0), sensor_nodes_observation_list[1].unsqueeze(0)),
                                             dim=0)

        for index, values in enumerate(sensor_nodes_observation_list):
            if index > 1:
                sensor_nodes_observation = torch.cat((sensor_nodes_observation, values.unsqueeze(0)), dim=0)

        return sensor_nodes_observation.to(self.device)

    def init_edge_observation(self):
        observation = np.zeros(shape=self.get_edge_observation_size(),
                               dtype=np.float)
        index_start = 0
        observation[index_start] = float(self.state['time']) / self.config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for time_index in range(self.config.trajectories_predicted_time):
                observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.config.communication_range
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][0] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                observation[index_start] = 1
                index_start += 1

        return torch.from_numpy(observation).to(self.device)

    def init_reward_observation(self):
        observation = np.zeros(shape=self.get_global_state_size(),
                               dtype=np.float)
        index_start = 0
        observation[index_start] = float(self.state['time']) / self.config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.config.vehicle_number):
            observation[index_start] = self.state['action_time'][vehicle_index] / self.config.time_slots_number
            index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for time_index in range(self.config.trajectories_predicted_time):
                observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.config.communication_range
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][0] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                observation[index_start] = 1
                index_start += 1

        return torch.from_numpy(observation).to(self.device)

    """
    /*——————————————————————————————————————————————————————————————
        Init and update NN input and output End
    —————————————————————————————————————————————————————————————--*/
    """

    def step(self, action: dict):
        """
        Execute one time step within the environment
        :param action:
        :return: self.next_reward_observation, self.next_sensor_nodes_observation,
                 self.next_edge_node_observation, self.reward, self.done
        """
        self.action = action

        if self.episode_step == (self.config.max_episode_length - 1):
            self.episode_index = self.episode_index + 1
            self.done = True
        else:
            self.done = False

        """
        When the sensor node conduct the action, update the action time
        and the average waiting time in the queue in stable system
        """
        for vehicle_index in range(self.config.vehicle_number):
            """When the action time equal to now time"""
            if self.next_action_time_of_sensor_nodes[vehicle_index] == self.episode_step:

                for data_type_index in range(self.config.data_types_number):
                    self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = \
                        self.data_size_of_types[data_type_index]

                vehicle_action = []
                for data_type_index in range(self.config.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1:
                        priority = self.action['priority'][vehicle_index][data_type_index]
                        arrival_rate = self.action['arrival_rate'][vehicle_index][data_type_index]
                        vehicle_action.append(
                            {'priority': priority, 'arrival_rate': arrival_rate, 'data_type': data_type_index})

                vehicle_action.sort(key=lambda value: value['priority'])
                max_average_waiting_time = 0
                for index, values in enumerate(vehicle_action):
                    data_type_index = values['data_type']
                    work_load_before_type = 0
                    mu_before_type = 0
                    if index != 0:
                        for i in range(index):
                            work_load_before_type += vehicle_action[i]['arrival_rate'] * \
                                                     self.config.mean_service_time_of_types[vehicle_index][
                                                         vehicle_action[i]['data_type']]
                            mu_before_type += vehicle_action[i]['arrival_rate'] * \
                                self.config.second_moment_service_time_of_types[vehicle_index][
                                vehicle_action[i]['data_type']]

                    average_sojourn_time = 1 / (1 - work_load_before_type + values['arrival_rate'] *
                                                self.config.mean_service_time_of_types[vehicle_index][data_type_index])
                    if index != 0:
                        average_sojourn_time *= self.config.mean_service_time_of_types[vehicle_index][data_type_index] + \
                                                (mu_before_type / (2 * (1 - work_load_before_type)))
                    else:
                        average_sojourn_time *= self.config.mean_service_time_of_types[vehicle_index][data_type_index]

                    average_waiting_time = average_sojourn_time - self.config.mean_service_time_of_types[vehicle_index][
                        data_type_index]

                    """Update the waiting time in queue"""
                    try:
                        self.waiting_time_in_queue[vehicle_index][data_type_index] = average_waiting_time
                    except ValueError:
                        print(vehicle_index)
                        print(data_type_index)
                        print(self.waiting_time_in_queue[vehicle_index][data_type_index])
                        print(average_waiting_time)
                        print(average_sojourn_time)
                        print(self.config.mean_service_time_of_types[vehicle_index][data_type_index])

                    if average_waiting_time > max_average_waiting_time:
                        max_average_waiting_time = average_waiting_time

                """Update the action time"""
                self.action_time_of_sensor_nodes[vehicle_index] = self.next_action_time_of_sensor_nodes[vehicle_index]
                """may raise OverflowError: cannot convert float infinity to integer"""
                try:
                    self.next_action_time_of_sensor_nodes[vehicle_index] += int(max_average_waiting_time)
                except OverflowError:
                    self.next_action_time_of_sensor_nodes[vehicle_index] += 1
        """
        Update data_in_edge_node
        """
        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                if self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] > 0:
                    transmission_start_time = self.action_time_of_sensor_nodes[vehicle_index] + \
                                              self.waiting_time_in_queue[vehicle_index][data_type_index]
                    if self.episode_step >= transmission_start_time:
                        self.data_in_edge_node[vehicle_index][data_type_index] = 0
                        SNR = self.compute_SNR(vehicle_index, self.episode_step)
                        SNR_wall = self.computer_SNR_wall_by_noise_uncertainty(
                            noise_uncertainty=np.random.uniform(low=self.config.noise_uncertainty_low_bound,
                                                                high=self.config.noise_uncertainty_up_bound))
                        if SNR <= SNR_wall:
                            self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = 0
                            self.data_in_edge_node[vehicle_index][data_type_index] = 0

                        bandwidth = self.action['bandwidth'][0][vehicle_index]
                        transmission_bytes = self.compute_transmission_rate(SNR, bandwidth) * 1
                        self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][
                            data_type_index] -= transmission_bytes
                        if self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] <= 0:
                            self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = 0
                            self.data_in_edge_node[vehicle_index][data_type_index] = self.episode_step

        """Computes the reward"""
        sum_age_of_view = 0
        view_required_number = 0
        for edge_view_index in range(self.config.edge_views_number):
            if self.edge_views_in_edge_node[edge_view_index][self.episode_step] == 1:
                view_required_number += 1
                received_data_number = 0
                required_data_number = 0
                average_generation_time = 0
                timeliness = 0
                consistence = 0
                for vehicle_index in range(self.config.vehicle_number):
                    for data_type_index in range(self.config.data_types_number):
                        if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                            required_data_number += 1
                            if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                received_data_number += 1
                                if self.action["arrival_rate"][vehicle_index][data_type_index] == 0:
                                    print("arrival_rate is zeros")
                                try:
                                    intel_arrival_time = 1 / self.action["arrival_rate"][vehicle_index][data_type_index]
                                except ZeroDivisionError:
                                    intel_arrival_time = 0
                                timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][
                                    data_type_index] - self.action_time_of_sensor_nodes[vehicle_index])
                                average_generation_time += self.action_time_of_sensor_nodes[vehicle_index]
                try:
                    average_generation_time /= received_data_number
                except ZeroDivisionError:
                    average_generation_time = 0
                    # print("Error in VehicularNetworkEnv line 655")
                    # print(required_data_number)
                    # print(received_data_number)
                    # print(average_generation_time)

                for vehicle_index in range(self.config.vehicle_number):
                    for data_type_index in range(self.config.data_types_number):
                        if (self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1) and (
                                self.data_in_edge_node[vehicle_index][data_type_index] > 0):
                            consistence += np.abs(
                                self.action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                if required_data_number == 0 or received_data_number == 0:
                    completeness = 0
                else:
                    completeness = received_data_number / required_data_number  # the number of successfully received data divided required data
                # timeliness and consistence should be min as they can
                # however, completeness should be max as it can
                age_of_view = 1 / 3 * (2 - (np.tanh(timeliness) + np.tanh(consistence)) + completeness)
                sum_age_of_view += age_of_view

        if view_required_number == 0 or sum_age_of_view == 0:
            self.reward = 0
        else:
            self.reward = sum_age_of_view / view_required_number

        """Update state and other observations"""

        self.update_trajectories()

        self.state["time"] = self.episode_step
        self.state["trajectories"] = self.trajectories
        self.state["action_time"] = self.action_time_of_sensor_nodes
        self.state["data_in_edge"] = self.data_in_edge_node

        self.update_sensor_observation()
        self.update_edge_observation()
        self.update_reward_observation()

        self.episode_step += 1

        return self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation, \
            self.reward, self.done

    """
    /*——————————————————————————————————————————————————————————————
        Update NN input
    —————————————————————————————————————————————————————————————--*/
    """

    def update_sensor_observation(self):
        """
        Update the input of actor network at each time slot
        """
        for vehicle_index in range(self.config.vehicle_number):
            index_start = 0
            self.sensor_nodes_observation[vehicle_index][index_start] = float(
                self.state['time']) / self.config.time_slots_number

            index_start = 1
            self.sensor_nodes_observation[vehicle_index][index_start] = self.state['action_time'][vehicle_index] / self.config.time_slots_number
            index_start += 1

            for data_type_index in range(self.config.data_types_number):
                self.sensor_nodes_observation[vehicle_index][index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.config.time_slots_number
                index_start += 1

            for data_type_index in range(self.config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:   # vehicle holds the data type
                    for edge_view_index in range(self.config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][int(self.state['time'])] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                self.sensor_nodes_observation[vehicle_index][index_start] = 1
                index_start += 1

    def update_edge_observation(self):
        index_start = 0
        self.edge_node_observation[index_start] = float(self.state['time']) / self.config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                self.edge_node_observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for time_index in range(self.config.trajectories_predicted_time):
                self.edge_node_observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.config.communication_range
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][int(self.state['time'])] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                self.edge_node_observation[index_start] = 1
                index_start += 1

    def update_reward_observation(self):
        index_start = 0
        self.reward_observation[index_start] = float(self.state['time']) / self.config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.config.vehicle_number):
            self.reward_observation[index_start] = self.state['action_time'][vehicle_index] / self.config.time_slots_number
            index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                self.reward_observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for time_index in range(self.config.trajectories_predicted_time):
                self.reward_observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.config.communication_range
                index_start += 1

        for vehicle_index in range(self.config.vehicle_number):
            for data_type_index in range(self.config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][int(self.state['time'])] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                self.reward_observation[index_start] = 1
                index_start += 1

    """
    /*——————————————————————————————————————————————————————————————
        Update NN input END
    —————————————————————————————————————————————————————————————--*/
    """

    def compute_SNR(self, vehicle_index, time_slot):
        """

        :param vehicle_index:
        :param time_slot:
        :return: SNR ratio
        """
        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.config.additive_white_gaussian_noise)
        channel_fading_gain = np.random.normal(loc=self.config.mean_channel_fading_gain,
                                               scale=self.config.second_moment_channel_fading_gain)
        distance = self.global_trajectories[vehicle_index][time_slot]
        SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
            1 / (np.power(distance, self.config.path_loss_exponent)) * \
            VehicularNetworkEnv.cover_mW_to_W(self.config.transmission_power)
        return SNR

    def compute_SNR_by_distance(self, distance):
        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.config.additive_white_gaussian_noise)
        channel_fading_gain = np.random.normal(loc=self.config.mean_channel_fading_gain,
                                               scale=self.config.second_moment_channel_fading_gain)
        SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
            1 / (np.power(distance, self.config.path_loss_exponent)) * \
            VehicularNetworkEnv.cover_mW_to_W(self.config.transmission_power)
        return SNR

    @staticmethod
    def computer_SNR_wall_by_noise_uncertainty(noise_uncertainty):
        return (np.power(VehicularNetworkEnv.cover_dB_to_ratio(noise_uncertainty), 2) - 1) / \
               VehicularNetworkEnv.cover_dB_to_ratio(noise_uncertainty)

    @staticmethod
    def compute_transmission_rate(SNR, bandwidth):
        """
        :param SNR:
        :param bandwidth:
        :return: transmission rate measure by Byte/s
        """
        return int(VehicularNetworkEnv.cover_MHz_to_Hz(bandwidth) * np.log2(1 + SNR) / 8)

    @staticmethod
    def cover_MHz_to_Hz(MHz):
        return MHz * 10e6

    @staticmethod
    def cover_ratio_to_dB(ratio):
        return 10 * np.log10(ratio)

    @staticmethod
    def cover_dB_to_ratio(dB):
        return np.power(10, (dB / 10))

    @staticmethod
    def cover_dBm_to_W(dBm):
        return np.power(10, (dBm / 10)) / 1000

    @staticmethod
    def cover_W_to_dBm(W):
        return 10 * np.log10(W * 1000)

    @staticmethod
    def cover_W_to_mW(W):
        return W * 1000

    @staticmethod
    def cover_mW_to_W(mW):
        return mW / 1000

    def get_mean_and_second_moment_service_time_of_types(self):
        mean_service_time_of_types = np.zeros(shape=(self.config.vehicle_number,
                                                     self.config.data_types_number),
                                              dtype=np.float)
        second_moment_service_time_of_types = np.zeros(shape=(self.config.vehicle_number,
                                                              self.config.data_types_number),
                                                       dtype=np.float)

        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.config.additive_white_gaussian_noise)

        data_type_index = 0
        for data_size in self.data_size_of_types:
            speed_time = []
            vehicle_index = 0
            for vehicle_distance in self.global_trajectories:
                for distance in vehicle_distance:
                    channel_fading_gain = np.random.normal(loc=self.config.mean_channel_fading_gain,
                                                           scale=self.config.second_moment_channel_fading_gain)
                    SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
                        np.power(distance, -self.config.path_loss_exponent) * \
                        VehicularNetworkEnv.cover_mW_to_W(self.config.transmission_power)
                    bandwidth = self.config.bandwidth * np.random.rand()
                    speed_time.append(data_size / self.compute_transmission_rate(SNR, bandwidth))
                array_speed_time = np.array(speed_time)
                mean_service_time = array_speed_time.mean()
                second_moment_service_time = array_speed_time.var()
                mean_service_time_of_types[vehicle_index][data_type_index] = mean_service_time
                second_moment_service_time_of_types[vehicle_index][data_type_index] = second_moment_service_time
                vehicle_index += 1
            data_type_index += 1

        self.config.config(
            mean_service_time_of_types=mean_service_time_of_types,
            second_moment_service_time_of_types=second_moment_service_time_of_types)

    # def get_arrival_rate_bounds(self):
    #     mean_service_time_of_types = self.config.mean_service_time_of_types
    #     second_moment_service_time_of_types = self.config.second_moment_service_time_of_types
    #
    #     for vehicle_index in range(self.config.vehicle_number):
    #         for data_type in range(self.config.data_types_number):
    #             if
    #     pass

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        :param mode:
        :param close:
        :return:
        """
        pass
