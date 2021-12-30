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

Changable：
1. bandwidth
2. views required at each time slot
"""

# noinspection PyPep8Naming
class VehicularNetworkEnv(gym.Env):
    """Vehicular Network Environment that follows gym interface"""
    action: dict
    metadata = {'render.modes': []}
    
    def __init__(self, experiment_config: ExperimentConfig, trajectories_file_name):
        """
        first the environment via Experiment_Config
        :param experiment_config:
        """
        super(VehicularNetworkEnv, self).__init__()

        self.experiment_config = experiment_config
        assert self.experiment_config is not None

        self.device = "cuda:0" if self.experiment_config.use_gpu else "cpu"
        self.bandwidth = self.experiment_config.bandwidth
        self.threshold_view_required_data = self.experiment_config.threshold_view_required_data

        """Experiment Setup"""
        self.episode_number = self.experiment_config.episode_number
        self.max_episode_length = self.experiment_config.max_episode_length

        """Random generated of data size of all types"""
        np.random.seed(self.experiment_config.seed_data_size_of_types)
        self.data_size_of_types = np.random.uniform(low=self.experiment_config.data_size_low_bound,
                                                    high=self.experiment_config.data_size_up_bound,
                                                    size=self.experiment_config.data_types_number)

        """Random generated of data types in all vehicles"""
        np.random.seed(self.experiment_config.seed_data_types_in_vehicles)
        self.data_types_in_vehicles = np.random.rand(self.experiment_config.vehicle_number, self.experiment_config.data_types_number)
        for value in np.nditer(self.data_types_in_vehicles, op_flags=['readwrite']):
            if value <= self.experiment_config.threshold_data_types_in_vehicles:
                value[...] = 1
            else:
                value[...] = 0

        """Random generated of edge views requirement at each time-slot in one edge node"""
        np.random.seed(self.experiment_config.seed_edge_views_in_edge_node)
        self.edge_views_in_edge_node = np.random.rand(self.experiment_config.edge_views_number, self.experiment_config.time_slots_number)
        for value in np.nditer(self.edge_views_in_edge_node, op_flags=['readwrite']):
            if value <= self.experiment_config.threshold_edge_views_in_edge_node:
                value[...] = 1
            else:
                value[...] = 0
        for time_slot_index in range(self.experiment_config.edge_view_required_start_time):
            for edge_view_index in range(self.experiment_config.edge_views_number):
                self.edge_views_in_edge_node[edge_view_index][time_slot_index] = 0

        """Random generated of view required data"""
        np.random.seed(self.experiment_config.seed_view_required_data)
        self.view_required_data = np.random.rand(
            self.experiment_config.edge_views_number,
            self.experiment_config.vehicle_number,
            self.experiment_config.data_types_number
        )

        for edge_view_index in range(self.experiment_config.edge_views_number):
            for vehicle_index in range(self.experiment_config.vehicle_number):
                for data_types_index in range(self.experiment_config.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_types_index] == 1:
                        if self.view_required_data[edge_view_index][vehicle_index][data_types_index] \
                                < self.threshold_view_required_data:
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
        self.new_reward = None  # new reward
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
        self.episode_step = 0    # in which step of whole one episode
        self.global_trajectories = None
        self.trajectories_file_name = trajectories_file_name
        self.trajectories = None

        self.init_experiences_global_trajectory()
        self.get_mean_and_second_moment_service_time_of_types()

        self.waiting_time_in_queue = None
        self.last_action_time_of_sensor_nodes = None
        self.action_time_of_sensor_nodes = None
        self.next_action_time_of_sensor_nodes = None
        self.required_to_transmit_data_size_of_sensor_nodes = None
        self.data_in_edge_node = None
        
    def config_bandwidth(self, new_bandwidth):
        self.bandwidth = new_bandwidth
        self.get_mean_and_second_moment_service_time_of_types()

    def config_datasize_of_types(self, new_data_size_of_types):
        np.random.seed(self.experiment_config.seed_data_size_of_types)
        self.data_size_of_types = np.random.uniform(low=self.experiment_config.data_size_low_bound,
                                                    high=new_data_size_of_types,
                                                    size=self.experiment_config.data_types_number)
        self.get_mean_and_second_moment_service_time_of_types()

    def config_threshold_view_required_data(self, new_threshold_view_required_data):
        self.threshold_view_required_data = new_threshold_view_required_data
        np.random.seed(self.experiment_config.seed_view_required_data)
        self.view_required_data = np.random.rand(
            self.experiment_config.edge_views_number,
            self.experiment_config.vehicle_number,
            self.experiment_config.data_types_number
        )

        for edge_view_index in range(self.experiment_config.edge_views_number):
            for vehicle_index in range(self.experiment_config.vehicle_number):
                for data_types_index in range(self.experiment_config.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_types_index] == 1:
                        if self.view_required_data[edge_view_index][vehicle_index][data_types_index] \
                                < self.threshold_view_required_data:
                            self.view_required_data[edge_view_index][vehicle_index][data_types_index] = 1
                        else:
                            self.view_required_data[edge_view_index][vehicle_index][data_types_index] = 0
                    else:
                        self.view_required_data[edge_view_index][vehicle_index][data_types_index] = 0

    def config_views_required_at_each_time_slot(self, threshold_edge_views_in_edge_node):
        self.experiment_config.config(
            threshold_edge_views_in_edge_node=threshold_edge_views_in_edge_node
        )
        self.get_mean_and_second_moment_service_time_of_types()
        np.random.seed(self.experiment_config.seed_edge_views_in_edge_node)
        self.edge_views_in_edge_node = np.random.rand(self.experiment_config.edge_views_number, self.experiment_config.time_slots_number)
        for value in np.nditer(self.edge_views_in_edge_node, op_flags=['readwrite']):
            if value <= self.experiment_config.threshold_edge_views_in_edge_node:
                value[...] = 1
            else:
                value[...] = 0
        for time_slot_index in range(self.experiment_config.edge_view_required_start_time):
            for edge_view_index in range(self.experiment_config.edge_views_number):
                self.edge_views_in_edge_node[edge_view_index][time_slot_index] = 0

    def reset(self):

        """
        Reset the environment to an initial state
        :return:
        """

        """Parameters for Reinforcement Learning"""
        self.episode_step = 0
        self.trajectories = np.zeros(shape=(self.experiment_config.vehicle_number, self.experiment_config.trajectories_predicted_time),
                                     dtype=np.float)
        self.init_trajectory()

        self.waiting_time_in_queue = np.zeros(shape=(self.experiment_config.vehicle_number, self.experiment_config.data_types_number))
        self.last_action_time_of_sensor_nodes = np.zeros(shape=self.experiment_config.vehicle_number)
        self.action_time_of_sensor_nodes = np.zeros(shape=self.experiment_config.vehicle_number)
        self.next_action_time_of_sensor_nodes = np.zeros(shape=self.experiment_config.vehicle_number)
        self.required_to_transmit_data_size_of_sensor_nodes = np.zeros(shape=(self.experiment_config.vehicle_number,
                                                                              self.experiment_config.data_types_number))
        self.data_in_edge_node = np.zeros(shape=(self.experiment_config.vehicle_number, self.experiment_config.data_types_number))

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
        self.new_reward = None  # new reward
        self.done = False

        """get Tensor type parameters"""
        self.sensor_nodes_observation = self.init_sensor_observation()  # individually observation state for sensor node
        self.edge_node_observation = self.init_edge_observation()
        self.reward_observation = self.init_reward_observation()

        return self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation

    def init_experiences_global_trajectory(self):
        self.global_trajectories = np.zeros(shape=(self.experiment_config.vehicle_number, self.experiment_config.time_slots_number),
                                            dtype=np.float)

        df = pd.read_csv(self.trajectories_file_name, names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)

        max_vehicle_id = df['vehicle_id'].max()

        random_vehicle_id = np.random.choice(int(max_vehicle_id), self.experiment_config.vehicle_number, replace=False)

        new_vehicle_id = 0
        for vehicle_id in random_vehicle_id:
            new_df = df[df['vehicle_id'] == vehicle_id]
            for row in new_df.itertuples():
                time = getattr(row, 'time')
                x = getattr(row, 'longitude')
                y = getattr(row, 'latitude')
                distance = np.sqrt((x - self.experiment_config.edge_node_x) ** 2 + (y - self.experiment_config.edge_node_y) ** 2)
                self.global_trajectories[new_vehicle_id][int(time)] = distance
            new_vehicle_id += 1

    def init_trajectory(self):
        for vehicle_index in range(self.experiment_config.vehicle_number):
            for time_slot_index in range(self.episode_step,
                                         self.episode_step + self.experiment_config.trajectories_predicted_time):
                self.trajectories[vehicle_index][time_slot_index] = self.global_trajectories[vehicle_index][
                    time_slot_index]
    
    def get_distances_of_sensor_nodes(self):
        distances_of_sensor_nodes = []
        for sensor_node_index in range(self.experiment_config.vehicle_number):
            distance = self.global_trajectories[sensor_node_index][self.episode_step]
            distances_of_sensor_nodes.append(distance)
        return distances_of_sensor_nodes

    def get_required_data_size_of_sensor_nodes(self):
        required_data_size_of_sensor_nodes = np.zeros(shape=self.experiment_config.vehicle_number)
        for edge_view_index in range(self.experiment_config.edge_views_number):
            if self.edge_views_in_edge_node[edge_view_index][self.episode_step] == 1:
                for vehicle_index in range(self.experiment_config.vehicle_number):
                    for data_type_index in range(self.experiment_config.data_types_number):
                        view_required_data_size = self.view_required_data[edge_view_index][vehicle_index][data_type_index] * self.data_size_of_types[data_type_index]
                        required_data_size_of_sensor_nodes[vehicle_index] += view_required_data_size
        return required_data_size_of_sensor_nodes

    def update_trajectories(self):
        if self.episode_step <= 290:
            for vehicle_index in range(self.experiment_config.vehicle_number):
                index = 0
                for time_slot_index in range(self.episode_step,
                                             self.episode_step + self.experiment_config.trajectories_predicted_time):
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
            + int(self.experiment_config.data_types_number)  # data_in_edge, changeable with action of vehicle
            + int(self.experiment_config.data_types_number)  # required data of the vehicle according to edge view in edge
            # + int(self.experiment_config.edge_views_number)  # edge_view_in_edge_node
            # + int(self.experiment_config.data_types_number)  # data_types_in_vehicle, unchangeable
            # + int(self.experiment_config.data_types_number * self.experiment_config.edge_views_number)  # view_required_data, unchangeable
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
            self.experiment_config.data_types_number  # priority of each data type
            + self.experiment_config.data_types_number  # arrival rate * mean service time of each data type
        )

    def get_critic_size_for_sensor(self):
        return self.get_sensor_observation_size() + self.get_sensor_action_size() * self.experiment_config.vehicle_number

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
            + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number)  # owned data types of all vehicles in edge node
            + int(self.experiment_config.vehicle_number * self.experiment_config.trajectories_predicted_time)  # predicted trajectories of all vehicles
            + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number)  # required data in all vehicles according to edge view
            # + int(self.experiment_config.edge_views_number)  # required edge view in edge node
            # + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number)  # data types of all vehicles
            # + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number * self.experiment_config.edge_views_number)   # view required data
        )

    def get_actor_input_size_for_edge(self):
        """
        Edge observation plus sensor nodes` actions
        :return:
        """
        return self.get_edge_observation_size() + self.get_sensor_action_size() * self.experiment_config.vehicle_number

    def get_edge_action_size(self):
        """
        :return
             Action output from neural network
             [
                    bandwidth
             ]
        """
        return int(
            self.experiment_config.vehicle_number
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
            + int(self.experiment_config.vehicle_number)  # action time of sensor nodes
            + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number)  # owned data types of all vehicles in edge node
            + int(self.experiment_config.vehicle_number * self.experiment_config.trajectories_predicted_time)  # predicted trajectories of all vehicles
            + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number)   # required data in all vehicles according to edge view
            # + int(self.experiment_config.edge_views_number)  # required edge view in edge node
            # + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number)  # data types of all vehicles
            # + int(self.experiment_config.vehicle_number * self.experiment_config.data_types_number * self.experiment_config.edge_views_number)   # view required data
        )

    def get_global_action_size(self):
        """
        :return
            sensor action of all vehicles
            edge action
        """
        return int(
            (self.experiment_config.data_types_number + self.experiment_config.data_types_number) * self.experiment_config.vehicle_number
            + self.experiment_config.vehicle_number
        )

    def get_actor_input_size_for_reward(self):
        return self.get_global_state_size() + self.get_global_action_size()

    def get_reward_action_size(self):
        """
        :return:
            internal reward for sensor nodes and edge node
        """
        return self.experiment_config.vehicle_number + 1

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
        for vehicle_index in range(self.experiment_config.vehicle_number):
            observation = np.zeros(shape=self.get_sensor_observation_size(),
                                   dtype=np.float)
            index_start = 0
            observation[index_start] = float(self.state['time']) / self.experiment_config.time_slots_number

            index_start = 1
            observation[index_start] = self.state['action_time'][vehicle_index] / self.experiment_config.time_slots_number
            index_start += 1

            for data_type_index in range(self.experiment_config.data_types_number):
                observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.experiment_config.time_slots_number
                index_start += 1

            for data_type_index in range(self.experiment_config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:   # vehicle holds the data type
                    for edge_view_index in range(self.experiment_config.edge_views_number):
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
        observation[index_start] = float(self.state['time']) / self.experiment_config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.experiment_config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for time_index in range(self.experiment_config.trajectories_predicted_time):
                observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.experiment_config.communication_range
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.experiment_config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][0] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                observation[index_start] = 1
                index_start += 1

        return torch.from_numpy(observation).to(self.device)

    def init_reward_observation(self):
        observation = np.zeros(shape=self.get_global_state_size(),
                               dtype=np.float)
        index_start = 0
        observation[index_start] = float(self.state['time']) / self.experiment_config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.experiment_config.vehicle_number):
            observation[index_start] = self.state['action_time'][vehicle_index] / self.experiment_config.time_slots_number
            index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.experiment_config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for time_index in range(self.experiment_config.trajectories_predicted_time):
                observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.experiment_config.communication_range
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.experiment_config.edge_views_number):
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

    def step_with_difference_rewards(self, action: dict):
        """
        Execute one time step within the environment
        :param action:
        :return: self.next_reward_observation, self.next_sensor_nodes_observation,
                 self.next_edge_node_observation, self.reward, self.done
        """
        self.action = action

        if self.episode_step == (self.experiment_config.max_episode_length - 1):
            self.episode_index = self.episode_index + 1
            self.done = True
        else:
            self.done = False

        sum_age_of_view = 0
        sum_new_age_of_view = 0
        sum_timeliness = 0
        sum_consistence = 0
        sum_completeness = 0

        sum_intel_arrival_time = 0
        sum_queuing_time = 0
        sum_transmitting_time = 0
        sum_service_time = 0
        sum_service_rate = 0

        """
        When the sensor node conduct the action, update the action time
        and the average waiting time in the queue in stable system
        """

        max_average_waiting_time = 0
        for vehicle_index in range(self.experiment_config.vehicle_number):
            """When the action time equal to now time"""
            if self.next_action_time_of_sensor_nodes[vehicle_index] == self.episode_step:

                for data_type_index in range(self.experiment_config.data_types_number):
                    self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = \
                        self.data_size_of_types[data_type_index]

                vehicle_action = []
                for data_type_index in range(self.experiment_config.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1:
                        priority = self.action['priority'][vehicle_index][data_type_index]
                        arrival_rate = self.action['arrival_rate'][vehicle_index][data_type_index]
                        vehicle_action.append({
                            'priority': priority, 
                            'arrival_rate': arrival_rate, 
                            'data_type': data_type_index})

                vehicle_action.sort(key=lambda value: value['priority'], reverse=True)

                sum_average_waiting_time = 0
                sum_average_intel_arrival_time = 0
                for index, values in enumerate(vehicle_action):
                    data_type_index = values['data_type']
                    priority = values['priority']
                    arrival_rate = values['arrival_rate']
                    work_load_before_type = 0
                    mu_before_type = 0
                    if index != 0:
                        for i in range(index):
                            work_load_before_type += vehicle_action[i]['arrival_rate'] * \
                                self.experiment_config.mean_service_time_of_types[vehicle_index][vehicle_action[i]['data_type']]
                            mu_before_type += vehicle_action[i]['arrival_rate'] * \
                                self.experiment_config.second_moment_service_time_of_types[vehicle_index][
                                vehicle_action[i]['data_type']]

                    if index != 0:
                        average_sojourn_time = 1 / (1 - work_load_before_type) * \
                                                (self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index] + 
                                                ((mu_before_type + arrival_rate * self.experiment_config.second_moment_service_time_of_types[vehicle_index][data_type_index]) 
                                                / (2 * (1 - work_load_before_type - arrival_rate *
                                                self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index]))))
                    else:
                        average_sojourn_time = 1 / 1 * \
                                                self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index]
                    
                    average_waiting_time = average_sojourn_time - self.experiment_config.mean_service_time_of_types[vehicle_index][
                        data_type_index]
                    sum_average_waiting_time += average_waiting_time
                    sum_average_intel_arrival_time += 1 / arrival_rate
                    """Update the waiting time in queue"""
                    try:
                        self.waiting_time_in_queue[vehicle_index][data_type_index] = average_waiting_time
                    except ValueError:
                        print(vehicle_index)
                        print(data_type_index)
                        print(self.waiting_time_in_queue[vehicle_index][data_type_index])
                        print(average_waiting_time)
                        print(average_sojourn_time)
                        print(self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index])
                    
                    if average_waiting_time > max_average_waiting_time:
                        max_average_waiting_time = average_waiting_time
                        
                if len(vehicle_action) != 0:
                    sum_queuing_time += (sum_average_waiting_time / len(vehicle_action))
                    sum_intel_arrival_time += (sum_average_intel_arrival_time / len(vehicle_action))
                """Update the action time"""
                self.last_action_time_of_sensor_nodes[vehicle_index] = self.action_time_of_sensor_nodes[vehicle_index]
                self.action_time_of_sensor_nodes[vehicle_index] = self.next_action_time_of_sensor_nodes[vehicle_index]
                """may raise OverflowError: cannot convert float infinity to integer"""
                try:
                    self.next_action_time_of_sensor_nodes[vehicle_index] += np.ceil(max_average_waiting_time)
                except OverflowError:
                    self.next_action_time_of_sensor_nodes[vehicle_index] += 1
        
        """
        Update data_in_edge_node
        """
        for vehicle_index in range(self.experiment_config.vehicle_number):
            transmitting_time = 0
            data_type_number = 0
            for data_type_index in range(self.experiment_config.data_types_number):

                if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1 and \
                    self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] > 0:
                    data_type_number += 1
                    transmission_start_time = self.last_action_time_of_sensor_nodes[vehicle_index] + \
                                                self.waiting_time_in_queue[vehicle_index][data_type_index]
                    if self.episode_step >= transmission_start_time:
                        self.data_in_edge_node[vehicle_index][data_type_index] = 0
                        SNR = self.compute_SNR(vehicle_index, self.episode_step)
                        SNR_wall = self.computer_SNR_wall_by_noise_uncertainty(
                            noise_uncertainty=np.random.uniform(low=self.experiment_config.noise_uncertainty_low_bound,
                                                                high=self.experiment_config.noise_uncertainty_up_bound))
                        if SNR <= SNR_wall:
                            self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = 0
                            self.data_in_edge_node[vehicle_index][data_type_index] = 0
                        else:
                            transmitting_time += 1

                            bandwidth = self.action['bandwidth'][vehicle_index] * self.bandwidth
            
                            transmission_bytes = self.compute_transmission_rate(SNR, bandwidth) * 1
                            self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][
                                data_type_index] -= transmission_bytes
                            if self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] <= 0:
                                self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = 0
                                self.data_in_edge_node[vehicle_index][data_type_index] = self.episode_step
            if transmitting_time != 0:
                transmitting_time /= data_type_number

            sum_transmitting_time += transmitting_time
        sum_intel_arrival_time /= self.experiment_config.vehicle_number
        sum_queuing_time /= self.experiment_config.vehicle_number
        sum_transmitting_time /= self.experiment_config.vehicle_number

        sum_service_time += sum_intel_arrival_time + sum_queuing_time + sum_transmitting_time

        """Computes the reward"""
        view_required_number = 0
        view_serviced_number = 0
        sum_received_data_number = 0
        sum_required_data_number = 0

        for edge_view_index in range(self.experiment_config.edge_views_number):
            if self.edge_views_in_edge_node[edge_view_index][self.episode_step] == 1:
                view_required_number += 1
                received_data_number = 0
                required_data_number = 0
                average_generation_time = 0
                timeliness = 0
                new_age_timeliness = 0
                new_timeliness = 0
                new_age_consistence = 0
                consistence = 0
                new_consistence = 0
                for vehicle_index in range(self.experiment_config.vehicle_number):
                    for data_type_index in range(self.experiment_config.data_types_number):
                        if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                            required_data_number += 1
                            sum_required_data_number += 1
                            if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                received_data_number += 1
                                sum_received_data_number += 1
                                if self.action["arrival_rate"][vehicle_index][data_type_index] == 0:
                                    print("arrival_rate is zeros")
                                try:
                                    intel_arrival_time = 1 / self.action["arrival_rate"][vehicle_index][data_type_index]
                                except ZeroDivisionError:
                                    intel_arrival_time = 0

                                new_age_timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                new_timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                average_generation_time += self.last_action_time_of_sensor_nodes[vehicle_index]

                            else:
                                new_age_timeliness += 300
                                timeliness += 100
                new_age_timeliness /= required_data_number
                timeliness /= required_data_number
                try:
                    average_generation_time /= received_data_number
                    new_timeliness /= received_data_number
                except ZeroDivisionError:
                    average_generation_time = 0
                    new_timeliness = 0

                consistence_number = 0
                new_consistence_number = 0
                for vehicle_index in range(self.experiment_config.vehicle_number):
                    for data_type_index in range(self.experiment_config.data_types_number):
                        if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                            consistence_number += 1
                            if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                new_consistence_number += 1
                                new_age_consistence += np.abs(
                                        self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                                consistence += np.abs(
                                        self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                                new_consistence += np.abs(
                                        self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                            else:
                                new_age_consistence += 3000
                                consistence += 500

                if consistence_number != 0:
                    new_age_consistence /= consistence_number
                    consistence /= consistence_number
                if new_consistence_number != 0:
                    new_consistence /= new_consistence_number
                
                if required_data_number == 0 or received_data_number == 0:
                    completeness = 0
                else:
                    completeness = received_data_number / required_data_number  # the number of successfully received data divided required data
                    # print(completeness)
                if completeness > 0.8:
                    view_serviced_number += 1
                
                # timeliness and consistence should be min as they can
                # however, completeness should be max as it can
                new_age_of_view = 3 / 10 * (1 - np.tanh(new_age_timeliness / 100)) + 3 / 10 * (1 - np.tanh(new_age_consistence / 1000)) + 4 / 10 * completeness

                age_of_view = 3 / 10 * (1 - np.tanh(new_age_timeliness / 100)) + 1 / 10 * (1 - np.tanh(new_age_consistence / 1000)) + 6 / 10 * completeness

                sum_timeliness += new_age_timeliness
                sum_consistence += new_age_consistence
                sum_completeness += completeness
                sum_age_of_view += age_of_view
                sum_new_age_of_view += new_age_of_view

        if view_required_number == 0 or sum_new_age_of_view == 0:
            self.new_reward = 0
        else:
            self.new_reward = sum_new_age_of_view / view_required_number

        if view_required_number == 0 or sum_age_of_view == 0:
            self.reward = 0
        else:
            self.reward = sum_age_of_view / view_required_number

        """Computes the difference rewards"""
        sensor_nodes_reward = np.zeros(self.experiment_config.vehicle_number)

        for difference_rewards_vehicle_index in range(self.experiment_config.vehicle_number):

            difference_rewards_sum_age_of_view = 0
            difference_rewards_view_required_number = 0
            difference_rewards_view_serviced_number = 0

            for edge_view_index in range(self.experiment_config.edge_views_number):
                if self.edge_views_in_edge_node[edge_view_index][self.episode_step] == 1:
                    difference_rewards_view_required_number += 1
                    difference_rewards_received_data_number = 0
                    difference_rewards_required_data_number = 0
                    difference_rewards_average_generation_time = 0
                    difference_rewards_timeliness = 0
                    difference_rewards_new_age_timeliness = 0
                    difference_rewards_new_timeliness = 0
                    difference_rewards_new_age_consistence = 0
                    difference_rewards_consistence = 0
                    difference_rewards_new_consistence = 0
                    for vehicle_index in range(self.experiment_config.vehicle_number):
                        for data_type_index in range(self.experiment_config.data_types_number):
                            if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                                difference_rewards_required_data_number += 1
                                if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                    if vehicle_index != difference_rewards_vehicle_index:
                                        difference_rewards_received_data_number += 1
                                    if self.action["arrival_rate"][vehicle_index][data_type_index] == 0:
                                        print("arrival_rate is zeros")
                                    try:
                                        difference_rewards_intel_arrival_time = 1 / self.action["arrival_rate"][vehicle_index][data_type_index]
                                    except ZeroDivisionError:
                                        difference_rewards_intel_arrival_time = 0
                                    if vehicle_index != difference_rewards_vehicle_index:
                                        difference_rewards_new_age_timeliness += (difference_rewards_intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                        difference_rewards_timeliness += (difference_rewards_intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                    else:
                                        difference_rewards_new_age_timeliness += 300
                                        difference_rewards_timeliness += 100
                                    difference_rewards_new_timeliness += (difference_rewards_intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                    difference_rewards_average_generation_time += self.last_action_time_of_sensor_nodes[vehicle_index]

                                else:
                                    difference_rewards_new_age_timeliness += 300
                                    difference_rewards_timeliness += 100
                    difference_rewards_new_age_timeliness /= difference_rewards_required_data_number
                    difference_rewards_timeliness /= difference_rewards_required_data_number

                    if difference_rewards_received_data_number == 0:
                        difference_rewards_average_generation_time = 0
                        difference_rewards_new_timeliness = 0
                    else:
                        difference_rewards_average_generation_time /= difference_rewards_received_data_number
                        difference_rewards_new_timeliness /= difference_rewards_received_data_number                        

                    difference_rewards_consistence_number = 0
                    difference_rewards_new_consistence_number = 0
                    for vehicle_index in range(self.experiment_config.vehicle_number):
                        for data_type_index in range(self.experiment_config.data_types_number):
                            if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                                difference_rewards_consistence_number += 1
                                if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                    difference_rewards_new_consistence_number += 1
                                    if vehicle_index != difference_rewards_vehicle_index:
                                        difference_rewards_new_age_consistence += np.abs(
                                            self.last_action_time_of_sensor_nodes[vehicle_index] - difference_rewards_average_generation_time) ** 2
                                        difference_rewards_consistence += np.abs(
                                            self.last_action_time_of_sensor_nodes[vehicle_index] - difference_rewards_average_generation_time) ** 2
                                    else:
                                        difference_rewards_new_age_consistence += 3000
                                        difference_rewards_consistence += 500
                                    difference_rewards_new_consistence += np.abs(
                                            self.last_action_time_of_sensor_nodes[vehicle_index] - difference_rewards_average_generation_time) ** 2
                                else:
                                    difference_rewards_new_age_consistence += 3000
                                    difference_rewards_consistence += 500

                    if difference_rewards_consistence_number != 0:
                        difference_rewards_new_age_consistence /= difference_rewards_consistence_number
                        difference_rewards_consistence /= difference_rewards_consistence_number
                    if difference_rewards_new_consistence_number != 0:
                        difference_rewards_new_consistence /= difference_rewards_new_consistence_number
                    
                    if difference_rewards_required_data_number == 0 or difference_rewards_received_data_number == 0:
                        difference_rewards_completeness = 0
                    else:
                        difference_rewards_completeness = difference_rewards_received_data_number / difference_rewards_required_data_number  # the number of successfully received data divided required data
                        # print(completeness)
                    if difference_rewards_completeness > 0.8:
                        difference_rewards_view_serviced_number += 1
                    
                    # timeliness and consistence should be min as they can
                    # however, completeness should be max as it can
                    difference_rewards_new_age_of_view = 3 / 10 * (1 - np.tanh(difference_rewards_new_age_timeliness / 100)) + 3 / 10 * (1 - np.tanh(difference_rewards_new_age_consistence / 1000)) + 4 / 10 * difference_rewards_completeness

                    difference_rewards_age_of_view = 3 / 10 * (1 - np.tanh(difference_rewards_new_age_timeliness / 100)) + 1 / 10 * (1 - np.tanh(difference_rewards_new_age_consistence / 1000)) + 6 / 10 * difference_rewards_completeness

                    difference_rewards_sum_age_of_view += difference_rewards_age_of_view
            
            if view_required_number == 0 or sum_age_of_view == 0:
                sensor_nodes_reward[difference_rewards_vehicle_index] = 0
            else:
                sensor_nodes_reward[difference_rewards_vehicle_index] = self.reward - difference_rewards_sum_age_of_view / difference_rewards_view_required_number
        

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
        
        return self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation, sensor_nodes_reward, \
            self.reward, view_required_number, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
            sum_intel_arrival_time, sum_queuing_time, sum_transmitting_time, sum_service_time, sum_service_rate, sum_received_data_number, \
            sum_required_data_number, self.new_reward


    def step(self, action: dict):
        """
        Execute one time step within the environment
        :param action:
        :return: self.next_reward_observation, self.next_sensor_nodes_observation,
                 self.next_edge_node_observation, self.reward, self.done
        """
        self.action = action

        if self.episode_step == (self.experiment_config.max_episode_length - 1):
            self.episode_index = self.episode_index + 1
            self.done = True
        else:
            self.done = False

        # self.get_mean_and_second_moment_service_time_of_types()

        sum_age_of_view = 0
        sum_new_age_of_view = 0
        sum_timeliness = 0
        sum_consistence = 0
        sum_completeness = 0

        sum_intel_arrival_time = 0
        sum_queuing_time = 0
        sum_transmitting_time = 0
        sum_service_time = 0
        sum_service_rate = 0

        """
        When the sensor node conduct the action, update the action time
        and the average waiting time in the queue in stable system
        """
        # print("*" * 64, "\n self.episode_step: ", self.episode_step)
        max_average_waiting_time = 0
        for vehicle_index in range(self.experiment_config.vehicle_number):
            """When the action time equal to now time"""
            # print("self.next_action_time[", vehicle_index, "]", self.next_action_time_of_sensor_nodes[vehicle_index])
            if self.next_action_time_of_sensor_nodes[vehicle_index] == self.episode_step:

                for data_type_index in range(self.experiment_config.data_types_number):
                    self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = \
                        self.data_size_of_types[data_type_index]

                vehicle_action = []
                for data_type_index in range(self.experiment_config.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1:
                        priority = self.action['priority'][vehicle_index][data_type_index]
                        arrival_rate = self.action['arrival_rate'][vehicle_index][data_type_index]
                        vehicle_action.append({
                            'priority': priority, 
                            'arrival_rate': arrival_rate, 
                            'data_type': data_type_index})

                vehicle_action.sort(key=lambda value: value['priority'], reverse=True)
                # print("*" * 64, "/n", "vehicle_action: /n")
                # print(vehicle_action)

                sum_average_waiting_time = 0
                sum_average_intel_arrival_time = 0
                for index, values in enumerate(vehicle_action):
                    data_type_index = values['data_type']
                    priority = values['priority']
                    arrival_rate = values['arrival_rate']
                    work_load_before_type = 0
                    mu_before_type = 0
                    if index != 0:
                        for i in range(index):
                            work_load_before_type += vehicle_action[i]['arrival_rate'] * \
                                self.experiment_config.mean_service_time_of_types[vehicle_index][vehicle_action[i]['data_type']]
                            mu_before_type += vehicle_action[i]['arrival_rate'] * \
                                self.experiment_config.second_moment_service_time_of_types[vehicle_index][
                                vehicle_action[i]['data_type']]

                    if index != 0:
                        average_sojourn_time = 1 / (1 - work_load_before_type) * \
                                                (self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index] + 
                                                ((mu_before_type + arrival_rate * self.experiment_config.second_moment_service_time_of_types[vehicle_index][data_type_index]) 
                                                / (2 * (1 - work_load_before_type - arrival_rate *
                                                self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index]))))
                    else:
                        average_sojourn_time = 1 / 1 * \
                                                self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index]
                    
                    average_waiting_time = average_sojourn_time - self.experiment_config.mean_service_time_of_types[vehicle_index][
                        data_type_index]
                    sum_average_waiting_time += average_waiting_time
                    sum_average_intel_arrival_time += 1 / arrival_rate
                    """Update the waiting time in queue"""
                    try:
                        self.waiting_time_in_queue[vehicle_index][data_type_index] = average_waiting_time
                    except ValueError:
                        print(vehicle_index)
                        print(data_type_index)
                        print(self.waiting_time_in_queue[vehicle_index][data_type_index])
                        print(average_waiting_time)
                        print(average_sojourn_time)
                        print(self.experiment_config.mean_service_time_of_types[vehicle_index][data_type_index])
                    
                    if average_waiting_time > max_average_waiting_time:
                        max_average_waiting_time = average_waiting_time
                        
                if len(vehicle_action) != 0:
                    sum_queuing_time += (sum_average_waiting_time / len(vehicle_action))
                    sum_intel_arrival_time += (sum_average_intel_arrival_time / len(vehicle_action))
                """Update the action time"""
                self.last_action_time_of_sensor_nodes[vehicle_index] = self.action_time_of_sensor_nodes[vehicle_index]
                self.action_time_of_sensor_nodes[vehicle_index] = self.next_action_time_of_sensor_nodes[vehicle_index]
                """may raise OverflowError: cannot convert float infinity to integer"""
                # print("max_average_waiting_time: ", max_average_waiting_time)
                try:
                    self.next_action_time_of_sensor_nodes[vehicle_index] += np.ceil(max_average_waiting_time)
                except OverflowError:
                    self.next_action_time_of_sensor_nodes[vehicle_index] += 1
            # print(self.waiting_time_in_queue)
        
        """
        Update data_in_edge_node
        """
       
        for vehicle_index in range(self.experiment_config.vehicle_number):
            transmitting_time = 0
            data_type_number = 0
            for data_type_index in range(self.experiment_config.data_types_number):
                #  print("self.required_to_transmit_data_size_of_sensor_nodes[", vehicle_index, "][", data_type_index, "]: ", self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index])
                # print(self.data_types_in_vehicles[vehicle_index][data_type_index] == 1)
                if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1 and \
                    self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] > 0:
                    data_type_number += 1
                    transmission_start_time = self.last_action_time_of_sensor_nodes[vehicle_index] + \
                                                self.waiting_time_in_queue[vehicle_index][data_type_index]
                    if self.episode_step >= transmission_start_time:
                        self.data_in_edge_node[vehicle_index][data_type_index] = 0
                        SNR = self.compute_SNR(vehicle_index, self.episode_step)
                        SNR_wall = self.computer_SNR_wall_by_noise_uncertainty(
                            noise_uncertainty=np.random.uniform(low=self.experiment_config.noise_uncertainty_low_bound,
                                                                high=self.experiment_config.noise_uncertainty_up_bound))
                        if SNR <= SNR_wall:
                            self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = 0
                            self.data_in_edge_node[vehicle_index][data_type_index] = 0
                        else:
                            transmitting_time += 1
                            # print("self.bandwidth: ", self.bandwidth)
                            bandwidth = self.action['bandwidth'][0][vehicle_index] * self.bandwidth
                            # print("bandwidth: ", bandwidth)
                            transmission_bytes = self.compute_transmission_rate(SNR, bandwidth) * 1
                            # print("transmission_bytes: ", transmission_bytes)
                            self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][
                                data_type_index] -= transmission_bytes
                            # print("required_to_transmit_data_size_of_sensor_nodes[", vehicle_index, "][", data_type_index, "]: ", self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index])
                            if self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] <= 0:
                                self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index] = 0
                                # print("required_to_transmit_data_size_of_sensor_nodes[", vehicle_index, "][", data_type_index, "]: ", self.required_to_transmit_data_size_of_sensor_nodes[vehicle_index][data_type_index])
                                self.data_in_edge_node[vehicle_index][data_type_index] = self.episode_step
                                # print("data_in_edge_node[", vehicle_index, "][", data_type_index, "]: ", self.data_in_edge_node[vehicle_index][data_type_index])
            if transmitting_time != 0:
                transmitting_time /= data_type_number

            # print("transmitting_time: ", transmitting_time)

            sum_transmitting_time += transmitting_time
        sum_intel_arrival_time /= self.experiment_config.vehicle_number
        sum_queuing_time /= self.experiment_config.vehicle_number
        sum_transmitting_time /= self.experiment_config.vehicle_number

        sum_service_time += sum_intel_arrival_time + sum_queuing_time + sum_transmitting_time

        """Computes the reward"""
        view_required_number = 0
        view_serviced_number = 0
        sum_received_data_number = 0
        sum_required_data_number = 0
        for edge_view_index in range(self.experiment_config.edge_views_number):
            if self.edge_views_in_edge_node[edge_view_index][self.episode_step] == 1:
                view_required_number += 1
                received_data_number = 0
                required_data_number = 0
                average_generation_time = 0
                timeliness = 0
                new_age_timeliness = 0
                new_timeliness = 0
                new_age_consistence = 0
                consistence = 0
                new_consistence = 0
                for vehicle_index in range(self.experiment_config.vehicle_number):
                    for data_type_index in range(self.experiment_config.data_types_number):
                        if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                            required_data_number += 1
                            sum_required_data_number += 1
                            if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                received_data_number += 1
                                sum_received_data_number += 1
                                if self.action["arrival_rate"][vehicle_index][data_type_index] == 0:
                                    print("arrival_rate is zeros")
                                try:
                                    intel_arrival_time = 1 / self.action["arrival_rate"][vehicle_index][data_type_index]
                                except ZeroDivisionError:
                                    intel_arrival_time = 0
                                # print("*" * 32)
                                # print("intel_arrival_time: ", intel_arrival_time)
                                # print("data_in_edge_node: ", self.data_in_edge_node[vehicle_index][data_type_index])
                                # print("action_time_of_sensor_nodes", self.action_time_of_sensor_nodes[vehicle_index])
                                # print("last_action_time_of_sensor_nodes", self.last_action_time_of_sensor_nodes[vehicle_index])
                                # print("*" * 32)
                                new_age_timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                new_timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                average_generation_time += self.last_action_time_of_sensor_nodes[vehicle_index]
                                # if self.data_in_edge_node[vehicle_index][data_type_index] - self.action_time_of_sensor_nodes[vehicle_index] < 0:
                                #     timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.last_action_time_of_sensor_nodes[vehicle_index])
                                #     average_generation_time += self.last_action_time_of_sensor_nodes[vehicle_index]
                                # else:
                                #     timeliness += (intel_arrival_time + self.data_in_edge_node[vehicle_index][data_type_index] - self.action_time_of_sensor_nodes[vehicle_index])
                                #     average_generation_time += self.action_time_of_sensor_nodes[vehicle_index]
                            else:
                                new_age_timeliness += 300
                                timeliness += 100
                new_age_timeliness /= required_data_number
                timeliness /= required_data_number
                # print("timeliness: ", timeliness)
                try:
                    average_generation_time /= received_data_number
                    new_timeliness /= received_data_number
                except ZeroDivisionError:
                    average_generation_time = 0
                    new_timeliness = 0

                consistence_number = 0
                new_consistence_number = 0
                for vehicle_index in range(self.experiment_config.vehicle_number):
                    for data_type_index in range(self.experiment_config.data_types_number):
                        if self.view_required_data[edge_view_index][vehicle_index][data_type_index] == 1:
                            consistence_number += 1
                            if self.data_in_edge_node[vehicle_index][data_type_index] > 0:
                                new_consistence_number += 1
                                new_age_consistence += np.abs(
                                        self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                                consistence += np.abs(
                                        self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                                new_consistence += np.abs(
                                        self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                            else:
                                new_age_consistence += 3000
                                consistence += 500
                            # if self.data_in_edge_node[vehicle_index][data_type_index] - self.action_time_of_sensor_nodes[vehicle_index] < 0:
                            #     consistence += np.abs(
                            #         self.last_action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                            # else:
                            #     consistence += np.abs(
                            #         self.action_time_of_sensor_nodes[vehicle_index] - average_generation_time) ** 2
                if consistence_number != 0:
                    new_age_consistence /= consistence_number
                    consistence /= consistence_number
                if new_consistence_number != 0:
                    new_consistence /= new_consistence_number
                    # print("consistence: ", consistence)
                
                if required_data_number == 0 or received_data_number == 0:
                    completeness = 0
                else:
                    completeness = received_data_number / required_data_number  # the number of successfully received data divided required data
                    # print(completeness)
                if completeness > 0.8:
                    view_serviced_number += 1
                
                # timeliness and consistence should be min as they can
                # however, completeness should be max as it can
                new_age_of_view = 3 / 10 * (1 - np.tanh(new_age_timeliness / 100)) + 3 / 10 * (1 - np.tanh(new_age_consistence / 1000)) + 4 / 10 * completeness

                age_of_view = 3 / 10 * (1 - np.tanh(new_age_timeliness / 100)) + 1 / 10 * (1 - np.tanh(new_age_consistence / 1000)) + 6 / 10 * completeness

                sum_timeliness += new_age_timeliness
                sum_consistence += new_age_consistence
                sum_completeness += completeness
                sum_age_of_view += age_of_view
                sum_new_age_of_view += new_age_of_view

        if view_required_number == 0 or sum_new_age_of_view == 0:
            self.new_reward = 0
        else:
            self.new_reward = sum_new_age_of_view / view_required_number

        if view_required_number == 0 or sum_age_of_view == 0:
            self.reward = 0
        else:
            self.reward = sum_age_of_view / view_required_number
            # sum_age_of_view /= view_required_number
            # sum_timeliness /= view_required_number
            # sum_consistence /= view_required_number
            # sum_completeness /= view_required_number
            # sum_service_rate = view_serviced_number / view_required_number
            # print(sum_completeness)

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
            self.reward, view_required_number, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
            sum_intel_arrival_time, sum_queuing_time, sum_transmitting_time, sum_service_time, sum_service_rate, sum_received_data_number, \
            sum_required_data_number, self.new_reward

    """
    /*——————————————————————————————————————————————————————————————
        Update NN input
    —————————————————————————————————————————————————————————————--*/
    """

    def update_sensor_observation(self):
        """
        Update the input of actor network at each time slot
        """
        for vehicle_index in range(self.experiment_config.vehicle_number):
            index_start = 0
            self.sensor_nodes_observation[vehicle_index][index_start] = float(
                self.state['time']) / self.experiment_config.time_slots_number

            index_start = 1
            self.sensor_nodes_observation[vehicle_index][index_start] = self.state['action_time'][vehicle_index] / self.experiment_config.time_slots_number
            index_start += 1

            for data_type_index in range(self.experiment_config.data_types_number):
                self.sensor_nodes_observation[vehicle_index][index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.experiment_config.time_slots_number
                index_start += 1

            for data_type_index in range(self.experiment_config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:   # vehicle holds the data type
                    for edge_view_index in range(self.experiment_config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][int(self.state['time'])] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                self.sensor_nodes_observation[vehicle_index][index_start] = 1
                index_start += 1

    def update_edge_observation(self):
        index_start = 0
        self.edge_node_observation[index_start] = float(self.state['time']) / self.experiment_config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                self.edge_node_observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.experiment_config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for time_index in range(self.experiment_config.trajectories_predicted_time):
                self.edge_node_observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.experiment_config.communication_range
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.experiment_config.edge_views_number):
                        if self.state['edge_view'][edge_view_index][int(self.state['time'])] == 1:  # required edge view index at time slot 0
                            if self.state['view'][vehicle_index][data_type_index][edge_view_index] == 1:
                                self.edge_node_observation[index_start] = 1
                index_start += 1

    def update_reward_observation(self):
        index_start = 0
        self.reward_observation[index_start] = float(self.state['time']) / self.experiment_config.time_slots_number

        index_start = 1
        for vehicle_index in range(self.experiment_config.vehicle_number):
            self.reward_observation[index_start] = self.state['action_time'][vehicle_index] / self.experiment_config.time_slots_number
            index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                self.reward_observation[index_start] = float(
                    self.state['data_in_edge'][vehicle_index][data_type_index]) / self.experiment_config.time_slots_number
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for time_index in range(self.experiment_config.trajectories_predicted_time):
                self.reward_observation[index_start] = float(
                    self.state['trajectories'][vehicle_index][time_index]) / self.experiment_config.communication_range
                index_start += 1

        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                if self.state['data_types'][vehicle_index][data_type_index] == 1:  # vehicle holds the data type
                    for edge_view_index in range(self.experiment_config.edge_views_number):
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
        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.experiment_config.additive_white_gaussian_noise)
        channel_fading_gain = self.experiment_config.mean_channel_fading_gain
        # channel_fading_gain = np.random.normal(loc=self.experiment_config.mean_channel_fading_gain,
        #                                        scale=self.experiment_config.second_moment_channel_fading_gain)
        distance = self.global_trajectories[vehicle_index][time_slot]
        SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
            1 / (np.power(distance, self.experiment_config.path_loss_exponent)) * \
            VehicularNetworkEnv.cover_mW_to_W(self.experiment_config.transmission_power)
        return SNR

    def compute_SNR_by_distance(self, distance):
        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.experiment_config.additive_white_gaussian_noise)
        channel_fading_gain = self.experiment_config.mean_channel_fading_gain
        channel_fading_gain = np.random.normal(
            loc=self.experiment_config.mean_channel_fading_gain,
            scale=self.experiment_config.second_moment_channel_fading_gain)
        SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
            1 / (np.power(distance, self.experiment_config.path_loss_exponent)) * \
            VehicularNetworkEnv.cover_mW_to_W(self.experiment_config.transmission_power)
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
        mean_service_time_of_types = np.zeros(
            shape=(
                self.experiment_config.vehicle_number,
                self.experiment_config.data_types_number),
            dtype=np.float)
        second_moment_service_time_of_types = np.zeros(
            shape=(
                self.experiment_config.vehicle_number,
                self.experiment_config.data_types_number
                ),
            dtype=np.float)
        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.experiment_config.additive_white_gaussian_noise)
        
        if self.episode_step <= self.max_episode_length - 10:
            time_slot_index_list = [i for i in range(self.episode_step, self.episode_step + 10)]
        else:
            time_slot_index_list = [i for i in range(self.max_episode_length - 10, self.max_episode_length)]
        
        for vehicle_index in range(self.experiment_config.vehicle_number):
            for data_type_index in range(self.experiment_config.data_types_number):
                if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1:
                    spend_time = []
                    for time_slot_index in time_slot_index_list:
                        distance = self.global_trajectories[vehicle_index][time_slot_index]
                        channel_fading_gain = self.experiment_config.mean_channel_fading_gain
                        # channel_fading_gain = np.random.normal(
                        #         loc=self.experiment_config.mean_channel_fading_gain,
                        #         scale=self.experiment_config.second_moment_channel_fading_gain)
                        # print("channel_fading_gain: ", channel_fading_gain)
                        SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
                            (1 / np.power(distance, self.experiment_config.path_loss_exponent)) * \
                            VehicularNetworkEnv.cover_mW_to_W(self.experiment_config.transmission_power)
                        bandwidth = self.bandwidth * (1 / 10)
                        spend_time.append(self.data_size_of_types[data_type_index] / self.compute_transmission_rate(SNR, bandwidth))
                    array_spend_time = np.array(spend_time)
                    mean_service_time = array_spend_time.mean()
                    second_moment_service_time = array_spend_time.var()
                    mean_service_time_of_types[vehicle_index][data_type_index] = mean_service_time
                    second_moment_service_time_of_types[vehicle_index][data_type_index] = second_moment_service_time
                else:
                    mean_service_time_of_types[vehicle_index][data_type_index] = 0
                    second_moment_service_time_of_types[vehicle_index][data_type_index] = 0
                

        self.experiment_config.config(
            mean_service_time_of_types=mean_service_time_of_types,
            second_moment_service_time_of_types=second_moment_service_time_of_types)

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        :param mode:
        :param close:
        :return:
        """
        pass
