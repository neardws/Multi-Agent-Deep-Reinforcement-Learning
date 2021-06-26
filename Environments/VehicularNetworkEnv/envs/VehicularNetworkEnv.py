# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：VehicularNetworkEnv.py
@Author  ：Neardws
@Date    ：6/22/21 4:09 下午 
"""
import gym
import numpy as np
from gym import spaces
from Utilities.Data_structures.Config import Experiment_Config


class VehicularNetworkEnv(gym.Env):
    """Vehicular Network Environment that follows gym interface"""
    metadata = {'render.modes': []}

    def __init__(self, experiment_config=Experiment_Config()):
        """
        init the environment via Experiment_Config
        :param experiment_config:
        """
        super(VehicularNetworkEnv, self).__init__()
        assert experiment_config is not None

        self.episode_number = experiment_config.episode_number
        self.max_episode_length = experiment_config.max_episode_length

        self.state = None  # global state
        self.observation_state = None  # individually observation state for sensor node
        self.reward = None  # external reward
        self.next_state = None
        self.done = False
        self.episode_steps = 0

        self.vehicular_number = experiment_config.vehicle_number
        self.data_types_number = experiment_config.data_types_number
        self.time_slots_number = experiment_config.time_slots_number

        self.arrival_rate_low_bound = experiment_config.arrival_rate_low_bound
        self.arrival_rate_high_bound = experiment_config.arrival_rate_high_bound
        self.action_time_of_sensor_nodes = experiment_config.action_time_of_sensor_nodes

        self.data_types_in_vehicles = experiment_config.data_types_in_vehicles
        self.edge_views_in_edge_node = experiment_config.edge_views_in_edge_node
        self.view_required_data = experiment_config.view_required_data

        self.communication_range = experiment_config.communication_range

        """
        Define action and observation space
        They must be gym.spaces objects
        Action:
            top-level controller:
                [priority, arrival rate] of all sensor node
            bottom-level controller：
                [bandwidth] of all sensor node
        """
        self.action_space = spaces.Dict({
            'sensor_nodes': spaces.Dict({
                'priority': spaces.Box(low=0, high=1,
                                       shape=(self.vehicular_number, self.data_types_number),
                                       dtype=np.float32),
                'arrival_rate': spaces.Box(low=self.arrival_rate_low_bound,
                                           high=self.arrival_rate_high_bound,
                                           shape=(self.vehicular_number, self.data_types_number),
                                           dtype=np.float32)
            }),
            'edge_nodes_bandwidth': spaces.Box(low=0, high=1,    # TODO  the sum of bandwidth may exceed 1, it should be minus the average value
                                               shape=self.vehicular_number,
                                               dtype=np.float32)
        })

        """
        State:
            [Time-slot, Data_types_in_vehicles, Action_time_of_vehicles Edge_views_in_edge_node, View_required_data, Trajectories， Data_in_edge_node]
            View:
            Trajectories:
            Location:
            Bandwidth:
        """
        self.observation_space = spaces.Dict({
            'time': spaces.Discrete(int(self.time_slots_number)),
            'data_types': spaces.MultiBinary(list(self.data_types_in_vehicles.shape)), # the MultiBinary require list
            'action_time': spaces.MultiBinary([self.vehicular_number, self.time_slots_number]),
            'edge_view': spaces.MultiBinary(list(self.edge_views_in_edge_node.shape)),
            'view': spaces.MultiBinary(list(self.view_required_data.shape)),
            'trajectories': spaces.Box(low=0, high=self.communication_range, shape=(self.vehicular_number, self.time_slots_number), dtype=np.float32),
            'data_in_edge': spaces.MultiBinary([self.data_types_number, self.time_slots_number])
        })


    def reset(self):
        """
        Reset the state of the environment to an initial state
        :return:
        """
        self.episode_steps = 0
        self.state = self.observe()  # global state
        self.observation_state = None  # individually observation state for sensor node
        self.reward = None  # external reward
        self.next_state = None
        self.done = False
        return self.state

    def observe(self):
        return {
            'time': self.episode_steps,
            'data_types': self.data_types_in_vehicles,
            'action_time': self.action_time_of_sensor_nodes,
            'edge_view': self.edge_views_in_edge_node,
            'view': self.view_required_data,
            'trajectories': self._get_trajectories(),
            'data_in_edge': self._get_data_in_edge()
        }

    def _get_trajectories(self):
        # TODO
        pass

    def _get_data_in_edge(self):
        # TODO
        pass

    def step(self, action):
        """
        Execute one time step within the environment
        :param action:
        :return: state, action, reward, next_state, done
        """

        pass

    


    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        :param mode:
        :param close:
        :return:
        """
        pass
