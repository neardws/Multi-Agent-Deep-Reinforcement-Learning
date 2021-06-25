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

        vehicular_number = experiment_config.vehicle_number
        data_types_number = experiment_config.data_types_number
        time_slots_number = experiment_config.time_slots_number

        arrival_rate_low_bound = experiment_config.arrival_rate_low_bound
        arrival_rate_high_bound = experiment_config.arrival_rate_high_bound

        data_types_in_vehicles = experiment_config.data_types_in_vehicles
        edge_views_in_edge_node = experiment_config.edge_views_in_edge_node
        view_required_data = experiment_config.view_required_data

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
                                       shape=(vehicular_number, data_types_number),
                                       dtype=np.float32),
                'arrival_rate': spaces.Box(low=arrival_rate_low_bound,
                                           high=arrival_rate_high_bound,
                                           shape=(vehicular_number, data_types_number),
                                           dtype=np.float32)
            }),
            'edge_nodes_bandwidth': spaces.Box(low=0, high=1,    # TODO  the sum of bandwidth may exceed 1, it should be minus the average value
                                               shape=vehicular_number,
                                               dtype=np.float32)
        })

        """
        State:
            [Time-slot, Data_types_in_vehicles, Edge_views_in_edge_node, View_required_data , Trajectories]
            View:
            Trajectories:
            Location:
            Bandwidth:
        """
        self.observation_space = spaces.Dict({
            'time': spaces.Discrete(int(time_slots_number)),
            'data_types': spaces.MultiBinary(list(data_types_in_vehicles.shape)), # the MultiBinary require list
            'edge_view': spaces.MultiBinary(list(edge_views_in_edge_node.shape)),
            'view': spaces.MultiBinary(list(view_required_data.shape)),

        })

    def step(self, action):
        pass
    # Execute one time step within the environment...

    def reset(self):
        pass
    # Reset the state of the environment to an initial state...

    def render(self, mode='human', close=False):
        pass
    # Render the environment to the screen