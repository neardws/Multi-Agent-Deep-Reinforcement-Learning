# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：HMAIMD.py
@Author  ：Neardws
@Date    ：7/1/21 9:49 上午 
"""
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim
from nn_builder.pytorch.NN import NN  # construct a neural network via PyTorch
from Utilities.Data_structures.Config import Agent_Config
from Utilities.Data_structures.Replay_Buffer import Replay_Buffer
from Exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv

class HMAIMD_Agent(object):
    """

    """
    def __init__(self, agent_config = Agent_Config(), environment = VehicularNetworkEnv()):
        self.config = agent_config
        self.environment = environment
        self.hyperparameters = self.config.hyperparameters

        """
        ______________________________________________________________________________________________________________
        Replay Buffer and Exploration Strategy
        ______________________________________________________________________________________________________________
        """

        """Experience Replay Buffer"""
        self.experience_replay_buffer = Replay_Buffer(buffer_size=self.config.experience_replay_buffer_buffer_size,
                                                      batch_size=self.config.experience_replay_buffer_batch_size,
                                                      seed=self.config.experience_replay_buffer_seed)

        """Reward Replay Buffer"""
        self.reward_replay_buffer = Replay_Buffer(buffer_size=self.config.reward_replay_buffer_buffer_size,
                                                  batch_size=self.config.reward_replay_buffer_batch_size,
                                                  seed=self.config.reward_replay_buffer_seed)

        """Exploration Strategy"""
        self.exploration_strategy = OU_Noise_Exploration(self.config)

        """
        ______________________________________________________________________________________________________________
        Replay Buffer and Exploration Strategy End
        ______________________________________________________________________________________________________________
        """

        """
        ______________________________________________________________________________________________________________
        Actor network and Critic network
        ______________________________________________________________________________________________________________
        """

        """Actor Network of Sensor Nodes"""
        self.actor_local_of_sensor_nodes = list()
        for index in range(self.environment.vehicle_number):
            self.actor_local_of_sensor_nodes.append(
                self.create_NN_for_actor_network_of_sensor_node(
                    input_dim=self.environment.get_sensor_observations_size(),
                    output_dim=self.environment.get_sensor_action_size()
                )
            )
        self.actor_target_of_sensor_nodes = list()
        for index in range(self.environment.vehicle_number):
            self.actor_target_of_sensor_nodes.append(
                self.create_NN_for_actor_network_of_sensor_node(
                    input_dim=self.environment.get_sensor_observations_size(),
                    output_dim=self.environment.get_sensor_action_size()
                )
            )
        for index in range(self.environment.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_sensor_nodes[index],
                                         to_model=self.actor_target_of_sensor_nodes[index])
        self.actor_of_sensor_nodes_optimizer = list()
        for index in range(self.environment.vehicle_number):
            self.actor_of_sensor_nodes_optimizer.append(
                optim.Adam(params=self.actor_local_of_sensor_nodes[index].parameters(),
                           lr=self.hyperparameters['actor_of_sensor']['learning_rate'],
                           eps=1e-4)
            )

        """Critic Network of Sensor Nodes"""
        self.critic_local_of_sensor_nodes = list()
        for index in range(self.environment.vehicle_number):
            self.critic_local_of_sensor_nodes.append(
                self.create_NN_for_critic_network_of_sensor_node(
                    input_dim=self.environment.get_critic_size_for_sensor(),
                    output_dim=1
                )
            )
        self.critic_target_of_sensor_nodes = list()
        for index in range(self.environment.vehicle_number):
            self.critic_target_of_sensor_nodes.append(
                self.create_NN_for_critic_network_of_sensor_node(
                    input_dim=self.environment.get_critic_size_for_sensor(),
                    output_dim=1
                )
            )
        for index in range(self.environment.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_sensor_nodes[index],
                                         to_model=self.critic_target_of_sensor_nodes[index])
        self.critic_of_sensor_nodes_optimizer = list()
        for index in range(self.environment.vehicle_number):
            self.critic_of_sensor_nodes_optimizer.append(
                optim.Adam(params=self.critic_local_of_sensor_nodes[index].parameters(),
                           lr=self.hyperparameters['critic_of_sensor']['learning_rate'],
                           eps=1e-4)
            )

        """Actor Network for Edge Node"""
        self.actor_local_of_edge_node = self.create_NN_for_actor_network_of_edge_node(
            input_dim=self.environment.get_actor_input_size_for_edge(),
            output_dim=self.environment.get_edge_action_size()
        )
        self.actor_target_of_edge_node = self.create_NN_for_actor_network_of_edge_node(
            input_dim=self.environment.get_actor_input_size_for_edge(),
            output_dim=self.environment.get_edge_action_size()
        )
        HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_edge_node,
                                     to_model=self.actor_target_of_edge_node)
        self.actor_of_edge_node_optimizer = optim.Adam(
            params=self.actor_local_of_edge_node.parameters(),
            lr=self.hyperparameters['actor_of_edge']['learning_rate'],
            eps=1e-4
        )

        """Critic Network for Edge Node"""
        self.critic_local_of_edge_node = self.create_NN_for_critic_network_of_edge_node(
            input_dim=self.environment.get_critic_size_for_edge(),
            output_dim=1
        )
        self.critic_target_of_edge_node = self.create_NN_for_critic_network_of_edge_node(
            input_dim=self.environment.get_critic_size_for_edge(),
            output_dim=1
        )
        HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_edge_node,
                                     to_model=self.critic_target_of_edge_node)
        self.critic_of_edge_node_optimizer = optim.Adam(
            params=self.critic_local_of_edge_node.parameters(),
            lr=self.hyperparameters['critic_of_edge_node']['learning_rate'],
            eps=1e-4
        )

        """Actor Network for Reward Function"""
        self.actor_local_of_reward_function = self.create_NN_for_actor_network_of_reward_function(
            input_dim=self.environment.get_actor_input_size_for_reward(),
            output_dim=self.environment.get_reward_action_size()
        )
        self.actor_target_of_reward_function = self.create_NN_for_actor_network_of_reward_function(
            input_dim=self.environment.get_actor_input_size_for_reward(),
            output_dim=self.environment.get_reward_action_size()
        )
        HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_reward_function,
                                     to_model=self.actor_target_of_reward_function)
        self.actor_target_of_reward_function_optimizer = optim.Adam(
            params=self.actor_local_of_reward_function.parameters(),
            lr=self.hyperparameters['actor_of_reward_function']['learning_rate'],
            eps=1e-4
        )

        """Critic Network for Reward Function"""
        self.critic_local_of_reward_function = self.create_NN_for_critic_network_of_reward_function(
            input_dim=self.environment.get_critic_size_for_reward(),
            output_dim=1
        )
        self.critic_target_of_reward_function = self.create_NN_for_critic_network_of_reward_function(
            input_dim=self.environment.get_critic_size_for_reward(),
            output_dim=1
        )
        HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_reward_function,
                                     to_model=self.critic_target_of_reward_function)
        self.critic_target_of_reward_function_optimizer = optim.Adam(
            params=self.critic_local_of_reward_function.parameters(),
            lr=self.hyperparameters['critic_of_reward_function']['learning_rate'],
            eps=1e-4
        )

        """
        ______________________________________________________________________________________________________________
        Actor network and Critic network End
        ______________________________________________________________________________________________________________
        """


    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    """
    ______________________________________________________________________________________________________________
    Create Neural Network for Actor and Critic Network
    ______________________________________________________________________________________________________________
    """

    def create_NN_for_actor_network_of_sensor_node(self, input_dim, output_dim, hyperparameters=None):   # the structure of network is different from other actor networks
        return NN(input_dim=input_dim)

    def create_NN_for_critic_network_of_sensor_node(self, input_dim, output_dim, hyperparameters=None):   # the structure of network is different from other actor networks
        return NN(input_dim=input_dim)

    def create_NN_for_actor_network_of_edge_node(self, input_dim, output_dim, hyperparameters=None):
        return NN(input_dim=input_dim)

    def create_NN_for_critic_network_of_edge_node(self, input_dim, output_dim, hyperparameters=None):
        return NN(input_dim=input_dim)

    def create_NN_for_actor_network_of_reward_function(self, input_dim, output_dim, hyperparameters=None):
        return NN(input_dim=input_dim)

    def create_NN_for_critic_network_of_reward_function(self, input_dim, output_dim, hyperparameters=None):
        return NN(input_dim=input_dim)

    """
    ______________________________________________________________________________________________________________
    Create Neural Network for Actor and Critic Network End
    ______________________________________________________________________________________________________________
    """

    def step(self):
        pass