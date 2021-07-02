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

class HMAIMD_Agent(object):
    """

    """
    def __init__(self, agent_config = Agent_Config()):
        self.config = agent_config

        self.global_action_size = None
        self.global_state_size = None

        """Experience Replay Buffer"""
        self.experience_replay_buffer = Replay_Buffer(buffer_size=self.config.experience_replay_buffer_buffer_size,
                                                      batch_size=self.config.experience_replay_buffer_batch_size,
                                                      seed=self.config.experience_replay_buffer_seed)

        """Reward Replay Buffer"""
        self.reward_replay_buffer = Replay_Buffer(buffer_size=self.config.reward_replay_buffer_buffer_size,
                                                  batch_size=self.config.reward_replay_buffer_batch_size,
                                                  seed=self.config.reward_replay_buffer_seed)

        """
        Actor network of sensor nodes
        Input:
            [
                time
                data_types_in_vehicle
                action_time
                edge_view_in_edge_node
                view_required_data
                data_in_edge
            ]
            dimension:
                1
                data_types_number
                time_slots_number
                
        Output:
            [data_types_number, 2]
        """
        self.actor_networks_of_sensor_nodes = list()
        for i in range(self.config.vehicle_number):
            self.actor_networks_of_sensor_nodes.append(
                self.create_NN(
                    input_dim=
                )
            )

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """
        Creates a neural network for the agents to use
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param key_to_use:
        :param override_seed:
        :param hyperparameters:
        :return:
        """
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None,
                                          "hidden_activations": "relu",
                                          "dropout": 0.0,
                                          "initialiser": "default",
                                          "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [],
                                          "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        """Creates a PyTorch neural network
           Args:
               - input_dim: Integer to indicate the dimension of the input into the network
               - layers_info: List of integers to indicate the width and number of linear layers you want in your network,
                             e.g. [5, 8, 1] would produce a network with 3 linear layers of width 5, 8 and then 1
               - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
                                     (not including the output layer). Default is ReLU.
               - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                                    strings if you want multiple output heads
               - dropout: Float to indicate what dropout probability you want applied after each hidden layer
               - initialiser: String to indicate which initialiser you want used to initialise all the parameters. All PyTorch
                              initialisers are supported. PyTorch's default initialisation is the default.
               - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default is False
               - columns_of_data_to_be_embedded: List to indicate the columns numbers of the data that you want to be put through an embedding layer
                                                 before being fed through the other layers of the network. Default option is no embeddings
               - embedding_dimensions: If you have categorical variables you want embedded before flowing through the network then
                                       you specify the embedding dimensions here with a list like so: [ [embedding_input_dim_1, embedding_output_dim_1],
                                       [embedding_input_dim_2, embedding_output_dim_2] ...]. Default is no embeddings
               - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
                          output values to in regression tasks. Default is no range restriction
               - random_seed: Integer to indicate the random seed you want to use
        """
        return NN(input_dim=input_dim,
                  layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"],
                  dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"],
                  initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"],
                  y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)