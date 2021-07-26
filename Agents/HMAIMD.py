# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：HMAIMD.py
@Author  ：Neardws
@Date    ：7/1/21 9:49 上午 
"""
import time

import numpy as np
import torch
import torch.nn.functional as functional
from nn_builder.pytorch.NN import NN  # construct a neural network via PyTorch
from torch import Tensor
from torch import optim

from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv
from Exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from Utilities.Data_structures.Config import AgentConfig
from Utilities.Data_structures.ExperienceReplayBuffer import ExperienceReplayBuffer
from Utilities.Data_structures.RewardReplayBuffer import RewardReplayBuffer


class HMAIMD_Agent(object):
    """
    Workflow of HMAIMD_Agent

    Step.1 Environments reset to get self.reward_observation, self.sensor_nodes_observation, self.edge_node_observation
    Step.2 sensor nodes pick actions according to self.sensor_nodes_observation
    Step.3 edge node pick action according to self.edge_node_observation plus sensor actions at step. 2
    Step.4 combine sensor nodes actions and edge node action into one global action, which type is dict
    Step.5 conduct the global action to environment, and return self.next_sensor_nodes_observation,
           self.next_edge_node_observation, self.next_reward_observation, self.reward, self.done
    Step.6 reward pick action according to self.reward_observation plus the global action
    Step.7 save replay experience
    Step.8 renew self.reward_observation, self.sensor_nodes_observation, self.edge_node_observation
           according to next parameters at step.5
    Step.9 replay step.2 - step.8

    """

    def __init__(self, agent_config=AgentConfig(), environment=VehicularNetworkEnv()):
        self.config = agent_config
        self.environment = environment
        self.hyperparameters = self.config.hyperparameters

        """boolean parameters"""
        self.done = None  # True or False indicate is episode finished
        """float parameters"""
        self.reward = None
        """dict() parameters"""
        self.action = None
        """torch.Tensor parameters, save to replay buffer"""
        self.last_reward_observation = None
        self.last_global_action = None  # Combine the Tensor sensor nodes action and edge node action
        self.last_reward_action = None
        self.reward_observation = None
        self.global_action = None
        self.reward_action = None

        self.sensor_nodes_observation = None
        self.edge_node_observation = None

        self.sensor_nodes_action = None
        self.edge_node_action = None

        self.sensor_nodes_reward = None
        self.edge_node_reward = None

        self.next_sensor_nodes_observation = None
        self.next_edge_node_observation = None
        self.next_reward_observation = None

        self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation = self.environment.reset()

        """
        Some parameters
        """
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")  # max score in one episode
        self.max_episode_score_seen = float("-inf")  # max score in whole episodes
        self.episode_index = 0  # episode index in whole episodes
        self.episode_step = 0  # step index in one episode
        self.device = "cuda" if self.config.use_gpu else "cpu"

        """
        ______________________________________________________________________________________________________________
        Replay Buffer and Exploration Strategy
        ______________________________________________________________________________________________________________
        """

        """Experience Replay Buffer"""
        self.experience_replay_buffer = ExperienceReplayBuffer(
            buffer_size=self.config.experience_replay_buffer_buffer_size,
            batch_size=self.config.experience_replay_buffer_batch_size,
            seed=self.config.experience_replay_buffer_seed
        )

        """Reward Replay Buffer"""
        self.reward_replay_buffer = RewardReplayBuffer(
            buffer_size=self.config.reward_replay_buffer_buffer_size,
            batch_size=self.config.reward_replay_buffer_batch_size,
            seed=self.config.reward_replay_buffer_seed
        )

        """Init input and output size of neural network"""
        self.sensor_observation_size = self.environment.get_sensor_observation_size()
        self.sensor_action_size = self.environment.get_sensor_action_size()
        self.critic_size_for_sensor = self.environment.get_critic_size_for_sensor()

        self.edge_observation_size = self.environment.get_actor_input_size_for_edge()
        self.edge_action_size = self.environment.get_edge_action_size()
        self.critic_size_for_edge = self.environment.get_critic_size_for_edge()

        self.reward_state_size = self.environment.get_actor_input_size_for_reward()
        self.reward_action_size = self.environment.get_reward_action_size()
        self.critic_size_for_reward = self.environment.get_critic_size_for_reward()

        """Exploration Strategy"""
        self.sensor_exploration_strategy = OU_Noise_Exploration(size=self.sensor_action_size,
                                                                hyperparameters=self.config,
                                                                key_to_use="Actor_of_Sensor")

        self.edge_exploration_strategy = OU_Noise_Exploration(size=self.edge_action_size,
                                                              hyperparameters=self.config,
                                                              key_to_use="Actor_of_Edge")

        self.reward_exploration_strategy = OU_Noise_Exploration(size=self.reward_action_size,
                                                                hyperparameters=self.config,
                                                                key_to_use="Actor_of_Reward")

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

        self.actor_local_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.sensor_observation_size,
                output_dim=[self.environment.config.data_types_number, self.environment.config.data_types_number],
                key_to_use="Actor_of_Sensor"
            ) for _ in range(self.environment.config.vehicle_number)
        ]

        self.actor_target_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.sensor_observation_size,
                output_dim=[self.environment.config.data_types_number, self.environment.config.data_types_number],
                key_to_use="Actor_of_Sensor"
            ) for _ in range(self.environment.config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.config.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_sensor_nodes[vehicle_index],
                                         to_model=self.actor_target_of_sensor_nodes[vehicle_index])

        """
        optim.Adam()
        params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) – learning rate (default: 1e-3)
        betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its 
        square (default: (0.9, 0.999))
        eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional) – whether to use the AMSGrad variant of this algorithm from the paper 
        On the Convergence of Adam and Beyond (default: False)
        """
        self.actor_optimizer_of_sensor_nodes = [
            optim.Adam(params=self.actor_local_of_sensor_nodes[vehicle_index].parameters(),
                       lr=self.hyperparameters["Actor_of_Sensor"]["learning_rate"],
                       eps=1e-4
                       ) for vehicle_index in range(self.environment.config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.config.vehicle_number):
            """
            optimizer (Optimizer) – Wrapped optimizer.
            mode (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped 
            decreasing; 
            in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            
            factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            patience (int) – Number of epochs with no improvement after which learning rate will be reduced. 
            For example, 
            if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR 
            after the 3rd epoch if the loss still has not improved then. Default: 10.
            
            threshold (float) – Threshold for measuring the new optimum, to only focus on significant changes. 
            Default: 1e-4.
            threshold_mode (str) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) 
            in ‘max’ mode 
            or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode 
            or best - threshold in min mode. Default: ‘rel’.
            
            cooldown (int) – Number of epochs to wait before resuming normal operation after lr has been reduced. 
            Default: 0.
            min_lr (float or list) – A scalar or a list of scalars. A lower bound on the learning rate of 
            all param groups or each group respectively. Default: 0.
            eps (float) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, 
            the update is ignored. Default: 1e-8.
            verbose (bool) – If True, prints a message to stdout for each update. Default: False.
            """
            optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer_of_sensor_nodes[vehicle_index], mode='min',
                                                 factor=0.1, patience=10, verbose=False, threshold=0.0001,
                                                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        """Critic Network of Sensor Nodes"""

        self.critic_local_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.critic_size_for_sensor,
                output_dim=1,
                key_to_use="Critic_of_Sensor"
            ) for _ in range(self.environment.config.vehicle_number)
        ]

        self.critic_target_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.critic_size_for_sensor,
                output_dim=1,
                key_to_use="Critic_of_Sensor"
            ) for _ in range(self.environment.config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.config.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_sensor_nodes[vehicle_index],
                                         to_model=self.critic_target_of_sensor_nodes[vehicle_index])

        self.critic_optimizer_of_sensor_nodes = [
            optim.Adam(params=self.critic_local_of_sensor_nodes[vehicle_index].parameters(),
                       lr=self.hyperparameters["Critic_of_Sensor"]["learning_rate"],
                       eps=1e-4
                       ) for vehicle_index in range(self.environment.config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.config.vehicle_number):
            optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer_of_sensor_nodes[vehicle_index], mode='min',
                                                 factor=0.1, patience=10, verbose=False, threshold=0.0001,
                                                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        """Actor Network for Edge Node"""

        self.actor_local_of_edge_node = self.create_nn(
            input_dim=self.edge_observation_size,
            output_dim=self.edge_action_size,
            key_to_use="Actor_of_Edge"
        )

        self.actor_target_of_edge_node = self.create_nn(
            input_dim=self.edge_observation_size,
            output_dim=self.edge_action_size,
            key_to_use="Actor_of_Edge"
        )

        HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_edge_node,
                                     to_model=self.actor_target_of_edge_node)

        self.actor_optimizer_of_edge_node = optim.Adam(
            params=self.actor_local_of_edge_node.parameters(),
            lr=self.hyperparameters["Actor_of_Edge"]["learning_rate"],
            eps=1e-4
        )

        optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer_of_edge_node, mode='min', factor=0.1,
                                             patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
                                             cooldown=0, min_lr=0, eps=1e-08)

        """Critic Network for Edge Node"""

        self.critic_local_of_edge_node = self.create_nn(
            input_dim=self.critic_size_for_edge,
            output_dim=1,
            key_to_use="Critic_of_Edge"
        )

        self.critic_target_of_edge_node = self.create_nn(
            input_dim=self.critic_size_for_edge,
            output_dim=1,
            key_to_use="Critic_of_Edge"
        )

        HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_edge_node,
                                     to_model=self.critic_target_of_edge_node)

        self.critic_optimizer_of_edge_node = optim.Adam(
            params=self.critic_local_of_edge_node.parameters(),
            lr=self.hyperparameters["Critic_of_Edge"]["learning_rate"],
            eps=1e-4
        )

        optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer_of_edge_node, mode='min', factor=0.1,
                                             patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
                                             cooldown=0, min_lr=0, eps=1e-08)

        """Actor Network for Reward Function"""

        self.actor_local_of_reward_function = self.create_nn(
            input_dim=self.reward_state_size,
            output_dim=self.reward_action_size,
            key_to_use="Actor_of_Reward"
        )

        self.actor_target_of_reward_function = self.create_nn(
            input_dim=self.reward_state_size,
            output_dim=self.reward_action_size,
            key_to_use="Actor_of_Reward"
        )

        HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_reward_function,
                                     to_model=self.actor_target_of_reward_function)

        self.actor_optimizer_of_reward_function = optim.Adam(
            params=self.actor_local_of_reward_function.parameters(),
            lr=self.hyperparameters["Actor_of_Reward"]["learning_rate"],
            eps=1e-4
        )

        optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer_of_reward_function, mode='min', factor=0.1,
                                             patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
                                             cooldown=0, min_lr=0, eps=1e-08)

        """Critic Network for Reward Function"""

        self.critic_local_of_reward_function = self.create_nn(
            input_dim=self.critic_size_for_reward,
            output_dim=1,
            key_to_use="Critic_of_Reward"
        )

        self.critic_target_of_reward_function = self.create_nn(
            input_dim=self.critic_size_for_reward,
            output_dim=1,
            key_to_use="Critic_of_Reward"
        )

        HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_reward_function,
                                     to_model=self.critic_target_of_reward_function)

        self.critic_optimizer_of_reward_function = optim.Adam(
            params=self.critic_local_of_reward_function.parameters(),
            lr=self.hyperparameters["Critic_of_Reward"]["learning_rate"],
            eps=1e-4
        )

        optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer_of_reward_function, mode='min', factor=0.1,
                                             patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
                                             cooldown=0, min_lr=0, eps=1e-08)

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

    def create_nn(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """
        Creates a neural network for the agents to use
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param key_to_use:
        :param override_seed:
        :param hyperparameters:
        :return:
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        if key_to_use:
            hyperparameters = hyperparameters[key_to_use]
        if override_seed:
            seed = override_seed
        else:
            seed = self.config.nn_seed

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
               - layers_info: List of integers to indicate the width and number of linear layers you want in your 
               network,
                 e.g. [5, 8, 1] would produce a network with 3 linear layers of width 5, 8 and then 1
               - hidden_activations: String or list of string to indicate the activations you want used on the output 
               of hidden layers (not including the output layer). Default is ReLU.
               - output_activation: String to indicate the activation function you want the output to go through.
                Provide a list of strings if you want multiple output heads
               - dropout: Float to indicate what dropout probability you want applied after each hidden layer
               - initialiser: String to indicate which initialiser you want used to initialise all the parameters. 
               All PyTorch initialisers are supported. PyTorch's default initialisation is the default.
               - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer
               . Default is False
               - columns_of_data_to_be_embedded: List to indicate the columns numbers of the data that you want to 
               be put through an embedding layer before being fed through the other layers of the network. 
               Default option is no embeddings
               - embedding_dimensions: If you have categorical variables you want embedded before flowing 
               through the network then you specify the embedding dimensions here with a list like so: 
               [ [embedding_input_dim_1, embedding_output_dim_1],
               [embedding_input_dim_2, embedding_output_dim_2] ...]. Default is no embeddings
               - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range 
               you want to restrict the output values to in regression tasks. Default is no range restriction
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

    """
    ______________________________________________________________________________________________________________
    Create Neural Network for Actor and Critic Network End
    ______________________________________________________________________________________________________________
    """

    """
    Workflow of each step of HMAIMD_Agent
    No.1 If action time of sensor node equal to now time , 
         then Actor of each sensor node to pick an action according to sensor observation
    No.2 Actor of edge node to pick an action according to edge observation and action of sensor nodes
    No.3 Combine action of sensor nodes and edge node into one global action
    No.4 Conduct global action to environment and get return with next_state, reward and etc.
    No.5 Actor of reward function to pick an action according to global state and global action
    No.6 Save experiences into experiences replay buffer
    No.7 Save reward experiences into reward reward buffer
    No.8 If time to learn, sample from experiences
    No.9 Train each critic target network and actor target network
    
    """

    def step(self):
        """Runs a step in the game"""
        while not self.done:  # when the episode is not over
            self.sensor_nodes_pick_actions()
            self.edge_node_pick_action()
            self.combined_action()
            self.conduct_action()
            self.reward_function_pick_action()
            self.save_experience()
            self.save_reward_experience()

            if self.time_for_critic_and_actor_of_sensor_nodes_and_edge_node_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    sensor_nodes_observations, edge_node_observations, sensor_nodes_actions, edge_node_actions, \
                        sensor_nodes_rewards, edge_node_rewards, next_sensor_nodes_observations, \
                        next_edge_node_observations, dones = self.experience_replay_buffer.sample()

                    self.sensor_nodes_and_edge_node_to_learn(sensor_nodes_observations=sensor_nodes_observations,
                                                             edge_node_observations=edge_node_observations,
                                                             sensor_nodes_actions=sensor_nodes_actions,
                                                             edge_node_actions=edge_node_actions,
                                                             sensor_nodes_rewards=sensor_nodes_rewards,
                                                             edge_node_rewards=edge_node_rewards,
                                                             next_sensor_nodes_observations=next_sensor_nodes_observations,
                                                             next_edge_node_observations=next_edge_node_observations)

            if self.time_for_critic_and_actor_of_reward_function_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    last_reward_observations, last_global_actions, last_reward_actions, rewards, reward_observations, \
                    global_actions, dones = self.reward_replay_buffer.sample()
                    self.reward_function_to_learn(last_reward_observations=last_reward_observations,
                                                  last_global_actions=last_global_actions,
                                                  last_reward_actions=last_reward_actions,
                                                  rewards=rewards,
                                                  reward_observations=reward_observations,
                                                  global_actions=global_actions,
                                                  dones=dones)

            """Renew by reward function"""
            self.last_reward_observation = self.reward_observation
            self.last_global_action = self.global_action
            self.last_reward_action = self.reward_action

            """Renew by environment"""
            self.sensor_nodes_observation = self.next_sensor_nodes_observation
            self.edge_node_observation = self.next_edge_node_observation
            self.reward_observation = self.next_reward_observation

            self.episode_step += 1
        self.episode_index += 1

    def sensor_nodes_pick_actions(self):
        """Picks an action using the actor network of each sensor node
        and then adds some noise to it to ensure exploration"""
        for sensor_node_index in range(self.environment.config.vehicle_number):
            if self.environment.state["action_time"][sensor_node_index][self.episode_step] == 1:
                sensor_node_observation = self.sensor_nodes_observation[sensor_node_index, :].unsqueeze(0)
                self.actor_local_of_sensor_nodes[sensor_node_index].eval()  # set the model to evaluation state
                with torch.no_grad():  # do not compute the gradient
                    sensor_action = self.actor_local_of_sensor_nodes[sensor_node_index](sensor_node_observation)
                self.actor_local_of_sensor_nodes[sensor_node_index].train()  # set the model to training state
                sensor_action = self.sensor_exploration_strategy.perturb_action_for_exploration_purposes(
                    {"action": sensor_action.numpy()})
                for action_index in range(self.sensor_action_size):
                    self.sensor_nodes_action[sensor_node_index, action_index] = \
                        torch.from_numpy(sensor_action)[action_index]

    def edge_node_pick_action(self):
        edge_node_state = torch.cat((self.edge_node_observation,
                                     torch.flatten(self.sensor_nodes_action)), 1).unsqueeze(0)
        self.actor_local_of_edge_node.eval()
        with torch.no_grad():
            edge_action = self.actor_local_of_edge_node(edge_node_state)
        self.actor_local_of_edge_node.train()
        self.edge_node_action = torch.from_numpy(
            self.edge_exploration_strategy.perturb_action_for_exploration_purposes({"action": edge_action.numpy()})
        )

    def combined_action(self):

        self.global_action = torch.cat((torch.flatten(self.sensor_nodes_action), self.edge_node_action), dim=1)

        priority = np.zeros(shape=(self.environment.config.vehicle_number, self.environment.config.data_types_number),
                            dtype=np.float)
        arrival_rate = np.zeros(
            shape=(self.environment.config.vehicle_number, self.environment.config.data_types_number), dtype=np.float)

        for sensor_node_index in range(self.environment.config.vehicle_number):
            sensor_node_action = self.sensor_nodes_action[sensor_node_index, :]
            sensor_node_action_of_priority = \
                sensor_node_action[0:self.environment.config.data_types_number - 1]  # first data types are priority
            sensor_node_action_of_arrival_rate = \
                sensor_node_action[
                    self.environment.config.data_types_number:-1]  # second data types number are arrival rate
            for data_type_index in range(self.environment.config.data_types_number):
                if self.environment.state["data_types"][sensor_node_index][data_type_index] == 1:
                    priority[sensor_node_index][data_type_index] = sensor_node_action_of_priority[data_type_index]
                    arrival_rate[sensor_node_index][data_type_index] = \
                        float(sensor_node_action_of_arrival_rate[data_type_index]) / \
                        self.environment.config.mean_service_time_of_types[data_type_index]

        edge_nodes_bandwidth = self.edge_node_action.numpy() * self.environment.config.bandwidth

        self.action = {"priority": priority,
                       "arrival_rate": arrival_rate,
                       "edge_nodes_bandwidth": edge_nodes_bandwidth}

    def conduct_action(self):
        """Conducts an action in the environment"""
        self.next_sensor_nodes_observation, self.next_edge_node_observation, self.next_reward_observation, \
            self.reward, self.done = self.environment.step(self.action)
        self.total_episode_score_so_far += self.reward

    def reward_function_pick_action(self):
        reward_function_state = torch.cat((self.reward_observation, self.global_action), 1).unsqueeze(0)
        self.actor_local_of_reward_function.eval()
        with torch.no_grad():
            reward_function_action = self.actor_local_of_reward_function(reward_function_state)
        self.actor_local_of_reward_function.train()
        self.reward_action = torch.from_numpy(
            self.reward_exploration_strategy.perturb_action_for_exploration_purposes(
                {"action": reward_function_action.numpy()})
        )
        self.sensor_nodes_reward = self.reward * self.reward_action[:self.environment.config.vehicle_number - 1]
        self.edge_node_reward = self.reward * self.reward_action[-1]

    def save_experience(self):
        """
        sensor_nodes_observations=torch.empty(), sensor_actions=torch.empty(),
                           sensor_nodes_rewards=torch.empty(), next_sensor_nodes_observations=torch.empty()
        Saves the recent experience to the experience replay buffer
        :return: None
        """
        if self.experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        experience = \
            self.sensor_nodes_observation, self.edge_node_observation, \
            self.sensor_nodes_action, self.edge_node_action, \
            self.sensor_nodes_reward, self.edge_node_reward, \
            self.next_sensor_nodes_observation, self.next_edge_node_observation, self.done
        self.experience_replay_buffer.add_experience(*experience)

    def save_reward_experience(self):
        if self.reward_replay_buffer is None:
            raise Exception("reward_replay_buffer is None, function save_reward_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        reward_experience = \
            self.last_reward_observation, self.last_global_action, self.last_reward_action, \
            self.reward, self.reward_observation, self.global_action, self.done
        self.reward_replay_buffer.add_experience(*reward_experience)

    def time_for_critic_and_actor_of_sensor_nodes_and_edge_node_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        return len(self.experience_replay_buffer) > self.config.experience_replay_buffer_batch_size and \
            self.episode_step % self.hyperparameters["update_every_n_steps"] == 0

    def time_for_critic_and_actor_of_reward_function_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        return len(self.experience_replay_buffer) > self.config.reward_replay_buffer_batch_size and \
            self.episode_step % self.hyperparameters["update_every_n_steps"] == 0

    def sensor_nodes_and_edge_node_to_learn(self,
                                            sensor_nodes_observations: list,
                                            edge_node_observations: Tensor,
                                            sensor_nodes_actions: list,
                                            edge_node_actions: Tensor,
                                            sensor_nodes_rewards: list,
                                            edge_node_rewards: Tensor,
                                            next_sensor_nodes_observations: list,
                                            next_edge_node_observations: Tensor,
                                            dones=torch.empty()):

        """Runs a learning iteration for the critic of sensor nodes"""
        sensor_nodes_actions_next_list = []
        next_sensor_node_observations_list = []
        for sensor_node_index in range(self.environment.config.vehicle_number):
            next_sensor_node_observations_tensor = torch.cat(
                (next_sensor_nodes_observations[0][sensor_node_index, :], next_sensor_nodes_observations[1][sensor_node_index, :]), dim=0)
            for index, values in enumerate(next_sensor_nodes_observations):
                if index > 1:
                    next_sensor_node_observations_tensor = torch.cat(
                        (next_sensor_node_observations_tensor, values[sensor_node_index, :]), dim=0)

            next_sensor_node_observations_list.append(next_sensor_node_observations_tensor)

            sensor_node_action_next = self.actor_target_of_sensor_nodes[sensor_node_index](next_sensor_node_observations_tensor)
            sensor_nodes_actions_next_list.append(sensor_node_action_next)

        sensor_nodes_actions_next_tensor = torch.cat(
            (sensor_nodes_actions_next_list[0], sensor_nodes_actions_next_list[1]), dim=1)
        for index, sensor_nodes_actions_next in enumerate(sensor_nodes_actions_next_list):
            if index > 1:
                sensor_nodes_actions_next_tensor = torch.cat(
                    (sensor_nodes_actions_next_tensor, sensor_nodes_actions_next), dim=1)

        sensor_nodes_actions_tensor = torch.cat(
            (torch.flatten(sensor_nodes_actions[0]), torch.flatten(sensor_nodes_actions[1])), dim=0
        )
        for index, sensor_nodes_action in enumerate(sensor_nodes_actions):
            if index > 1:
                sensor_nodes_actions_tensor = torch.cat(
                    (sensor_nodes_actions_tensor, torch.flatten(sensor_nodes_action)), dim=0
                )

        for sensor_node_index in range(self.environment.config.vehicle_number):
            sensor_node_observations = torch.cat(
                (sensor_nodes_observations[0][sensor_node_index, :], sensor_nodes_observations[1][sensor_node_index, :]), dim=0)
            for index, sensor_nodes_observation in enumerate(sensor_nodes_observations):
                if index > 1:
                    sensor_node_observations = torch.cat(
                        (sensor_node_observations, sensor_nodes_observation[sensor_node_index, :]), dim=0)

            sensor_node_rewards = torch.cat(
                (sensor_nodes_rewards[0][sensor_node_index, :], sensor_nodes_rewards[1][sensor_node_index, :]), dim=0)
            for index, sensor_nodes_reward in enumerate(sensor_nodes_rewards):
                if index > 1:
                    sensor_node_rewards = torch.cat(
                        (sensor_node_rewards, sensor_nodes_reward[sensor_node_index, :]), dim=0)

            next_sensor_node_observations = next_sensor_node_observations_list[sensor_node_index]

            """Runs a learning iteration for the critic"""
            """Computes the loss for the critic"""
            with torch.no_grad():
                critic_targets_next_of_sensor_node = self.critic_target_of_sensor_nodes[sensor_node_index](
                    torch.cat(next_sensor_node_observations, sensor_nodes_actions_next_tensor),
                    dim=1)  # dim=1 indicate joint as row
                critic_targets_of_sensor_node = sensor_node_rewards + (
                        self.hyperparameters["discount_rate"] * critic_targets_next_of_sensor_node * (1.0 - dones))
            critic_expected_of_sensor_node = self.critic_local_of_sensor_nodes[sensor_node_index](
                torch.cat((sensor_node_observations, sensor_nodes_actions_tensor), dim=1))
            critic_loss_of_sensor_node: Tensor = functional.mse_loss(critic_expected_of_sensor_node,
                                                                     critic_targets_of_sensor_node)

            """Update target critic networks"""
            self.take_optimisation_step(self.critic_optimizer_of_sensor_nodes[sensor_node_index],
                                        self.critic_local_of_sensor_nodes[sensor_node_index],
                                        critic_loss_of_sensor_node,
                                        self.hyperparameters["Critic_of_Sensor"]["gradient_clipping_norm"])
            self.soft_update_of_target_network(self.critic_local_of_sensor_nodes[sensor_node_index],
                                               self.critic_target_of_sensor_nodes[sensor_node_index],
                                               self.hyperparameters["Critic_of_Sensor"]["tau"])

            """Runs a learning iteration for the actor"""

            """Calculates the loss for the actor"""
            actions_predicted_of_sensor_node = self.actor_local_of_sensor_nodes[sensor_node_index](
                sensor_node_observations)

            sensor_nodes_actions_add_actions_pred = []
            for index, sensor_nodes_action in enumerate(sensor_nodes_actions):
                sensor_nodes_action[sensor_node_index, :] = actions_predicted_of_sensor_node[index]
                sensor_nodes_actions_add_actions_pred.append(torch.flatten(sensor_nodes_action))

            sensor_nodes_actions_add_actions_pred_tensor = torch.cat(
                (sensor_nodes_actions_add_actions_pred[0], sensor_nodes_actions_add_actions_pred[1]), dim=0
            )
            for index, values in enumerate(sensor_nodes_actions_add_actions_pred):
                if index > 1:
                    sensor_nodes_actions_add_actions_pred_tensor = torch.cat(
                        (sensor_nodes_actions_add_actions_pred_tensor, values), dim=0
                    )

            actor_loss_of_sensor_node = -self.critic_local_of_sensor_nodes[sensor_node_index](
                torch.cat((sensor_node_observations, sensor_nodes_actions_add_actions_pred_tensor), dim=1)).mean()

            self.take_optimisation_step(self.actor_optimizer_of_sensor_nodes[sensor_node_index],
                                        self.actor_local_of_sensor_nodes[sensor_node_index],
                                        actor_loss_of_sensor_node,
                                        self.hyperparameters["Actor_of_Sensor"]["gradient_clipping_norm"])
            self.soft_update_of_target_network(self.actor_local_of_sensor_nodes[sensor_node_index],
                                               self.actor_target_of_sensor_nodes[sensor_node_index],
                                               self.hyperparameters["Actor_of_Sensor"]["tau"])

        """Runs a learning iteration for the critic of edge node"""
        """Computes the loss for the critic"""
        with torch.no_grad():
            """Computes the critic target values to be used in the loss for the critic"""
            actions_next_of_edge_node = self.actor_target_of_edge_node(
                torch.cat((next_edge_node_observations, sensor_nodes_actions_next_tensor), dim=1))
            critic_targets_next_of_edge_node = self.critic_target_of_edge_node(
                torch.cat((next_edge_node_observations, sensor_nodes_actions_next_tensor, actions_next_of_edge_node),
                          dim=1))
            critic_targets_of_edge_node = edge_node_rewards + (
                    self.hyperparameters["discount_rate"] * critic_targets_next_of_edge_node * (1.0 - dones))

        critic_expected_of_edge_node = self.critic_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_nodes_actions_tensor, edge_node_actions), dim=1))
        loss_of_edge_node = functional.mse_loss(critic_expected_of_edge_node, critic_targets_of_edge_node)

        self.take_optimisation_step(self.critic_optimizer_of_edge_node,
                                    self.critic_local_of_edge_node,
                                    loss_of_edge_node,
                                    self.hyperparameters["Critic_of_Edge"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_of_edge_node, self.critic_target_of_edge_node,
                                           self.hyperparameters["Critic_of_Edge"]["tau"])

        """Runs a learning iteration for the actor of edge node"""

        """Calculates the loss for the actor"""
        actions_predicted_of_edge_node = self.actor_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_nodes_actions_tensor), dim=1))
        actor_loss_of_edge_node = -self.critic_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_nodes_actions_tensor, actions_predicted_of_edge_node), dim=1)).mean()

        self.take_optimisation_step(self.actor_optimizer_of_edge_node, self.actor_local_of_edge_node,
                                    actor_loss_of_edge_node,
                                    self.hyperparameters["Actor_of_Edge"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local_of_edge_node, self.actor_target_of_edge_node,
                                           self.hyperparameters["Actor_of_Edge"]["tau"])

    def reward_function_to_learn(self,
                                 last_reward_observations: Tensor,
                                 last_global_actions: Tensor,
                                 last_reward_actions: Tensor,
                                 rewards: Tensor,
                                 reward_observations: Tensor,
                                 global_actions: Tensor,
                                 dones: Tensor):

        """Runs a learning iteration for the critic of reward function"""
        with torch.no_grad():
            reward_actions_next = self.actor_target_of_reward_function(
                torch.cat((reward_observations, global_actions), dim=1))
            critic_targets_next = self.critic_target_of_reward_function(
                torch.cat((reward_observations, global_actions, reward_actions_next), 1))
            critic_targets = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        critic_expected = self.critic_local_of_reward_function(
            torch.cat((last_reward_observations, last_global_actions, last_reward_actions), dim=1))
        loss = functional.mse_loss(critic_expected, critic_targets)
        self.take_optimisation_step(self.critic_optimizer_of_reward_function,
                                    self.critic_local_of_reward_function, loss,
                                    self.hyperparameters["Critic_of_Reward"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_of_reward_function, self.critic_target_of_reward_function,
                                           self.hyperparameters["Critic_of_Reward"]["tau"])

        """Runs a learning iteration for the actor"""

        """Calculates the loss for the actor"""
        actions_predicted = self.actor_local_of_reward_function(
            torch.cat((last_reward_observations, last_global_actions), dim=1))
        actor_loss = -self.critic_local_of_reward_function(
            torch.cat((last_reward_observations, last_global_actions, actions_predicted), dim=1)).mean()
        self.take_optimisation_step(self.actor_optimizer_of_reward_function, self.actor_local_of_reward_function,
                                    actor_loss,
                                    self.hyperparameters["Actor_of_Reward"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local_of_reward_function, self.actor_target_of_reward_function,
                                           self.hyperparameters["Actor_of_Reward"]["tau"])

    @staticmethod
    def take_optimisation_step(optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()  # this applies the gradients

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau):
        """
        Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training
        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def run_n_episodes(self, num_episodes=None):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.environment.config.num_episodes_to_run
        start = time.time()
        while self.episode_index < num_episodes:
            self.reset_game()
            self.step()

            """Saves the result of an episode of the game"""
            self.game_full_episode_scores.append(self.total_episode_score_so_far)
            self.rolling_results.append(
                np.mean(self.game_full_episode_scores[-1 * self.environment.config.rolling_score_window:]))

            """Updates the best episode result seen so far"""
            if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
                self.max_episode_score_seen = self.game_full_episode_scores[-1]

            if self.rolling_results[-1] > self.max_rolling_score_seen:
                if len(self.rolling_results) > self.environment.config.rolling_score_window:
                    self.max_rolling_score_seen = self.rolling_results[-1]

        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.environment_seed)

        """float parameters"""
        self.reward = None
        self.done = None  # 1 or 0 indicate is episode finished
        """dict() parameters"""
        self.action = None
        """torch.Tensor parameters"""
        self.last_reward_observation = None
        self.last_global_action = None  # Combine the Tensor sensor nodes action and edge node action
        self.last_reward_action = None
        self.reward_observation = None
        self.global_action = None
        self.reward_action = None

        self.sensor_nodes_action = None
        self.edge_node_action = None
        self.sensor_nodes_reward = None
        self.edge_node_reward = None
        self.next_sensor_nodes_observation = None
        self.next_edge_node_observation = None
        self.next_reward_observation = None

        self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation = self.environment.reset()

        self.total_episode_score_so_far = 0
        self.episode_step = 0
        self.sensor_exploration_strategy.reset()
        self.edge_exploration_strategy.reset()
        self.reward_exploration_strategy.reset()
