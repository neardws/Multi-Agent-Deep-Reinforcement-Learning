# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：HMAIMD.py
@Author  ：Neardws
@Date    ：7/1/21 9:49 上午 
"""
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as functional
from nn_builder.pytorch.NN import NN  # construct a neural network via PyTorch
from torch import Tensor
from torch import optim
import pandas as pd
from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv
from Exploration_strategies.Gaussian_Exploration import Gaussian_Exploration
from Config.AgentConfig import AgentConfig
from Utilities.Data_structures.ExperienceReplayBuffer import ExperienceReplayBuffer
from Utilities.Data_structures.ActorExperienceReplayBuffer import ActorExperienceReplayBuffer
from Utilities.Data_structures.RewardReplayBuffer import RewardReplayBuffer
from Utilities.Data_structures.ActorRewardReplayBuffer import ActorRewardReplayBuffer
from Utilities.FileOperator import save_obj, load_obj




np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)


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

    def __init__(self, agent_config: AgentConfig, environment: VehicularNetworkEnv):

        torch.autograd.set_detect_anomaly(True)

        self.agent_config = agent_config
        self.environment = environment
        self.hyperparameters = self.agent_config.hyperparameters

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

        self.saved_sensor_nodes_action = None
        self.saved_edge_node_action = None
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
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")  # max score in one episode
        self.max_episode_score_seen = float("-inf")  # max score in whole episodes
        self.device = "cuda:1" if self.environment.experiment_config.use_gpu else "cpu"

        """
        ______________________________________________________________________________________________________________
        Replay Buffer and Exploration Strategy
        ______________________________________________________________________________________________________________
        """

        """Experience Replay Buffer"""
        self.actor_experience_replay_buffer = ActorExperienceReplayBuffer(
            buffer_size=self.agent_config.actor_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.actor_experience_replay_buffer_batch_size,
            seed=self.agent_config.actor_experience_replay_buffer_seed,
            dropout=self.agent_config.actor_experience_replay_buffer_dropout,
            device=self.device
        )

        self.critic_experience_replay_buffer = ExperienceReplayBuffer(
            buffer_size=self.agent_config.critic_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.critic_experience_replay_buffer_batch_size,
            seed=self.agent_config.critic_experience_replay_buffer_seed,
            dropout=self.agent_config.critic_experience_replay_buffer_dropout,
            device=self.device
        )

        """Reward Replay Buffer"""
        self.actor_reward_replay_buffer = ActorRewardReplayBuffer(
            buffer_size=self.agent_config.actor_reward_replay_buffer_buffer_size,
            batch_size=self.agent_config.actor_reward_replay_buffer_batch_size,
            seed=self.agent_config.actor_reward_replay_buffer_seed,
            dropout=self.agent_config.actor_reward_replay_buffer_dropout,
            device=self.device
        )

        self.critic_reward_replay_buffer = RewardReplayBuffer(
            buffer_size=self.agent_config.critic_reward_replay_buffer_buffer_size,
            batch_size=self.agent_config.critic_reward_replay_buffer_batch_size,
            seed=self.agent_config.critic_reward_replay_buffer_seed,
            dropout=self.agent_config.critic_reward_replay_buffer_dropout,
            device=self.device
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
        self.sensor_exploration_strategy = Gaussian_Exploration(
            size=self.sensor_action_size,
            hyperparameters=self.hyperparameters,
            key_to_use="Actor_of_Sensor",
            device=self.device
        )

        self.edge_exploration_strategy = Gaussian_Exploration(
            size=self.edge_action_size,
            hyperparameters=self.hyperparameters,
            key_to_use="Actor_of_Edge",
            device=self.device
        )

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
                output_dim=self.sensor_action_size,
                key_to_use="Actor_of_Sensor"
            ) for _ in range(self.environment.experiment_config.vehicle_number)
        ]

        self.actor_target_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.sensor_observation_size,
                output_dim=self.sensor_action_size,
                key_to_use="Actor_of_Sensor"
            ) for _ in range(self.environment.experiment_config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.experiment_config.vehicle_number):
            HMAIMD_Agent.copy_model_over(
                from_model=self.actor_local_of_sensor_nodes[vehicle_index],
                to_model=self.actor_target_of_sensor_nodes[vehicle_index]
            )

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
            optim.Adam(
                params=self.actor_local_of_sensor_nodes[vehicle_index].parameters(),
                lr=self.hyperparameters["Actor_of_Sensor"]["learning_rate"],
                eps=1e-8
            ) for vehicle_index in range(self.environment.experiment_config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.experiment_config.vehicle_number):
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
            optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer_of_sensor_nodes[vehicle_index], 
                mode='min',
                factor=0.1, 
                patience=10, 
                verbose=False, 
                threshold=0.0001,
                threshold_mode='rel', 
                cooldown=0, 
                min_lr=0, 
                eps=1e-08
            )
        
        # for vehicle_index in range(self.environment.experiment_config.vehicle_number):
        #    self.actor_local_of_sensor_nodes[vehicle_index], self.actor_optimizer_of_sensor_nodes[vehicle_index] = amp.initialize(self.actor_local_of_sensor_nodes[vehicle_index], self.actor_optimizer_of_sensor_nodes[vehicle_index], opt_level="O1")

        """Critic Network of Sensor Nodes"""

        self.critic_local_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.critic_size_for_sensor,
                output_dim=1,
                key_to_use="Critic_of_Sensor"
            ) for _ in range(self.environment.experiment_config.vehicle_number)
        ]

        self.critic_target_of_sensor_nodes = [
            self.create_nn(
                input_dim=self.critic_size_for_sensor,
                output_dim=1,
                key_to_use="Critic_of_Sensor"
            ) for _ in range(self.environment.experiment_config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.experiment_config.vehicle_number):
            HMAIMD_Agent.copy_model_over(
                from_model=self.critic_local_of_sensor_nodes[vehicle_index],
                to_model=self.critic_target_of_sensor_nodes[vehicle_index]
            )

        self.critic_optimizer_of_sensor_nodes = [
            optim.Adam(
                params=self.critic_local_of_sensor_nodes[vehicle_index].parameters(),
                lr=self.hyperparameters["Critic_of_Sensor"]["learning_rate"],
                eps=1e-8
            ) for vehicle_index in range(self.environment.experiment_config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.experiment_config.vehicle_number):
            optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer_of_sensor_nodes[vehicle_index], mode='min',
                factor=0.1, patience=10, verbose=False, threshold=0.0001,
                threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        
        # for vehicle_index in range(self.environment.experiment_config.vehicle_number):
        #    self.critic_local_of_sensor_nodes[vehicle_index], self.critic_optimizer_of_sensor_nodes[vehicle_index] = amp.initialize(self.critic_local_of_sensor_nodes[vehicle_index], self.critic_optimizer_of_sensor_nodes[vehicle_index], opt_level="O1")

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

        HMAIMD_Agent.copy_model_over(
            from_model=self.actor_local_of_edge_node,
            to_model=self.actor_target_of_edge_node)

        self.actor_optimizer_of_edge_node = optim.Adam(
            params=self.actor_local_of_edge_node.parameters(),
            lr=self.hyperparameters["Actor_of_Edge"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer_of_edge_node, mode='min', factor=0.1,
            patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-08)

        # self.actor_local_of_edge_node, self.actor_optimizer_of_edge_node = amp.initialize(self.actor_local_of_edge_node, self.actor_optimizer_of_edge_node, opt_level="O1")

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

        HMAIMD_Agent.copy_model_over(
            from_model=self.critic_local_of_edge_node,
            to_model=self.critic_target_of_edge_node)

        self.critic_optimizer_of_edge_node = optim.Adam(
            params=self.critic_local_of_edge_node.parameters(),
            lr=self.hyperparameters["Critic_of_Edge"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer_of_edge_node, mode='min', factor=0.1,
            patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-08)

        # self.critic_local_of_edge_node, self.critic_optimizer_of_edge_node = amp.initialize(self.critic_local_of_edge_node, self.critic_optimizer_of_edge_node, opt_level="O1")

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

        HMAIMD_Agent.copy_model_over(
            from_model=self.actor_local_of_reward_function,
            to_model=self.actor_target_of_reward_function)

        self.actor_optimizer_of_reward_function = optim.Adam(
            params=self.actor_local_of_reward_function.parameters(),
            lr=self.hyperparameters["Actor_of_Reward"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer_of_reward_function, mode='min', factor=0.1,
            patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-08)

        # self.actor_local_of_reward_function, self.actor_optimizer_of_reward_function = amp.initialize(self.actor_local_of_reward_function, self.actor_optimizer_of_reward_function, opt_level="O1")

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

        HMAIMD_Agent.copy_model_over(
            from_model=self.critic_local_of_reward_function,
            to_model=self.critic_target_of_reward_function)

        self.critic_optimizer_of_reward_function = optim.Adam(
            params=self.critic_local_of_reward_function.parameters(),
            lr=self.hyperparameters["Critic_of_Reward"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer_of_reward_function, mode='min', factor=0.1,
            patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-08)

        # self.critic_local_of_reward_function, self.critic_optimizer_of_reward_function = amp.initialize(self.critic_local_of_reward_function, self.critic_optimizer_of_reward_function, opt_level="O1")
        
        """
        ______________________________________________________________________________________________________________
        Actor network and Critic network End
        ______________________________________________________________________________________________________________
        """
    
    def config_hyperparameters(self, hyperparameters):
        self.agent_config.config(hyperparameters=hyperparameters)
        self.hyperparameters = self.agent_config.hyperparameters

    def config_environment(self, environment):
        self.environment = environment
        self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation = self.environment.reset()

    def config_actor_target_of_sensor_nodes(self, actor_target_of_sensor_nodes):
        self.actor_target_of_sensor_nodes = actor_target_of_sensor_nodes

    def config_actor_target_of_edge_node(self, actor_target_of_edge_node):
        self.actor_target_of_edge_node = actor_target_of_edge_node

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
            seed = self.agent_config.nn_seed

        default_hyperparameter_choices = {"output_activation": None,
                                          "hidden_activations": "relu",
                                          "dropout": 0,
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
        average_actor_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)
        average_critic_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)
        average_actor_loss_of_edge_node = 0
        average_critic_loss_of_edge_node = 0
        average_actor_loss_of_reward_node = 0
        average_critic_loss_of_reward_node = 0

        number_of_actor_nodes_buffer = self.agent_config.actor_experience_replay_buffer_batch_size * self.agent_config.hyperparameters["actor_nodes_learning_updates_per_learning_session"] * (self.environment.max_episode_length / self.agent_config.hyperparameters["actor_nodes_update_every_n_steps"])
        number_of_critic_nodes_buffer = self.agent_config.critic_experience_replay_buffer_batch_size * self.agent_config.hyperparameters["critic_nodes_learning_updates_per_learning_session"] * (self.environment.max_episode_length / self.agent_config.hyperparameters["critic_nodes_update_every_n_steps"])
        number_of_actor_reward_buffer = self.agent_config.actor_reward_replay_buffer_batch_size * self.agent_config.hyperparameters["actor_reward_learning_updates_per_learning_session"] * (self.environment.max_episode_length / self.agent_config.hyperparameters["actor_reward_update_every_n_steps"])
        number_of_critic_reward_buffer = self.agent_config.critic_reward_replay_buffer_batch_size * self.agent_config.hyperparameters["critic_reward_learning_updates_per_learning_session"] * (self.environment.max_episode_length / self.agent_config.hyperparameters["critic_reward_update_every_n_steps"])

        max_buffer_number = max([number_of_actor_nodes_buffer, number_of_critic_nodes_buffer, number_of_actor_reward_buffer, number_of_critic_reward_buffer])
        
        nodes_start_episode_num = 300 * 15
        reward_start_episode_num = 300 * 15

        # nodes_start_episode_num = max_buffer_number * 1
        # reward_start_episode_num = max_buffer_number * 0.5

        during_episode_number = 1
        update_every_n_steps = 300

        # print(self.environment.episode_index + 1)
        with tqdm(total=self.environment.max_episode_length) as my_bar:

            while not self.done:  # when the episode is not over
                self.sensor_nodes_pick_actions()
                self.edge_node_pick_action()
                self.combined_action()
                self.conduct_action()
                self.reward_function_pick_action()
                self.save_actor_experience()
                self.save_critic_experience()
                self.save_actor_reward_experience()
                self.save_critic_reward_experience()
                
                if self.time_for_actor_of_sensor_nodes_and_edge_node_to_learn(nodes_start_episode_num, during_episode_number, update_every_n_steps):
                    # print("time_for_actor_of_sensor_nodes_and_edge_node_to_learn")
                    one_time_average_actor_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)
                    one_time_average_actor_loss_of_edge_node = 0

                    for _ in range(self.hyperparameters["actor_nodes_learning_updates_per_learning_session"]):
                        sensor_nodes_observations, edge_node_observations, sensor_nodes_actions, next_sensor_nodes_observations = self.actor_experience_replay_buffer.sample()

                        actor_loss_of_sensor_nodes, actor_loss_of_edge_node \
                            = self.actor_sensor_nodes_and_edge_node_to_learn(
                            sensor_nodes_observations=sensor_nodes_observations,
                            edge_node_observations=edge_node_observations,
                            sensor_nodes_actions=sensor_nodes_actions,
                            next_sensor_nodes_observations=next_sensor_nodes_observations)

                        for index in range(self.environment.experiment_config.vehicle_number):
                            one_time_average_actor_loss_of_sensor_nodes[index] += actor_loss_of_sensor_nodes[index]

                        one_time_average_actor_loss_of_edge_node += actor_loss_of_edge_node

                    for index in range(self.environment.experiment_config.vehicle_number):
                        one_time_average_actor_loss_of_sensor_nodes[index] /= self.hyperparameters[
                            "actor_nodes_learning_updates_per_learning_session"]

                    one_time_average_actor_loss_of_edge_node /= self.hyperparameters[
                        "actor_nodes_learning_updates_per_learning_session"]

                    for index in range(self.environment.experiment_config.vehicle_number):
                        average_actor_loss_of_sensor_nodes[index] += one_time_average_actor_loss_of_sensor_nodes[index]

                    average_actor_loss_of_edge_node += one_time_average_actor_loss_of_edge_node

                if self.time_for_critic_of_sensor_nodes_and_edge_node_to_learn(nodes_start_episode_num, during_episode_number, update_every_n_steps):
                    # print("time_for_critic_of_sensor_nodes_and_edge_node_to_learn")
                    one_time_average_critic_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)
                    one_time_average_critic_loss_of_edge_node = 0

                    for _ in range(self.hyperparameters["critic_nodes_learning_updates_per_learning_session"]):
                        sensor_nodes_observations, edge_node_observations, sensor_nodes_actions, edge_node_actions, \
                        sensor_nodes_rewards, edge_node_rewards, next_sensor_nodes_observations, \
                        next_edge_node_observations, dones = self.critic_experience_replay_buffer.sample()

                        critic_loss_of_sensor_nodes, critic_loss_of_edge_node \
                            = self.critic_sensor_nodes_and_edge_node_to_learn(
                            sensor_nodes_observations=sensor_nodes_observations,
                            edge_node_observations=edge_node_observations,
                            sensor_nodes_actions=sensor_nodes_actions,
                            edge_node_actions=edge_node_actions,
                            sensor_nodes_rewards=sensor_nodes_rewards,
                            edge_node_rewards=edge_node_rewards,
                            next_sensor_nodes_observations=next_sensor_nodes_observations,
                            next_edge_node_observations=next_edge_node_observations,
                            dones=dones)
                        for index in range(self.environment.experiment_config.vehicle_number):
                            one_time_average_critic_loss_of_sensor_nodes[index] += critic_loss_of_sensor_nodes[index]
                        one_time_average_critic_loss_of_edge_node += critic_loss_of_edge_node

                    for index in range(self.environment.experiment_config.vehicle_number):
                        one_time_average_critic_loss_of_sensor_nodes[index] /= self.hyperparameters[
                            "critic_nodes_learning_updates_per_learning_session"]
                    one_time_average_critic_loss_of_edge_node /= self.hyperparameters[
                        "critic_nodes_learning_updates_per_learning_session"]

                    for index in range(self.environment.experiment_config.vehicle_number):
                        average_critic_loss_of_sensor_nodes[index] += one_time_average_critic_loss_of_sensor_nodes[
                            index]
                    average_critic_loss_of_edge_node += one_time_average_critic_loss_of_edge_node

                if self.time_for_actor_of_reward_function_to_learn(reward_start_episode_num, during_episode_number, update_every_n_steps):
                    # print("time_for_actor_of_reward_function_to_learn")
                    one_time_average_actor_loss_of_reward_node = 0

                    for _ in range(self.hyperparameters["actor_reward_learning_updates_per_learning_session"]):
                        last_reward_observations, last_global_actions = self.actor_reward_replay_buffer.sample()
                        actor_loss_of_reward_node = self.actor_reward_function_to_learn(
                            last_reward_observations=last_reward_observations,
                            last_global_actions=last_global_actions)

                        one_time_average_actor_loss_of_reward_node += actor_loss_of_reward_node

                    one_time_average_actor_loss_of_reward_node /= self.hyperparameters[
                        "actor_reward_learning_updates_per_learning_session"]

                    average_actor_loss_of_reward_node += one_time_average_actor_loss_of_reward_node

                if self.time_for_critic_of_reward_function_to_learn(reward_start_episode_num, during_episode_number, update_every_n_steps):
                    # print("time_for_critic_of_reward_function_to_learn")
                    one_time_average_critic_loss_of_reward_node = 0

                    for _ in range(self.hyperparameters["critic_reward_learning_updates_per_learning_session"]):
                        last_reward_observations, last_global_actions, last_reward_actions, rewards, reward_observations, \
                        global_actions, dones = self.critic_reward_replay_buffer.sample()
                        critic_loss_of_reward_node = self.critic_reward_function_to_learn(
                            last_reward_observations=last_reward_observations,
                            last_global_actions=last_global_actions,
                            last_reward_actions=last_reward_actions,
                            rewards=rewards,
                            reward_observations=reward_observations,
                            global_actions=global_actions,
                            dones=dones)

                        one_time_average_critic_loss_of_reward_node += critic_loss_of_reward_node

                    one_time_average_critic_loss_of_reward_node /= self.hyperparameters[
                        "critic_reward_learning_updates_per_learning_session"]

                    average_critic_loss_of_reward_node += one_time_average_critic_loss_of_reward_node

                """Renew by reward function"""
                self.last_reward_observation = self.reward_observation.clone().detach()
                self.last_global_action = self.global_action.clone().detach()
                self.last_reward_action = self.reward_action.clone().detach()

                """Renew by environment"""
                self.sensor_nodes_observation = self.next_sensor_nodes_observation.clone().detach()
                self.edge_node_observation = self.next_edge_node_observation.clone().detach()
                self.reward_observation = self.next_reward_observation.clone().detach()

                my_bar.update(n=1)

        for index in range(self.environment.experiment_config.vehicle_number):
            average_actor_loss_of_sensor_nodes[index] /= \
                (self.environment.max_episode_length / self.hyperparameters["actor_nodes_update_every_n_steps"])
            average_critic_loss_of_sensor_nodes[index] /= \
                (self.environment.max_episode_length / self.hyperparameters["critic_nodes_update_every_n_steps"])
        average_actor_loss_of_edge_node /= \
            (self.environment.max_episode_length / self.hyperparameters["actor_nodes_update_every_n_steps"])
        average_critic_loss_of_edge_node /= \
            (self.environment.max_episode_length / self.hyperparameters["critic_nodes_update_every_n_steps"])
        average_actor_loss_of_reward_node /= \
            (self.environment.max_episode_length / self.hyperparameters["actor_reward_update_every_n_steps"])
        average_critic_loss_of_reward_node /= \
            (self.environment.max_episode_length / self.hyperparameters["actor_reward_update_every_n_steps"])

        return average_actor_loss_of_sensor_nodes, average_critic_loss_of_sensor_nodes, \
            average_actor_loss_of_edge_node, average_critic_loss_of_edge_node, \
            average_actor_loss_of_reward_node, average_critic_loss_of_reward_node
    

    def target_step(self):
        """Runs a step in the game"""

        with tqdm(total=self.environment.max_episode_length) as my_bar:

            while not self.done:  # when the episode is not over
                self.sensor_nodes_target_pick_actions()
                self.edge_node_target_pick_action()
                self.combined_action()
                self.conduct_action()
            
                """Renew by environment"""
                self.sensor_nodes_observation = self.next_sensor_nodes_observation.clone().detach()
                self.edge_node_observation = self.next_edge_node_observation.clone().detach()

                my_bar.update(n=1)

    """
    Sensor nodes local and target network to pick actions
    """
    def sensor_nodes_pick_actions(self):
        """
        Pick actions via local network
        Picks an action using the actor network of each sensor node
        and then adds some noise to it to ensure exploration"""
        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):
            if self.environment.next_action_time_of_sensor_nodes[sensor_node_index] == self.environment.episode_step:

                sensor_node_observation = self.sensor_nodes_observation[sensor_node_index, :].unsqueeze(0).to(
                    self.device)
                self.actor_local_of_sensor_nodes[sensor_node_index].eval()  # set the model to evaluation state
                with torch.no_grad():  # do not compute the gradient
                    sensor_action = self.actor_local_of_sensor_nodes[sensor_node_index](sensor_node_observation)
                self.actor_local_of_sensor_nodes[sensor_node_index].train()  # set the model to training state

                sensor_action_add_noise = self.sensor_exploration_strategy.perturb_action_for_exploration_purposes(
                    {"action": sensor_action})

                for action_index in range(self.sensor_action_size):
                    self.saved_sensor_nodes_action[sensor_node_index, action_index] = \
                        sensor_action_add_noise[0][action_index]

                softmax = torch.nn.Softmax(dim=0).to(self.device)
                sensor_action = torch.cat(
                    (softmax(sensor_action_add_noise[0][0:self.environment.experiment_config.data_types_number]),
                     softmax(sensor_action_add_noise[0][self.environment.experiment_config.data_types_number:
                                                        self.environment.experiment_config.data_types_number * 2])),
                    dim=-1).unsqueeze(0)

                for action_index in range(self.sensor_action_size):
                    self.sensor_nodes_action[sensor_node_index, action_index] = \
                        sensor_action[0][action_index]
    
    def sensor_nodes_target_pick_actions(self):
        """
        Pick actions via target network
        Picks an action using the actor network of each sensor node
        and then adds some noise to it to ensure exploration"""
        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):
            if self.environment.next_action_time_of_sensor_nodes[sensor_node_index] == self.environment.episode_step:

                sensor_node_observation = self.sensor_nodes_observation[sensor_node_index, :].unsqueeze(0).to(
                    self.device)
                self.actor_target_of_sensor_nodes[sensor_node_index].eval()  # set the model to evaluation state
                with torch.no_grad():  # do not compute the gradient
                    sensor_action = self.actor_target_of_sensor_nodes[sensor_node_index](sensor_node_observation)

                sensor_action_add_noise = self.sensor_exploration_strategy.perturb_action_for_exploration_purposes(
                    {"action": sensor_action})

                softmax = torch.nn.Softmax(dim=0).to(self.device)
                sensor_action = torch.cat(
                    (softmax(sensor_action_add_noise[0][0:self.environment.experiment_config.data_types_number]),
                     softmax(sensor_action_add_noise[0][self.environment.experiment_config.data_types_number:
                                                        self.environment.experiment_config.data_types_number * 2])),
                    dim=-1).unsqueeze(0)

                for action_index in range(self.sensor_action_size):
                    self.sensor_nodes_action[sensor_node_index, action_index] = \
                        sensor_action[0][action_index]

    """
    Edge node local and target network to pick actions
    """
    def edge_node_pick_action(self):
        """
        pick actions via local network
        """
        edge_node_state = torch.cat(
            (self.edge_node_observation.unsqueeze(0).to(self.device),
            torch.flatten(self.sensor_nodes_action).unsqueeze(0).to(self.device)),
         dim=1).float().to(self.device)

        self.actor_local_of_edge_node.eval()
        with torch.no_grad():
            edge_action = self.actor_local_of_edge_node(edge_node_state)
        self.actor_local_of_edge_node.train()

        edge_action_add_noise = self.edge_exploration_strategy.perturb_action_for_exploration_purposes(
            {"action": edge_action})
        self.saved_edge_node_action = edge_action_add_noise

        softmax = torch.nn.Softmax(dim=-1).to(self.device)
        edge_action = softmax(edge_action_add_noise)

        self.edge_node_action = edge_action

    def edge_node_target_pick_action(self):
        """
        pick actions via local network
        """
        edge_node_state = torch.cat(
            (self.edge_node_observation.unsqueeze(0).to(self.device),
            torch.flatten(self.sensor_nodes_action).unsqueeze(0).to(self.device)),
         dim=1).float().to(self.device)

        self.actor_target_of_edge_node.eval()
        with torch.no_grad():
            edge_action = self.actor_target_of_edge_node(edge_node_state)

        edge_action_add_noise = self.edge_exploration_strategy.perturb_action_for_exploration_purposes(
            {"action": edge_action})

        softmax = torch.nn.Softmax(dim=-1).to(self.device)
        edge_action = softmax(edge_action_add_noise)

        self.edge_node_action = edge_action

    def combined_action(self):

        self.global_action = torch.cat(
            (torch.flatten(self.sensor_nodes_action).unsqueeze(0), self.edge_node_action),
            dim=1).to(self.device)

        priority = np.zeros(shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number),
                            dtype=np.float)
        arrival_rate = np.zeros(
            shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number), dtype=np.float)

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):

            sensor_node_action = self.sensor_nodes_action[sensor_node_index, :]
            sensor_node_action_of_priority = \
                sensor_node_action[0:self.environment.experiment_config.data_types_number]  # first data types are priority
            sensor_node_action_of_arrival_rate = \
                sensor_node_action[
                self.environment.experiment_config.data_types_number:]  # second data types number are arrival rate

            for data_type_index in range(self.environment.experiment_config.data_types_number):
                if self.environment.state["data_types"][sensor_node_index][data_type_index] == 1:
                    priority[sensor_node_index][data_type_index] = sensor_node_action_of_priority[data_type_index]

                    arrival_rate[sensor_node_index][data_type_index] = \
                        float(sensor_node_action_of_arrival_rate[data_type_index]) / \
                        self.environment.experiment_config.mean_service_time_of_types[sensor_node_index][data_type_index]
                else:
                    priority[sensor_node_index][data_type_index] = 0
                    arrival_rate[sensor_node_index][data_type_index] = 0

        edge_nodes_bandwidth = self.edge_node_action.cpu().data.numpy() * self.environment.experiment_config.bandwidth

        self.action = {
            "priority": priority,
            "arrival_rate": arrival_rate,
            "bandwidth": edge_nodes_bandwidth
        }

    def conduct_action(self):
        """Conducts an action in the environment"""
        self.next_sensor_nodes_observation, self.next_edge_node_observation, self.next_reward_observation, \
            self.reward, view_required_number, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
            sum_intel_arrival_time, sum_queuing_time, sum_transmitting_time, sum_service_time, sum_service_rate, sum_received_data_number, \
            sum_required_data_number, new_reward = self.environment.step(self.action)
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

    def reward_function_pick_action(self):
        reward_function_state = torch.cat((self.reward_observation.unsqueeze(0).to(
            self.device), self.global_action.to(
            self.device)), dim=1).float().to(
            self.device)
        self.actor_local_of_reward_function.eval()
        with torch.no_grad():
            reward_function_action = self.actor_local_of_reward_function(reward_function_state)
        self.actor_local_of_reward_function.train()
        self.reward_action = reward_function_action

        self.sensor_nodes_reward = self.reward * self.reward_action[0][:self.environment.experiment_config.vehicle_number]
        self.edge_node_reward = self.reward * self.reward_action[0][-1]

        self.sensor_nodes_reward = self.sensor_nodes_reward.unsqueeze(0)
        self.edge_node_reward = self.edge_node_reward.unsqueeze(0).unsqueeze(0)


    def save_actor_experience(self):
        """
        sensor_nodes_observations=torch.empty(), sensor_actions=torch.empty(),
        sensor_nodes_rewards=torch.empty(), next_sensor_nodes_observations=torch.empty()
        Saves the recent experience to the experience replay buffer
        :return: None
        """
        if self.actor_experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        self.actor_experience_replay_buffer.add_experience(
            sensor_nodes_observation=self.sensor_nodes_observation.clone().detach(),
            edge_node_observation=self.edge_node_observation.clone().detach(),
            sensor_nodes_action=self.saved_sensor_nodes_action.clone().detach(),
            next_sensor_nodes_observation=self.next_sensor_nodes_observation.clone().detach())

    def save_critic_experience(self):
        """
        sensor_nodes_observations=torch.empty(), sensor_actions=torch.empty(),
        sensor_nodes_rewards=torch.empty(), next_sensor_nodes_observations=torch.empty()
        Saves the recent experience to the experience replay buffer
        :return: None
        """
        if self.critic_experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        self.critic_experience_replay_buffer.add_experience(
            sensor_nodes_observation=self.sensor_nodes_observation.clone().detach(),
            edge_node_observation=self.edge_node_observation.clone().detach(),
            sensor_nodes_action=self.saved_sensor_nodes_action.clone().detach(),
            edge_node_action=self.saved_edge_node_action.clone().detach(),
            sensor_nodes_reward=self.sensor_nodes_reward.clone().detach(),
            edge_node_reward=self.edge_node_reward.clone().detach(),
            next_sensor_nodes_observation=self.next_sensor_nodes_observation.clone().detach(),
            next_edge_node_observation=self.next_edge_node_observation.clone().detach(),
            done=self.done)

    def save_actor_reward_experience(self):
        if self.actor_reward_replay_buffer is None:
            raise Exception("reward_replay_buffer is None, function save_reward_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        if self.last_reward_observation is None:
            pass
        else:
            self.actor_reward_replay_buffer.add_experience(
                last_reward_observation=self.last_reward_observation.clone().detach(),
                last_global_action=self.last_global_action.clone().detach())

    def save_critic_reward_experience(self):
        if self.critic_reward_replay_buffer is None:
            raise Exception("reward_replay_buffer is None, function save_reward_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        if self.last_reward_observation is None:
            pass
        else:
            self.critic_reward_replay_buffer.add_experience(
                last_reward_observation=self.last_reward_observation.clone().detach(),
                last_global_action=self.last_global_action.clone().detach(),
                last_reward_action=self.last_reward_action.clone().detach(),
                reward=self.reward,
                reward_observation=self.reward_observation.clone().detach(),
                global_action=self.global_action.clone().detach(),
                done=self.done)

    def time_for_actor_of_sensor_nodes_and_edge_node_to_learn(self, nodes_start_episode_num, during_episode, update_every_n_steps):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if (self.environment.episode_index) >= start_episode_index:
            # if self.max_episode_score_seen >= 110:
            #     return False
            added_episode_index = self.environment.episode_index - start_episode_index
            return self.environment.episode_step % self.hyperparameters["actor_nodes_update_every_n_steps"] == 0

            # if (added_episode_index / during_episode) % 1 <= 5 /150:
            #     return self.environment.episode_step % self.hyperparameters["actor_nodes_update_every_n_steps"] == 0
            # else:
            #     return self.environment.episode_step % update_every_n_steps == 0
        else:
            return False

    def time_for_critic_of_sensor_nodes_and_edge_node_to_learn(self, nodes_start_episode_num, during_episode, update_every_n_steps):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if self.environment.episode_index >= start_episode_index:
            # if self.max_episode_score_seen >= 110:
            #     return False
            added_episode_index = self.environment.episode_index - start_episode_index
            return self.environment.episode_step % self.hyperparameters["critic_nodes_update_every_n_steps"] == 0
            # if (added_episode_index / during_episode) % 1 <= 5 / 150:
            #     return self.environment.episode_step % self.hyperparameters["critic_nodes_update_every_n_steps"] == 0
            # else:
            #     return self.environment.episode_step % update_every_n_steps == 0
        else:
            return False

    def time_for_actor_of_reward_function_to_learn(self, reward_start_episode_num, during_episode, update_every_n_steps):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = reward_start_episode_num / self.environment.experiment_config.max_episode_length
        if self.environment.episode_index >= start_episode_index:
            if self.environment.episode_index >= 500:
                return False
            added_episode_index = self.environment.episode_index - start_episode_index
            return self.environment.episode_step % self.hyperparameters["actor_reward_update_every_n_steps"] == 0
            # if (added_episode_index / during_episode) % 1 <= 5 / 150:
            #     return self.environment.episode_step % self.hyperparameters["actor_reward_update_every_n_steps"] == 0
            # else:
            #     return self.environment.episode_step % update_every_n_steps == 0
        else:
            return False

    def time_for_critic_of_reward_function_to_learn(self, reward_start_episode_num, during_episode, update_every_n_steps):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""        
        start_episode_index = reward_start_episode_num / self.environment.experiment_config.max_episode_length
        if self.environment.episode_index >= start_episode_index:
            if self.environment.episode_index >= 500:
                return False
            added_episode_index = self.environment.episode_index - start_episode_index
            return self.environment.episode_step % self.hyperparameters["actor_reward_update_every_n_steps"] == 0
            # if (added_episode_index / during_episode) % 1 <= 5 / 150:
            #     return self.environment.episode_step % self.hyperparameters["actor_reward_update_every_n_steps"] == 0
            # else:
            #     return self.environment.episode_step % update_every_n_steps == 0
        else:
            return False

    def actor_sensor_nodes_and_edge_node_to_learn(self,
                                                  sensor_nodes_observations: list,
                                                  edge_node_observations: Tensor,
                                                  sensor_nodes_actions: list,
                                                  next_sensor_nodes_observations: list):
        time_start = time.time()
        actor_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)

        """Runs a learning iteration for the critic of sensor nodes"""
        sensor_nodes_actions_next_list = []  # next action of sensor nodes according to next_sensor_nodes_observations
        next_sensor_node_observations_list = []  # next observation of single sensor node, Reorganized by next_sensor_nodes_observations

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):
            next_sensor_node_observations_tensor = torch.cat(
                (next_sensor_nodes_observations[0][sensor_node_index, :].unsqueeze(0),
                 next_sensor_nodes_observations[1][sensor_node_index, :].unsqueeze(0)), dim=0)
            for index, values in enumerate(next_sensor_nodes_observations):
                if index > 1:
                    next_sensor_node_observations_tensor = torch.cat(
                        (next_sensor_node_observations_tensor, values[sensor_node_index, :].unsqueeze(0)), dim=0)

            next_sensor_node_observations_list.append(next_sensor_node_observations_tensor)
            sensor_node_action_next = self.actor_target_of_sensor_nodes[sensor_node_index](
                next_sensor_node_observations_tensor.float().to(self.device))
            sensor_nodes_actions_next_list.append(sensor_node_action_next)

        new_sensor_nodes_actions_next_list = []  # next action of sensor nodes at each batch
        for tensor_index in range(sensor_nodes_actions_next_list[0].shape[0]):  # need 256 batch number
            new_sensor_nodes_actions_next_tensor = torch.cat(
                (sensor_nodes_actions_next_list[0][tensor_index, :].unsqueeze(0),
                 sensor_nodes_actions_next_list[1][tensor_index, :].unsqueeze(0)),
                dim=1
            )
            for index, sensor_nodes_actions_next in enumerate(sensor_nodes_actions_next_list):
                if index > 1:
                    new_sensor_nodes_actions_next_tensor = torch.cat(
                        (new_sensor_nodes_actions_next_tensor, sensor_nodes_actions_next[tensor_index].unsqueeze(0)),
                        dim=1
                    )
            new_sensor_nodes_actions_next_list.append(new_sensor_nodes_actions_next_tensor)
        sensor_nodes_actions_next_tensor = torch.cat(
            (new_sensor_nodes_actions_next_list[0], new_sensor_nodes_actions_next_list[1]), dim=0)
        for index, sensor_nodes_actions_next in enumerate(new_sensor_nodes_actions_next_list):
            if index > 1:
                sensor_nodes_actions_next_tensor = torch.cat(
                    (sensor_nodes_actions_next_tensor, sensor_nodes_actions_next), dim=0)

        sensor_nodes_actions_tensor = torch.cat(
            (torch.flatten(sensor_nodes_actions[0]).unsqueeze(0), torch.flatten(sensor_nodes_actions[1]).unsqueeze(0)),
            dim=0
        )
        for index, sensor_nodes_action in enumerate(sensor_nodes_actions):
            if index > 1:
                sensor_nodes_actions_tensor = torch.cat(
                    (sensor_nodes_actions_tensor, torch.flatten(sensor_nodes_action).unsqueeze(0)), dim=0
                )
        sensor_nodes_actions_tensor = sensor_nodes_actions_tensor.to(self.device)

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):

            sensor_node_observations = torch.cat(
                (sensor_nodes_observations[0][sensor_node_index, :].unsqueeze(0),
                 sensor_nodes_observations[1][sensor_node_index, :].unsqueeze(0)), dim=0)
            for index, sensor_nodes_observation in enumerate(sensor_nodes_observations):
                if index > 1:
                    sensor_node_observations = torch.cat(
                        (sensor_node_observations, sensor_nodes_observation[sensor_node_index, :].unsqueeze(0)), dim=0)
            sensor_node_observations = sensor_node_observations.float().to(self.device)

            """Runs a learning iteration for the actor"""

            """Calculates the loss for the actor"""
            actions_predicted_of_sensor_node = self.actor_local_of_sensor_nodes[sensor_node_index](
                sensor_node_observations)

            sensor_nodes_actions_add_actions_pred = []  # actions of other sensor node plus action that predicted by the sensor node

            for index in range(len(sensor_nodes_actions)):
                sensor_nodes_action = sensor_nodes_actions[index].clone().detach()
                sensor_nodes_action[sensor_node_index, :] = actions_predicted_of_sensor_node[index]
                sensor_nodes_actions_add_actions_pred.append(torch.flatten(sensor_nodes_action))

            sensor_nodes_actions_add_actions_pred_tensor = torch.cat(
                (sensor_nodes_actions_add_actions_pred[0].unsqueeze(0),
                 sensor_nodes_actions_add_actions_pred[1].unsqueeze(0)), dim=0
            )
            for index, values in enumerate(sensor_nodes_actions_add_actions_pred):
                if index > 1:
                    sensor_nodes_actions_add_actions_pred_tensor = torch.cat(
                        (sensor_nodes_actions_add_actions_pred_tensor, values.unsqueeze(0)), dim=0
                    )
            sensor_nodes_actions_add_actions_pred_tensor = sensor_nodes_actions_add_actions_pred_tensor.float().to(
                self.device)

            """
            ________________________________________________________________
            
            Calculates the actor loss of sensor nodes
            ________________________________________________________________
            """

            actor_loss_of_sensor_node = -self.critic_local_of_sensor_nodes[sensor_node_index](
                torch.cat((sensor_node_observations, sensor_nodes_actions_add_actions_pred_tensor), dim=1)).mean()

            actor_loss_of_sensor_nodes[sensor_node_index] = actor_loss_of_sensor_node.item()

            self.take_optimisation_step(self.actor_optimizer_of_sensor_nodes[sensor_node_index],
                                        self.actor_local_of_sensor_nodes[sensor_node_index],
                                        actor_loss_of_sensor_node,
                                        self.hyperparameters["Actor_of_Sensor"]["gradient_clipping_norm"])
            self.soft_update_of_target_network(self.actor_local_of_sensor_nodes[sensor_node_index],
                                               self.actor_target_of_sensor_nodes[sensor_node_index],
                                               self.hyperparameters["Actor_of_Sensor"]["tau"])
        time_end = time.time()

        # print("Actor of sensor nodes took: ", time_end - time_start, " seconds")
        """Runs a learning iteration for the actor of edge node"""
        time_start = time.time()
        """Calculates the loss for the actor"""
        actions_predicted_of_edge_node = self.actor_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_nodes_actions_tensor), dim=1))

        """
        ________________________________________________________________

        Calculates the actor loss of edge node
        ________________________________________________________________
        """

        loss_of_edge_node = -self.critic_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_nodes_actions_tensor, actions_predicted_of_edge_node),
                      dim=1)).mean()

        actor_loss_of_edge_node = loss_of_edge_node.item()

        self.take_optimisation_step(self.actor_optimizer_of_edge_node,
                                    self.actor_local_of_edge_node,
                                    loss_of_edge_node,
                                    self.hyperparameters["Actor_of_Edge"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local_of_edge_node, self.actor_target_of_edge_node,
                                           self.hyperparameters["Actor_of_Edge"]["tau"])
        time_end = time.time()
        # print("Actor of edge node took: ", time_end - time_start, " seconds")
        return actor_loss_of_sensor_nodes, actor_loss_of_edge_node

    def critic_sensor_nodes_and_edge_node_to_learn(self,
                                                   sensor_nodes_observations: list,
                                                   edge_node_observations: Tensor,
                                                   sensor_nodes_actions: list,
                                                   edge_node_actions: Tensor,
                                                   sensor_nodes_rewards: list,
                                                   edge_node_rewards: Tensor,
                                                   next_sensor_nodes_observations: list,
                                                   next_edge_node_observations: Tensor,
                                                   dones: Tensor):
        time_start = time.time()
        critic_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)

        """Runs a learning iteration for the critic of sensor nodes"""
        sensor_nodes_actions_next_list = []  # next action of sensor nodes according to next_sensor_nodes_observations
        next_sensor_node_observations_list = []  # next observation of single sensor node, Reorganized by next_sensor_nodes_observations

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):
            next_sensor_node_observations_tensor = torch.cat(
                (next_sensor_nodes_observations[0][sensor_node_index, :].unsqueeze(0),
                 next_sensor_nodes_observations[1][sensor_node_index, :].unsqueeze(0)), dim=0)
            for index, values in enumerate(next_sensor_nodes_observations):
                if index > 1:
                    next_sensor_node_observations_tensor = torch.cat(
                        (next_sensor_node_observations_tensor, values[sensor_node_index, :].unsqueeze(0)), dim=0)

            next_sensor_node_observations_list.append(next_sensor_node_observations_tensor)
            sensor_node_action_next = self.actor_target_of_sensor_nodes[sensor_node_index](
                next_sensor_node_observations_tensor.float().to(self.device))
            sensor_nodes_actions_next_list.append(sensor_node_action_next)

        new_sensor_nodes_actions_next_list = []  # next action of sensor nodes at each batch
        for tensor_index in range(sensor_nodes_actions_next_list[0].shape[0]):  # need 256 batch number
            new_sensor_nodes_actions_next_tensor = torch.cat(
                (sensor_nodes_actions_next_list[0][tensor_index, :].unsqueeze(0),
                 sensor_nodes_actions_next_list[1][tensor_index, :].unsqueeze(0)),
                dim=1
            )
            for index, sensor_nodes_actions_next in enumerate(sensor_nodes_actions_next_list):
                if index > 1:
                    new_sensor_nodes_actions_next_tensor = torch.cat(
                        (new_sensor_nodes_actions_next_tensor, sensor_nodes_actions_next[tensor_index].unsqueeze(0)),
                        dim=1
                    )
            new_sensor_nodes_actions_next_list.append(new_sensor_nodes_actions_next_tensor)
        sensor_nodes_actions_next_tensor = torch.cat(
            (new_sensor_nodes_actions_next_list[0], new_sensor_nodes_actions_next_list[1]), dim=0)
        for index, sensor_nodes_actions_next in enumerate(new_sensor_nodes_actions_next_list):
            if index > 1:
                sensor_nodes_actions_next_tensor = torch.cat(
                    (sensor_nodes_actions_next_tensor, sensor_nodes_actions_next), dim=0)

        sensor_nodes_actions_tensor = torch.cat(
            (torch.flatten(sensor_nodes_actions[0]).unsqueeze(0), torch.flatten(sensor_nodes_actions[1]).unsqueeze(0)),
            dim=0
        )
        for index, sensor_nodes_action in enumerate(sensor_nodes_actions):
            if index > 1:
                sensor_nodes_actions_tensor = torch.cat(
                    (sensor_nodes_actions_tensor, torch.flatten(sensor_nodes_action).unsqueeze(0)), dim=0
                )
        sensor_nodes_actions_tensor = sensor_nodes_actions_tensor.to(self.device)

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):

            sensor_node_observations = torch.cat(
                (sensor_nodes_observations[0][sensor_node_index, :].unsqueeze(0),
                 sensor_nodes_observations[1][sensor_node_index, :].unsqueeze(0)), dim=0)
            for index, sensor_nodes_observation in enumerate(sensor_nodes_observations):
                if index > 1:
                    sensor_node_observations = torch.cat(
                        (sensor_node_observations, sensor_nodes_observation[sensor_node_index, :].unsqueeze(0)), dim=0)
            sensor_node_observations = sensor_node_observations.float().to(self.device)

            sensor_node_rewards = torch.cat(
                (sensor_nodes_rewards[0][0, sensor_node_index].unsqueeze(0).unsqueeze(0),
                 sensor_nodes_rewards[1][0, sensor_node_index].unsqueeze(0).unsqueeze(0)), dim=0)
            for index, sensor_nodes_reward in enumerate(sensor_nodes_rewards):
                if index > 1:
                    sensor_node_rewards = torch.cat(
                        (sensor_node_rewards, sensor_nodes_reward[0, sensor_node_index].unsqueeze(0).unsqueeze(0)),
                        dim=0)
            sensor_node_rewards = sensor_node_rewards.float().to(self.device)

            next_sensor_node_observations: Tensor = next_sensor_node_observations_list[sensor_node_index]

            """Runs a learning iteration for the critic"""
            """Computes the loss for the critic"""
            with torch.no_grad():

                critic_targets_next_of_sensor_node = self.critic_target_of_sensor_nodes[sensor_node_index](
                    torch.cat((next_sensor_node_observations.float().to(self.device),
                               sensor_nodes_actions_next_tensor.float().to(self.device)),
                              dim=1))  # dim=1 indicate joint as row

                critic_targets_of_sensor_node = sensor_node_rewards + (
                        self.hyperparameters["discount_rate"] * critic_targets_next_of_sensor_node * (1.0 - dones))

            critic_expected_of_sensor_node = self.critic_local_of_sensor_nodes[sensor_node_index](
                torch.cat((sensor_node_observations, sensor_nodes_actions_tensor), dim=1).float().to(self.device))

            """
            ________________________________________________________________

            Calculates the critic loss of sensor nodes
            ________________________________________________________________
            """

            critic_loss_of_sensor_node = functional.mse_loss(
                critic_expected_of_sensor_node,
                critic_targets_of_sensor_node.float().to(self.device))

            critic_loss_of_sensor_nodes[sensor_node_index] = critic_loss_of_sensor_node.item()

            """Update target critic networks"""

            self.take_optimisation_step(self.critic_optimizer_of_sensor_nodes[sensor_node_index],
                                        self.critic_local_of_sensor_nodes[sensor_node_index],
                                        critic_loss_of_sensor_node,
                                        self.hyperparameters["Critic_of_Sensor"]["gradient_clipping_norm"])

            self.soft_update_of_target_network(self.critic_local_of_sensor_nodes[sensor_node_index],
                                               self.critic_target_of_sensor_nodes[sensor_node_index],
                                               self.hyperparameters["Critic_of_Sensor"]["tau"])
        time_end = time.time()
        # print("Time taken for sensor nodes critic update: ", time_end - time_start)

        time_start = time.time()
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

        """
        ________________________________________________________________

        Calculates the critic loss of edge node
        ________________________________________________________________
        """

        loss_of_edge_node = functional.mse_loss(critic_expected_of_edge_node, critic_targets_of_edge_node)
        critic_loss_of_edge_node = loss_of_edge_node.item()

        self.take_optimisation_step(self.critic_optimizer_of_edge_node,
                                    self.critic_local_of_edge_node,
                                    loss_of_edge_node,
                                    self.hyperparameters["Critic_of_Edge"]["gradient_clipping_norm"])

        self.soft_update_of_target_network(
            self.critic_local_of_edge_node, self.critic_target_of_edge_node,
            self.hyperparameters["Critic_of_Edge"]["tau"])
        time_end = time.time()
        # print("Time taken for edge critic update: ", time_end - time_start)
        return critic_loss_of_sensor_nodes, critic_loss_of_edge_node

    def actor_reward_function_to_learn(self,
                                       last_reward_observations: Tensor,
                                       last_global_actions: Tensor):

        """Runs a learning iteration for the actor"""

        """Calculates the loss for the actor"""
        actions_predicted = self.actor_local_of_reward_function(
            torch.cat((last_reward_observations, last_global_actions), dim=1))

        """
        ________________________________________________________________

        Calculates the actor loss of reward node
        ________________________________________________________________
        """

        actor_loss = -self.critic_local_of_reward_function(
            torch.cat((last_reward_observations, last_global_actions, actions_predicted), dim=1)).mean()

        self.take_optimisation_step(self.actor_optimizer_of_reward_function,
                                    self.actor_local_of_reward_function,
                                    actor_loss,
                                    self.hyperparameters["Actor_of_Reward"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local_of_reward_function, self.actor_target_of_reward_function,
                                           self.hyperparameters["Actor_of_Reward"]["tau"])

        return actor_loss.item()

    def critic_reward_function_to_learn(self,
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

        """
        ________________________________________________________________

        Calculates the critic loss of reward node
        ________________________________________________________________
        """
        loss = functional.mse_loss(critic_expected, critic_targets)

        self.take_optimisation_step(self.critic_optimizer_of_reward_function,
                                    self.critic_local_of_reward_function, loss,
                                    self.hyperparameters["Critic_of_Reward"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_of_reward_function, self.critic_target_of_reward_function,
                                           self.hyperparameters["Critic_of_Reward"]["tau"])

        return loss.item()

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
                                               max_norm=clipping_norm,
                                               norm_type=2,
                                               error_if_nonfinite=False)  # clip gradients to help stabilise training
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
    
    def run_n_episodes_as_results(self, num_episodes, result_name, environment, actor_nodes_name, actor_edge_name):
        
        self.config_actor_target_of_sensor_nodes(load_obj(name=actor_nodes_name))
        self.config_actor_target_of_edge_node(load_obj(name=actor_edge_name))

        if environment is not None:
            self.config_environment(environment)

        try:
            result_data = pd.read_csv(
                result_name, 
                names=["Epoch index", "age_of_view", "new_age_of_view", "timeliness", "consistence", "completeness", "intel_arrival_time",  "queuing_time", "transmitting_time", "service_time", "service_rate", "received_data", "required_data"], 
                header=0)
        except FileNotFoundError:
            result_data = pd.DataFrame(
                data=None, 
                columns={
                    "Epoch index": "", 
                    "age_of_view": "", 
                    "new_age_of_view": "", 
                    "timeliness": "", 
                    "consistence": "", 
                    "completeness": "",
                    "intel_arrival_time": "", 
                    "queuing_time": "", 
                    "transmitting_time": "", 
                    "service_time": "", 
                    "service_rate": "", 
                    "received_data": "", 
                    "required_data": ""},
                index=[0])

        for i in range(num_episodes):
            print("*" * 64)
            self.reset_game()
            self.target_step()

            self.total_episode_timeliness_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_consistence_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_completeness_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_intel_arrival_time /= self.environment.experiment_config.max_episode_length
            self.total_episode_queuing_time_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_transmitting_time_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_service_time_so_far /= self.environment.experiment_config.max_episode_length

            print("Epoch index: ", i)
            print("Total reward: ", self.total_episode_score_so_far)
            print("new_age_of_view: ", self.new_total_episode_score_so_far)
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


    def run_n_episodes(self, num_episodes=None, temple_agent_config_name=None, temple_agent_name=None, 
        temple_result_name=None, temple_loss_name=None, actor_nodes_name=None, actor_edge_name=None):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.environment.experiment_config.episode_number

        try:
            result_data = pd.read_csv(
                temple_result_name, 
                names=["Epoch index", "age_of_view", "new_age_of_view", "timeliness", "consistence", "completeness", "intel_arrival_time", "queuing_time", "transmitting_time", "service_time", "service_rate", "received_data", "required_data"], 
                header=0)
            loss_data = pd.read_csv(temple_loss_name, names=["Epoch index",
                                                             "Actor of V1", "Actor of V2", "Actor of V3",
                                                             "Actor of V4", "Actor of V5", "Actor of V6",
                                                             "Actor of V7", "Actor of V8", "Actor of V9",
                                                             "Actor of V10",
                                                             "Critic of V1", "Critic of V2", "Critic of V3",
                                                             "Critic of V4", "Critic of V5", "Critic of V6",
                                                             "Critic of V7", "Critic of V8", "Critic of V9",
                                                             "Critic of V10",
                                                             "Actor of Edge", "Critic of Edge",
                                                             "Actor of Reward", "Critic of Reward"], header=0)
        except FileNotFoundError:
            result_data = pd.DataFrame(
                data=None, 
                columns={
                    "Epoch index": "", 
                    "age_of_view": "", 
                    "new_age_of_view": "", 
                    "timeliness": "", 
                    "consistence": "", 
                    "completeness": "", 
                    "intel_arrival_time": "",
                    "queuing_time": "", 
                    "transmitting_time": "", 
                    "service_time": "", 
                    "service_rate": "", 
                    "received_data": "", 
                    "required_data": ""},
                index=[0])
            loss_data = pd.DataFrame(data=None, columns={"Epoch index": "",
                                                         "Actor of V1": "",
                                                         "Actor of V2": "",
                                                         "Actor of V3": "",
                                                         "Actor of V4": "",
                                                         "Actor of V5": "",
                                                         "Actor of V6": "",
                                                         "Actor of V7": "",
                                                         "Actor of V8": "",
                                                         "Actor of V9": "",
                                                         "Actor of V10": "",
                                                         "Critic of V1": "",
                                                         "Critic of V2": "",
                                                         "Critic of V3": "",
                                                         "Critic of V4": "",
                                                         "Critic of V5": "",
                                                         "Critic of V6": "",
                                                         "Critic of V7": "",
                                                         "Critic of V8": "",
                                                         "Critic of V9": "",
                                                         "Critic of V10": "",
                                                         "Actor of Edge": "",
                                                         "Critic of Edge": "",
                                                         "Actor of Reward": "",
                                                         "Critic of Reward": ""}, index=[0])

        start = time.time()
        while self.environment.episode_index < num_episodes:
            print("*" * 64)
            start = time.time()
            self.reset_game()
            average_actor_loss_of_sensor_nodes, average_critic_loss_of_sensor_nodes, \
            average_actor_loss_of_edge_node, average_critic_loss_of_edge_node, \
            average_actor_loss_of_reward_node, average_critic_loss_of_reward_node = self.step()
            time_taken = time.time() - start
            
            self.new_total_episode_score_so_far = self.total_episode_age_of_view_so_far
            self.total_episode_age_of_view_so_far /= self.total_episode_view_required_number_so_far
            print("Epoch index: ", self.environment.episode_index)
            print("age_of_view: ", self.total_episode_age_of_view_so_far)
            print("new_age_of_view: ", self.new_total_episode_score_so_far)
            print("Time taken: ", time_taken)

            self.total_episode_timeliness_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_consistence_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_completeness_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_intel_arrival_time /= self.total_episode_view_required_number_so_far
            self.total_episode_queuing_time_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_transmitting_time_so_far /= self.total_episode_view_required_number_so_far
            self.total_episode_service_time_so_far /= self.total_episode_view_required_number_so_far
            
            new_line_in_result = pd.DataFrame({
                "Epoch index": str(self.environment.episode_index),
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

            new_line_in_loss = pd.DataFrame({"Epoch index": str(self.environment.episode_index),
                                             "Actor of V1": str(average_actor_loss_of_sensor_nodes[0]),
                                             "Actor of V2": str(average_actor_loss_of_sensor_nodes[1]),
                                             "Actor of V3": str(average_actor_loss_of_sensor_nodes[2]),
                                             "Actor of V4": str(average_actor_loss_of_sensor_nodes[3]),
                                             "Actor of V5": str(average_actor_loss_of_sensor_nodes[4]),
                                             "Actor of V6": str(average_actor_loss_of_sensor_nodes[5]),
                                             "Actor of V7": str(average_actor_loss_of_sensor_nodes[6]),
                                             "Actor of V8": str(average_actor_loss_of_sensor_nodes[7]),
                                             "Actor of V9": str(average_actor_loss_of_sensor_nodes[8]),
                                             "Actor of V10": str(average_actor_loss_of_sensor_nodes[9]),
                                             "Critic of V1": str(average_critic_loss_of_sensor_nodes[0]),
                                             "Critic of V2": str(average_critic_loss_of_sensor_nodes[1]),
                                             "Critic of V3": str(average_critic_loss_of_sensor_nodes[2]),
                                             "Critic of V4": str(average_critic_loss_of_sensor_nodes[3]),
                                             "Critic of V5": str(average_critic_loss_of_sensor_nodes[4]),
                                             "Critic of V6": str(average_critic_loss_of_sensor_nodes[5]),
                                             "Critic of V7": str(average_critic_loss_of_sensor_nodes[6]),
                                             "Critic of V8": str(average_critic_loss_of_sensor_nodes[7]),
                                             "Critic of V9": str(average_critic_loss_of_sensor_nodes[8]),
                                             "Critic of V10": str(average_critic_loss_of_sensor_nodes[9]),
                                             "Actor of Edge": str(average_actor_loss_of_edge_node),
                                             "Critic of Edge": str(average_critic_loss_of_edge_node),
                                             "Actor of Reward": str(average_actor_loss_of_reward_node),
                                             "Critic of Reward": str(average_critic_loss_of_reward_node)},
                                            index=["0"])
            loss_data = loss_data.append(new_line_in_loss, ignore_index=True)

            if self.environment.episode_index % 10 == 0:
                print(result_data)
            # for i in self.actor_target_of_sensor_nodes[0].named_parameters():
            #     print(i)
            if self.environment.episode_index == 1:
                result_data = result_data.drop(result_data.index[[0]])
                loss_data = loss_data.drop(loss_data.index[[0]])

            """Saves the result of an episode of the game"""
            self.game_full_episode_scores.append(self.total_episode_score_so_far)
            self.rolling_results.append(
                np.mean(self.game_full_episode_scores[-1 * self.environment.experiment_config.rolling_score_window:]))

            """Updates the best episode result seen so far"""
            if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
                self.max_episode_score_seen = self.game_full_episode_scores[-1]

            if self.rolling_results[-1] > self.max_rolling_score_seen:
                if len(self.rolling_results) > self.environment.experiment_config.rolling_score_window:
                    self.max_rolling_score_seen = self.rolling_results[-1]

            if self.environment.episode_index <= 1 and self.environment.episode_index % 1 == 0:
                save_obj(obj=self.agent_config, name=temple_agent_config_name)
                save_obj(obj=self, name=temple_agent_name)
                result_data.to_csv(temple_result_name)
                loss_data.to_csv(temple_loss_name)
                print("save result data successful")

            if self.environment.episode_index <= 10 and self.environment.episode_index % 10 == 0:
                actor_nodes_name = actor_nodes_name[:-4] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                actor_edge_name = actor_edge_name[:-4] + "_episode_" + str(self.environment.episode_index) + actor_edge_name[-4:]
                save_obj(obj=self.actor_target_of_sensor_nodes, name=actor_nodes_name)
                save_obj(obj=self.actor_target_of_edge_node, name=actor_edge_name)
                print("save actor targets objectives successful")
            else:
                if self.environment.episode_index <= 100 and self.environment.episode_index % 50 == 0:
                    actor_nodes_name = actor_nodes_name[:-15] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    actor_edge_name = actor_edge_name[:-15] + "_episode_" + str(self.environment.episode_index) + actor_edge_name[-4:]
                    save_obj(obj=self.actor_local_of_sensor_nodes, name=actor_nodes_name)
                    save_obj(obj=self.actor_local_of_edge_node, name=actor_edge_name)
                    print("save actor targets objectives successful")
                elif self.environment.episode_index > 100 and self.environment.episode_index <= 300 and self.environment.episode_index % 50 == 0 or self.environment.episode_index > 300 and self.environment.episode_index <= 1000 and self.environment.episode_index % 10 ==0:
                    actor_nodes_name = actor_nodes_name[:-16] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    actor_edge_name = actor_edge_name[:-16] + "_episode_" + str(self.environment.episode_index) + actor_edge_name[-4:]
                    save_obj(obj=self.actor_local_of_sensor_nodes, name=actor_nodes_name)
                    save_obj(obj=self.actor_local_of_edge_node, name=actor_edge_name)
                    print("save actor targets objectives successful")
                elif self.environment.episode_index > 1000 and self.environment.episode_index % 10 == 0:
                    actor_nodes_name = actor_nodes_name[:-17] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    actor_edge_name = actor_edge_name[:-17] + "_episode_" + str(self.environment.episode_index) + actor_edge_name[-4:]
                    save_obj(obj=self.actor_local_of_sensor_nodes, name=actor_nodes_name)
                    save_obj(obj=self.actor_local_of_edge_node, name=actor_edge_name)
                    print("save actor targets objectives successful")

            if self.environment.episode_index <= 2000 and self.environment.episode_index % 100 == 0:
                save_obj(obj=self.agent_config, name=temple_agent_config_name)
                save_obj(obj=self, name=temple_agent_name)
                print("save agent objective successful")

            if self.environment.episode_index % 10 == 0:
                result_data.to_csv(temple_result_name)
                loss_data.to_csv(temple_loss_name)
                print("save result data successful")

        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def reset_game(self):
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

        self.sensor_nodes_action = torch.from_numpy(np.zeros(shape=(self.environment.experiment_config.vehicle_number,
                                                                    self.sensor_action_size),
                                                             dtype=np.float)).float().to(self.device)
        self.saved_sensor_nodes_action = torch.from_numpy(np.zeros(shape=(self.environment.experiment_config.vehicle_number,
                                                                    self.sensor_action_size),
                                                             dtype=np.float)).float().to(self.device)
        self.edge_node_action = None
        self.saved_edge_action = None
        self.sensor_nodes_reward = None
        self.edge_node_reward = None
        self.next_sensor_nodes_observation = None
        self.next_edge_node_observation = None
        self.next_reward_observation = None

        """Resets the game information so we are ready to play a new episode"""
        self.sensor_nodes_observation, self.edge_node_observation, self.reward_observation = self.environment.reset()
        # assert self.sensor_nodes_observation.shape[0] == self.environment.experiment_config.vehicle_number\
        #     and self.sensor_nodes_observation.shape[1] == self.environment.get_sensor_observation_size(), "sensor_nodes_observation is not same"
        # assert self.edge_node_observation.shape[0] == self.environment.get_edge_observation_size(), "edge_node_observation is not same"
        # assert self.reward_observation.shape[0] == self.environment.get_global_state_size(), "reward_observation is not same"
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
        self.sensor_exploration_strategy.reset()
        self.edge_exploration_strategy.reset()
