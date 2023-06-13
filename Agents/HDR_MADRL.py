# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：HDR_MADRL.py
@Author  ：Neardws
@Date    ：12/10/21 2:50 下午 
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
from Utilities.Data_structures.HDRSensorCriticReplayBuffer import HDRSensorCriticReplayBuffer
from Utilities.Data_structures.HDRSensorActorReplayBuffer import HDRSensorActorReplayBuffer
from Utilities.Data_structures.HDREdgeCriticReplayBuffer import HDREdgeCriticReplayBuffer
from Utilities.Data_structures.HDREdgeActorReplayBuffer import HDREdgeActorReplayBuffer
from Utilities.FileOperator import save_obj, load_obj

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)


class HDR_MADRL_Agent(object):

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

        self.sensor_nodes_observation, self.edge_node_observation, _ = self.environment.reset()

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
        self.sensor_actor_experience_replay_buffer = HDRSensorActorReplayBuffer(
            buffer_size=self.agent_config.actor_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.actor_experience_replay_buffer_batch_size,
            seed=self.agent_config.actor_experience_replay_buffer_seed,
            dropout=self.agent_config.actor_experience_replay_buffer_dropout,
            device=self.device
        )

        self.sensor_critic_experience_replay_buffer = HDRSensorCriticReplayBuffer(
            buffer_size=self.agent_config.critic_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.critic_experience_replay_buffer_batch_size,
            seed=self.agent_config.critic_experience_replay_buffer_seed,
            dropout=self.agent_config.critic_experience_replay_buffer_dropout,
            device=self.device
        )

        self.edge_actor_experience_replay_buffer = HDREdgeActorReplayBuffer(
            buffer_size=self.agent_config.actor_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.actor_experience_replay_buffer_batch_size,
            seed=self.agent_config.actor_experience_replay_buffer_seed,
            dropout=self.agent_config.actor_experience_replay_buffer_dropout,
            device=self.device
        )

        self.edge_critic_experience_replay_buffer = HDREdgeCriticReplayBuffer(
            buffer_size=self.agent_config.critic_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.critic_experience_replay_buffer_batch_size,
            seed=self.agent_config.critic_experience_replay_buffer_seed,
            dropout=self.agent_config.critic_experience_replay_buffer_dropout,
            device=self.device
        )

        """Init input and output size of neural network"""
        self.sensor_observation_size = self.environment.get_sensor_observation_size()
        self.sensor_action_size = self.environment.get_sensor_action_size()
        self.critic_size_for_sensor = self.environment.get_critic_size_for_sensor()

        self.edge_observation_size = self.environment.get_actor_input_size_for_edge()
        self.edge_action_size = self.environment.get_edge_action_size()
        self.critic_size_for_edge = self.environment.get_critic_size_for_edge()


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
            HDR_MADRL_Agent.copy_model_over(
                from_model=self.actor_local_of_sensor_nodes[vehicle_index],
                to_model=self.actor_target_of_sensor_nodes[vehicle_index]
            )

        self.actor_optimizer_of_sensor_nodes = [
            optim.Adam(
                params=self.actor_local_of_sensor_nodes[vehicle_index].parameters(),
                lr=self.hyperparameters["Actor_of_Sensor"]["learning_rate"],
                eps=1e-8
            ) for vehicle_index in range(self.environment.experiment_config.vehicle_number)
        ]

        for vehicle_index in range(self.environment.experiment_config.vehicle_number):
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
            HDR_MADRL_Agent.copy_model_over(
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
                self.critic_optimizer_of_sensor_nodes[vehicle_index], 
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
        
        self.actor_local_of_edge_node = self.create_nn(
            input_dim=self.edge_observation_size,
            output_dim=self.edge_action_size,
            key_to_use="Actor_of_Edge"
        )

        self.actor_traget_of_edge_node = self.create_nn(
            input_dim=self.edge_observation_size,
            output_dim=self.edge_action_size,
            key_to_use="Actor_of_Edge"
        )

        HDR_MADRL_Agent.copy_model_over(
            from_model=self.actor_local_of_edge_node,
            to_model=self.actor_traget_of_edge_node
        )

        self.actor_optimizer_of_edge_node = optim.Adam(
            params=self.actor_local_of_edge_node.parameters(),
            lr=self.hyperparameters["Actor_of_Edge"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer_of_edge_node, 
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

        HDR_MADRL_Agent.copy_model_over(
            from_model=self.critic_local_of_edge_node,
            to_model=self.critic_target_of_edge_node
        )

        self.critic_optimizer_of_edge_node = optim.Adam(
            params=self.critic_local_of_edge_node.parameters(),
            lr=self.hyperparameters["Critic_of_Edge"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer_of_edge_node, 
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
        self.sensor_nodes_observation, self.edge_node_observation, _ = self.environment.reset()

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
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        if key_to_use:
            hyperparameters = hyperparameters[key_to_use]
        if override_seed:
            seed = override_seed
        else:
            seed = self.agent_config.nn_seed

        default_hyperparameter_choices = {
            "output_activation": None,
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

        return NN(
            input_dim=input_dim,
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
    
    def step(self):
        """Runs a step in the game"""
        average_actor_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)
        average_critic_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)
        average_actor_loss_of_edge_node = 0
        average_critic_loss_of_edge_node = 0
        average_actor_loss_of_reward_node = 0
        average_critic_loss_of_reward_node = 0
        
        nodes_start_episode_num = 300 * 128

        with tqdm(total=self.environment.max_episode_length) as my_bar:

            while not self.done:  # when the episode is not over
                self.sensor_nodes_pick_actions()
                self.edge_node_pick_actions()
                self.combined_action()
                self.conduct_action()
                self.save_sensor_actor_experience()
                self.save_sensor_critic_experience()
                self.save_edge_actor_experience()
                self.save_edge_critic_experience()
                
                if self.time_for_actor_of_sensor_nodes_to_learn(nodes_start_episode_num):
                    one_time_average_actor_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)

                    for _ in range(self.hyperparameters["actor_nodes_learning_updates_per_learning_session"]):
                        sensor_nodes_observations, sensor_nodes_actions = self.sensor_actor_experience_replay_buffer.sample()

                        actor_loss_of_sensor_nodes \
                            = self.actor_sensor_nodes_to_learn(
                            sensor_nodes_observations=sensor_nodes_observations,
                            sensor_nodes_actions=sensor_nodes_actions)

                        for index in range(self.environment.experiment_config.vehicle_number):
                            one_time_average_actor_loss_of_sensor_nodes[index] += actor_loss_of_sensor_nodes[index]


                    for index in range(self.environment.experiment_config.vehicle_number):
                        one_time_average_actor_loss_of_sensor_nodes[index] /= self.hyperparameters[
                            "actor_nodes_learning_updates_per_learning_session"]

                    for index in range(self.environment.experiment_config.vehicle_number):
                        average_actor_loss_of_sensor_nodes[index] += one_time_average_actor_loss_of_sensor_nodes[index]


                if self.time_for_critic_of_sensor_nodes_to_learn(nodes_start_episode_num):
                    one_time_average_critic_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)

                    for _ in range(self.hyperparameters["critic_nodes_learning_updates_per_learning_session"]):
                        sensor_nodes_observations, sensor_nodes_actions, \
                        sensor_nodes_rewards, next_sensor_nodes_observations, \
                        dones = self.sensor_critic_experience_replay_buffer.sample()

                        critic_loss_of_sensor_nodes \
                            = self.critic_sensor_nodes_to_learn(
                            sensor_nodes_observations=sensor_nodes_observations,
                            sensor_nodes_actions=sensor_nodes_actions,
                            sensor_nodes_rewards=sensor_nodes_rewards,
                            next_sensor_nodes_observations=next_sensor_nodes_observations,
                            dones=dones)
                        for index in range(self.environment.experiment_config.vehicle_number):
                            one_time_average_critic_loss_of_sensor_nodes[index] += critic_loss_of_sensor_nodes[index]

                    for index in range(self.environment.experiment_config.vehicle_number):
                        one_time_average_critic_loss_of_sensor_nodes[index] /= self.hyperparameters[
                            "critic_nodes_learning_updates_per_learning_session"]

                    for index in range(self.environment.experiment_config.vehicle_number):
                        average_critic_loss_of_sensor_nodes[index] += one_time_average_critic_loss_of_sensor_nodes[
                            index]

                if self.time_for_actor_of_edge_node_to_learn(nodes_start_episode_num):
                    one_time_average_actor_loss_of_edge_node = 0

                    for _ in range(self.hyperparameters["actor_edge_learning_updates_per_learning_session"]):
                        edge_node_observations, sensor_nodes_actions, edge_node_actions = self.edge_actor_experience_replay_buffer.sample()

                        actor_loss_of_edge_node \
                            = self.actor_edge_node_to_learn(
                            edge_node_observations=edge_node_observations,
                            sensor_nodes_actions=sensor_nodes_actions)

                        one_time_average_actor_loss_of_edge_node += actor_loss_of_edge_node


                    one_time_average_actor_loss_of_edge_node /= self.hyperparameters[
                            "actor_edge_learning_updates_per_learning_session"]

                    average_actor_loss_of_edge_node += one_time_average_actor_loss_of_edge_node

                if self.time_for_critic_of_edge_node_to_learn(nodes_start_episode_num):
                    one_time_average_critic_loss_of_edge_node = 0

                    for _ in range(self.hyperparameters["critic_edge_learning_updates_per_learning_session"]):
                        edge_node_observations, sensor_nodes_actions, edge_node_actions, \
                            edge_node_rewards, next_sensor_nodes_observations, next_edge_node_observations, \
                            dones = self.edge_critic_experience_replay_buffer.sample()

                        critic_loss_of_edge_node \
                            = self.critic_edge_node_to_learn(
                            edge_node_observations=edge_node_observations,
                            sensor_nodes_actions=sensor_nodes_actions,
                            edge_node_actions=edge_node_actions,
                            edge_node_rewards=edge_node_rewards,
                            next_sensor_nodes_observations=next_sensor_nodes_observations,
                            next_edge_node_observations=next_edge_node_observations,
                            dones=dones)

                        one_time_average_critic_loss_of_edge_node += critic_loss_of_edge_node

                    one_time_average_critic_loss_of_edge_node /= self.hyperparameters[
                            "critic_edge_learning_updates_per_learning_session"]

                    average_critic_loss_of_edge_node[index] += one_time_average_critic_loss_of_edge_node

                """Renew by environment"""
                self.sensor_nodes_observation = self.next_sensor_nodes_observation.clone().detach()
                self.edge_node_observation = self.next_edge_node_observation.clone().detach()

                my_bar.update(n=1)

        for index in range(self.environment.experiment_config.vehicle_number):
            average_actor_loss_of_sensor_nodes[index] /= \
                (self.environment.max_episode_length / self.hyperparameters["actor_nodes_update_every_n_steps"])
            average_critic_loss_of_sensor_nodes[index] /= \
                (self.environment.max_episode_length / self.hyperparameters["critic_nodes_update_every_n_steps"])

        average_actor_loss_of_edge_node /= \
                (self.environment.max_episode_length / self.hyperparameters["actor_edge_update_every_n_steps"])
        average_critic_loss_of_edge_node /= \
            (self.environment.max_episode_length / self.hyperparameters["critic_edge_update_every_n_steps"])

        return average_actor_loss_of_sensor_nodes, average_critic_loss_of_sensor_nodes, \
            average_actor_loss_of_edge_node, average_critic_loss_of_edge_node, \
            average_actor_loss_of_reward_node, average_critic_loss_of_reward_node
    

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
    
    def edge_node_pick_actions(self):
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

        for action_index in range(self.edge_action_size):
            self.edge_node_action[action_index] = edge_action[0][action_index]
    

    def combined_action(self):

        priority = np.zeros(shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number),
                            dtype=np.float)
        arrival_rate = np.zeros(
            shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number), 
            dtype=np.float)

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

        edge_nodes_bandwidth = self.edge_node_action.cpu().data.numpy()

        self.action = {
            "priority": priority,
            "arrival_rate": arrival_rate,
            "bandwidth": edge_nodes_bandwidth
        }

    def conduct_action(self):
        """Conducts an action in the environment"""
        self.next_sensor_nodes_observation, self.next_edge_node_observation, _, sensor_nodes_reward, edge_node_reward,\
            self.reward, view_required_number, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
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

        self.sensor_nodes_reward = torch.from_numpy(sensor_nodes_reward).float().to(self.device).unsqueeze(0)
        self.edge_node_reward= edge_node_reward

    def save_sensor_actor_experience(self):
        """
        sensor_nodes_observations=torch.empty(), sensor_actions=torch.empty(),
        sensor_nodes_rewards=torch.empty(), next_sensor_nodes_observations=torch.empty()
        Saves the recent experience to the experience replay buffer
        :return: None
        """
        if self.sensor_actor_experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HDR_MADRL.py")
        """Save as torch.Tensor"""
        self.sensor_actor_experience_replay_buffer.add_experience(
            sensor_nodes_observation=self.sensor_nodes_observation.clone().detach(),
            sensor_nodes_action=self.saved_sensor_nodes_action.clone().detach())

    def save_sensor_critic_experience(self):
        """
        sensor_nodes_observations=torch.empty(), sensor_actions=torch.empty(),
        sensor_nodes_rewards=torch.empty(), next_sensor_nodes_observations=torch.empty()
        Saves the recent experience to the experience replay buffer
        :return: None
        """
        if self.sensor_critic_experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HDR_MADRL.py")
        """Save as torch.Tensor"""
        self.sensor_critic_experience_replay_buffer.add_experience(
            sensor_nodes_observation=self.sensor_nodes_observation.clone().detach(),
            sensor_nodes_action=self.saved_sensor_nodes_action.clone().detach(),
            sensor_nodes_reward=self.sensor_nodes_reward.clone().detach(),
            next_sensor_nodes_observation=self.next_sensor_nodes_observation.clone().detach(),
            done=self.done)

    def save_edge_actor_experience(self):
        if self.edge_actor_experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HDR_MADRL.py")
        """Save as torch.Tensor"""
        self.edge_actor_experience_replay_buffer.add_experience(
            edge_node_observation=self.edge_node_observation.clone().detach(),
            sensor_nodes_action=self.saved_sensor_nodes_action.clone().detach())

    def save_edge_critic_experience(self):
        if self.edge_critic_experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HDR_MADRL.py")
        """Save as torch.Tensor"""
        self.edge_critic_experience_replay_buffer.add_experience(
            edge_node_observation=self.edge_node_observation.clone().detach(),
            sensor_nodes_action=self.saved_sensor_nodes_action.clone().detach(),
            edge_node_action=self.saved_edge_node_action.clone().detach(),
            edge_node_reward=self.edge_node_reward,
            next_sensor_nodes_observation=self.next_sensor_nodes_observation.clone().detach(),
            next_edge_node_observation=self.next_edge_node_observation.clone().detach(),
            done=self.done)


    def time_for_actor_of_sensor_nodes_to_learn(self, nodes_start_episode_num):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if (self.environment.episode_index) >= start_episode_index:
            return self.environment.episode_step % self.hyperparameters["actor_nodes_update_every_n_steps"] == 0
        else:
            return False

    def time_for_critic_of_sensor_nodes_to_learn(self, nodes_start_episode_num):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if self.environment.episode_index >= start_episode_index:
            return self.environment.episode_step % self.hyperparameters["critic_nodes_update_every_n_steps"] == 0
        else:
            return False

    def time_for_actor_of_edge_node_to_learn(self, nodes_start_episode_num):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if (self.environment.episode_index) >= start_episode_index:
            return self.environment.episode_step % self.hyperparameters["actor_edge_update_every_n_steps"] == 0
        else:
            return False

    def time_for_critic_of_edge_node_to_learn(self, nodes_start_episode_num):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if self.environment.episode_index >= start_episode_index:
            return self.environment.episode_step % self.hyperparameters["critic_edge_update_every_n_steps"] == 0
        else:
            return False

    def actor_sensor_nodes_to_learn(
        self,
        sensor_nodes_observations: list,
        sensor_nodes_actions: list,
    ):
        actor_loss_of_sensor_nodes = np.zeros(self.environment.experiment_config.vehicle_number)

        """Runs a learning iteration for the critic of sensor nodes"""
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

            self.take_optimisation_step(
                self.actor_optimizer_of_sensor_nodes[sensor_node_index],
                self.actor_local_of_sensor_nodes[sensor_node_index],
                actor_loss_of_sensor_node,
                self.hyperparameters["Actor_of_Sensor"]["gradient_clipping_norm"]
            )
            self.soft_update_of_target_network(
                self.actor_local_of_sensor_nodes[sensor_node_index],
                self.actor_target_of_sensor_nodes[sensor_node_index],
                self.hyperparameters["Actor_of_Sensor"]["tau"]
            )

        return actor_loss_of_sensor_nodes

    def critic_sensor_nodes_to_learn(
        self,
        sensor_nodes_observations: list,
        sensor_nodes_actions: list,
        sensor_nodes_rewards: list,
        next_sensor_nodes_observations: list,
        dones: Tensor
    ):
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
                (sensor_nodes_rewards[0][0][sensor_node_index].unsqueeze(0).unsqueeze(0),
                 sensor_nodes_rewards[1][0][sensor_node_index].unsqueeze(0).unsqueeze(0)), dim=0)
            for index, sensor_nodes_reward in enumerate(sensor_nodes_rewards):
                if index > 1:
                    sensor_node_rewards = torch.cat(
                        (sensor_node_rewards, sensor_nodes_reward[0][sensor_node_index].unsqueeze(0).unsqueeze(0)),
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

        return critic_loss_of_sensor_nodes

    def actor_edge_node_to_learn(
        self,
        edge_node_observations: list,
        sensor_nodes_actions: list
    ):
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

        self.take_optimisation_step(
            self.actor_optimizer_of_edge_node,
            self.actor_local_of_edge_node,
            loss_of_edge_node,
            self.hyperparameters["Actor_of_Edge"]["gradient_clipping_norm"]
        )
        self.soft_update_of_target_network(
            self.actor_local_of_edge_node, 
            self.actor_target_of_edge_node,
            self.hyperparameters["Actor_of_Edge"]["tau"]
        )

        return actor_loss_of_edge_node

    def critic_edge_node_to_learn(
        self,
        edge_node_observations: list,
        sensor_nodes_actions: list,
        edge_node_actions: list,
        edge_node_rewards: list,
        next_sensor_nodes_observations: list,
        next_edge_node_observations: list,
        dones: Tensor
    ):
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

        return critic_loss_of_edge_node


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
            loss_data = pd.read_csv(temple_loss_name, names=[
                "Epoch index",
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
            loss_data = pd.DataFrame(data=None, columns={
                "Epoch index": "",
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

            new_line_in_loss = pd.DataFrame({
                "Epoch index": str(self.environment.episode_index),
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
                save_obj(obj=self.actor_target_of_sensor_nodes, name=actor_nodes_name)
                print("save actor targets objectives successful")
            else:
                if self.environment.episode_index <= 100 and self.environment.episode_index % 50 == 0:
                    actor_nodes_name = actor_nodes_name[:-15] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    save_obj(obj=self.actor_local_of_sensor_nodes, name=actor_nodes_name)
                    print("save actor targets objectives successful")
                elif self.environment.episode_index > 100 and self.environment.episode_index <= 300 and self.environment.episode_index % 50 == 0 or self.environment.episode_index > 300 and self.environment.episode_index <= 1000 and self.environment.episode_index % 10 ==0:
                    actor_nodes_name = actor_nodes_name[:-16] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    save_obj(obj=self.actor_local_of_sensor_nodes, name=actor_nodes_name)
                    print("save actor targets objectives successful")
                elif self.environment.episode_index > 1000 and self.environment.episode_index % 10 == 0:
                    actor_nodes_name = actor_nodes_name[:-17] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    save_obj(obj=self.actor_local_of_sensor_nodes, name=actor_nodes_name)
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

        self.sensor_nodes_action = torch.from_numpy(np.zeros(
            shape=(
                self.environment.experiment_config.vehicle_number,
                self.sensor_action_size),
            dtype=np.float)).float().to(self.device)
        self.saved_sensor_nodes_action = torch.from_numpy(np.zeros(
            shape=(
                self.environment.experiment_config.vehicle_number,
                self.sensor_action_size),
            dtype=np.float)).float().to(self.device)
        self.edge_node_action = torch.from_numpy(np.zeros(
            shape=(
                self.environment.experiment_config.vehicle_number,),
            dtype=np.float)).float().to(self.device)
        self.saved_edge_node_action = torch.from_numpy(np.zeros(
            shape=(
                self.environment.experiment_config.vehicle_number,),
            dtype=np.float)).float().to(self.device)
        self.sensor_nodes_reward = None
        self.edge_node_reward = None
        self.next_sensor_nodes_observation = None
        self.next_edge_node_observation = None

        """Resets the game information so we are ready to play a new episode"""
        self.sensor_nodes_observation, self.edge_node_observation, _ = self.environment.reset()

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

