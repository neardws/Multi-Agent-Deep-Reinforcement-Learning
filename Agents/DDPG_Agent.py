# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：DDPG_Agent.py
@Author  ：Neardws
@Date    ：9/7/21 2:35 下午 
"""
import time
import torch
import numpy as np
import pandas as pd
from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv
import torch.nn.functional as functional
from Utilities.Data_structures.DDPG_ReplayBuffer import DDPG_ReplayBuffer
from Exploration_strategies.Gaussian_Exploration import Gaussian_Exploration
from nn_builder.pytorch.NN import NN
from torch import optim
from tqdm import tqdm
from Utilities.FileOperator import save_obj


class DDPG_Agent(object):
    def __init__(self, environment: VehicularNetworkEnv):
        self.environment = environment
        self.done = None
        self.reward = None
        self.action = None
        self.observation = None
        self.next_observation = None
        
        _, _, self.observation = self.environment.reset()
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

        self.device = "cuda" if self.environment.experiment_config.use_gpu else "cpu"

        self.action_size = self.environment.get_sensor_action_size() * self.environment.experiment_config.vehicle_number + self.environment.get_edge_action_size()
        
        self.hyperparameters = {

            "Actor_of_DDPG": {
                "learning_rate": 1e-5,
                "linear_hidden_units":
                    [512,
                    256
                    ],
                "final_layer_activation": [
                    "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax",
                    "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax",
                    "softmax"
                ],  # 20 actions of vehicles, and one action of edge node
                "batch_norm": False,
                "tau": 0.00001,
                "gradient_clipping_norm": 5,
                "noise_seed": np.random.randint(0, 2 ** 32 - 2),
                "mu": 0.0,
                "theta": 0.15,
                "sigma": 0.25,
                "action_noise_std": 0.001,
                "action_noise_clipping_range": 1.0
            },

            "Critic_of_DDPG": {
                "learning_rate": 1e-4,
                "linear_hidden_units":
                    [512,
                    256],
                "final_layer_activation": "tanh",
                "batch_norm": False,
                "tau": 0.00001,
                "gradient_clipping_norm": 5
            }
        }
        
        self.replay_buffer = DDPG_ReplayBuffer(
            buffer_size=50000,
            batch_size=64,
            seed=2801641275
        )

        self.sensor_exploration_strategy = Gaussian_Exploration(size=self.action_size,
                                                                hyperparameters=self.hyperparameters,
                                                                key_to_use="Actor_of_DDPG")

        self.actor_local_of_ddpg = self.create_nn(
            input_dim=self.environment.get_global_state_size(),
            output_dim=[
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            ],
            key_to_use="Actor_of_DDPG"
        )

        self.actor_target_of_ddpg = self.create_nn(
            input_dim=self.environment.get_global_state_size(),
            output_dim=[
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            ],
            key_to_use="Actor_of_DDPG"
        )

        DDPG_Agent.copy_model_over(
            from_model=self.actor_local_of_ddpg,
            to_model=self.actor_target_of_ddpg
        )

        self.actor_optimizer_of_ddpg = optim.Adam(
            params=self.actor_local_of_ddpg.parameters(),
            lr=self.hyperparameters["Actor_of_DDPG"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer_of_ddpg, 
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

        self.critic_local_of_ddpg = self.create_nn(
            input_dim=self.environment.get_global_state_size() + self.action_size,
            output_dim=1,
            key_to_use="Critic_of_DDPG"
        )

        self.critic_target_of_ddpg = self.create_nn(
            input_dim=self.environment.get_global_state_size() + self.action_size,
            output_dim=1,
            key_to_use="Critic_of_DDPG"
        )

        DDPG_Agent.copy_model_over(
            from_model=self.critic_local_of_ddpg,
            to_model=self.critic_target_of_ddpg
        )

        self.critic_optimizer_of_ddpg = optim.Adam(
            params=self.critic_local_of_ddpg.parameters(),
            lr=self.hyperparameters["Critic_of_DDPG"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer_of_ddpg, 
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

    def config_environment(self, environment):
        self.environment = environment
        _, _, self.observation = self.environment.reset()

    def target_step(self):
        with tqdm(total=self.environment.max_episode_length) as my_bar:
            while not self.done:
                self.target_pick_actions()
                self.conduct_action()
                my_bar.update(n=1)

    def step(self):
        with tqdm(total=self.environment.max_episode_length) as my_bar:
            while not self.done:
                self.pick_actions()
                self.conduct_action()
                self.save_experience()
                if self.time_for_learn():
                    for i in range(1):
                        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample()
                        self.actor_and_critic_to_learn(observations, actions, rewards, next_observations, dones)
                my_bar.update(n=1)
    
    def target_pick_actions(self):
        state = self.observation.unsqueeze(0).float().to(self.device)
        # self.actor_target_of_ddpg.eval()
        self.actor_local_of_ddpg.eval()
        with torch.no_grad():
            # action = self.actor_target_of_ddpg(state)
            action = self.actor_local_of_ddpg(state)
        self.action = action

    def pick_actions(self):
        state = self.observation.unsqueeze(0).float().to(self.device)
        self.actor_local_of_ddpg.eval()
        with torch.no_grad():
            action = self.actor_local_of_ddpg(state)
        self.actor_local_of_ddpg.train()
        self.action = action

    def conduct_action(self):

        priority = np.zeros(shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number),
                            dtype=np.float)
        arrival_rate = np.zeros(
            shape=(self.environment.experiment_config.vehicle_number, self.environment.experiment_config.data_types_number), dtype=np.float)

        for sensor_node_index in range(self.environment.experiment_config.vehicle_number):
            start_index = sensor_node_index * 2 * self.environment.experiment_config.data_types_number
            sensor_node_action = self.action[0][start_index: start_index + 2 * self.environment.experiment_config.data_types_number]
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

        edge_nodes_bandwidth = self.action[0][-self.environment.experiment_config.data_types_number:].unsqueeze(0).cpu().data.numpy() * self.environment.experiment_config.bandwidth
        
        dict_action = {
            "priority": priority,
            "arrival_rate": arrival_rate,
            "bandwidth": edge_nodes_bandwidth
        }
        _, _, self.next_observation, self.reward, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
        sum_intel_arrival_time, sum_queuing_time, sum_transmitting_time, sum_service_time, sum_service_rate, sum_received_data_number, \
        sum_required_data_number, new_reward = self.environment.step(dict_action)
        self.total_episode_score_so_far += self.reward
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
        

    def save_experience(self):
        if self.replay_buffer is None:
            raise Exception("Buffer is None")
        self.replay_buffer.add_experience(
            observation=self.observation,
            action=self.action,
            reward=self.reward,
            next_observation=self.next_observation,
            done=self.done
        )

    def time_for_learn(self):
        return len(self.replay_buffer) > (
                256 * 10) and \
               self.environment.episode_step % 300 == 0

    def actor_and_critic_to_learn(self, observations, actions, rewards, next_observations, dones):
        with torch.no_grad():
            actions_next = self.actor_target_of_ddpg(next_observations)
            critic_targets_next = self.critic_target_of_ddpg(torch.cat((next_observations, actions_next), 1))
            critic_targets = rewards + (0.996 * critic_targets_next * (1.0 - dones))

        critic_expected = self.critic_local_of_ddpg(torch.cat((observations, actions), 1))
        critic_loss = functional.mse_loss(critic_expected, critic_targets)
        self.take_optimisation_step(self.critic_optimizer_of_ddpg, 
                                    self.critic_local_of_ddpg, 
                                    critic_loss, 
                                    self.hyperparameters["Critic_of_DDPG"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_of_ddpg, self.critic_target_of_ddpg, self.hyperparameters["Critic_of_DDPG"]["tau"])

        actions_predicted = self.actor_local_of_ddpg(observations)
        
        actor_loss = -self.critic_local_of_ddpg(
            torch.cat((observations, actions_predicted), dim=1)
        ).mean()

        self.take_optimisation_step(self.actor_optimizer_of_ddpg,
                                    self.actor_local_of_ddpg,
                                    actor_loss,
                                    self.hyperparameters["Actor_of_DDPG"]["gradient_clipping_norm"])
    
    def run_n_episodes_as_results(self, num_episodes, result_name):

        try:
            result_data = pd.read_csv(result_name, names=["Epoch index", "age_of_view", "new_age_of_view", "timeliness", "consistence", "completeness", "intel_arrival_time", "queuing_time", "transmitting_time", "service_time", "service_rate", "received_data", "required_data"], header=0)
        except FileNotFoundError:
            result_data = pd.DataFrame(data=None, columns={"Epoch index": "", "age_of_view": "", "new_age_of_view": "", "timeliness": "", "consistence": "", "completeness": "", "intel_arrival_time": "", "queuing_time": "", "transmitting_time": "", "service_time": "", "service_rate": "",  "received_data": "", "required_data": ""},
                                       index=[0])

        for i in range(num_episodes):
            print("*" * 64)
            self.reset_game()
            self.target_step()
            print("Epoch index: ", i)
            print("Total reward: ", self.total_episode_score_so_far)

            self.total_episode_timeliness_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_consistence_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_completeness_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_intel_arrival_time /= self.environment.experiment_config.max_episode_length
            self.total_episode_queuing_time_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_transmitting_time_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_service_time_so_far /= self.environment.experiment_config.max_episode_length

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


    def run_n_episodes(self, num_episodes, temple_result_name, agent_name):

        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.environment.experiment_config.episode_number

        try:
            result_data = pd.read_csv(temple_result_name, names=["Epoch index", "age_of_view", "new_age_of_view", "timeliness", "consistence", "completeness", "intel_arrival_time", "queuing_time", "transmitting_time", "service_time", "service_rate", "received_data", "required_data"], header=0)
        except FileNotFoundError:
            result_data = pd.DataFrame(data=None, columns={"Epoch index": "", "age_of_view": "", "new_age_of_view": "", "timeliness": "", "consistence": "", "completeness": "", "intel_arrival_time": "", "queuing_time": "", "transmitting_time": "", "service_time": "", "service_rate": "",  "received_data": "", "required_data": ""},
                                       index=[0])

        start = time.time()
        while self.environment.episode_index < num_episodes:
            print("*" * 64)
            start = time.time()
            self.reset_game()
            self.step()
            time_taken = time.time() - start
            print("Epoch index: ", self.environment.episode_index)
            print("Total reward: ", self.total_episode_score_so_far)
            print("new_age_of_view: ", self.new_total_episode_score_so_far)
            print("Time taken: ", time_taken)
            
            self.total_episode_timeliness_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_consistence_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_completeness_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_intel_arrival_time /= self.environment.experiment_config.max_episode_length
            self.total_episode_queuing_time_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_transmitting_time_so_far /= self.environment.experiment_config.max_episode_length
            self.total_episode_service_time_so_far /= self.environment.experiment_config.max_episode_length

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

            if self.environment.episode_index % 10 == 0:
                print(result_data)

            if self.environment.episode_index == 1:
                result_data = result_data.drop(result_data.index[[0]])

            # """Saves the result of an episode of the game"""
            # # self.game_full_episode_scores.append(self.total_episode_score_so_far)
            # self.rolling_results.append(
            #     np.mean(self.game_full_episode_scores[-1 * self.environment.experiment_config.rolling_score_window:]))

            # """Updates the best episode result seen so far"""
            # if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            #     self.max_episode_score_seen = self.game_full_episode_scores[-1]

            # if self.rolling_results[-1] > self.max_rolling_score_seen:
            #     if len(self.rolling_results) > self.environment.experiment_config.rolling_score_window:
            #         self.max_rolling_score_seen = self.rolling_results[-1]

            if self.environment.episode_index <= 1 and self.environment.episode_index % 1 == 0:
                # save_obj(obj=self, name=temple_agent_name)
                result_data.to_csv(temple_result_name)
                # loss_data.to_csv(temple_loss_name)
                print("save result data successful")

            if self.environment.episode_index <= 500 and self.environment.episode_index % 100 == 0 or self.environment.episode_index > 500 and self.environment.episode_index % 50 == 0:
                save_obj(obj=self, name=agent_name)
                print("save objectives successful")

            if self.environment.episode_index % 25 == 0:
                result_data.to_csv(temple_result_name)
                # loss_data.to_csv(temple_loss_name)
                print("save result data successful")

        time_taken = time.time() - start


    def reset_game(self):
        self.done = None
        self.reward = None
        self.action = None
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
        _, _, self.observation = self.environment.reset()
        self.sensor_exploration_strategy.reset()


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
            seed = 2419977517

        default_hyperparameter_choices = {"output_activation": None,
                                          "hidden_activations": "relu",
                                          "dropout": 0.5,
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

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())


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