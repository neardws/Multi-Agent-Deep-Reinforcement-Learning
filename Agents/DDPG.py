# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：DDPG.py
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
from Config.AgentConfig import AgentConfig
from Utilities.Data_structures.DDPG_ReplayBuffer import DDPG_ReplayBuffer
from Utilities.FileOperator import save_obj, load_obj

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)

class DDPG_Agent(object):

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
        self.next_observation = None

        _, _, self.observation = self.environment.reset()

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
        self.experience_replay_buffer = DDPG_ReplayBuffer(
            buffer_size=self.agent_config.critic_experience_replay_buffer_buffer_size,
            batch_size=self.agent_config.actor_experience_replay_buffer_batch_size,
            seed=self.agent_config.actor_experience_replay_buffer_seed,
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

        self.action_size = self.environment.get_sensor_action_size() * self.environment.experiment_config.vehicle_number + self.environment.get_edge_action_size()

        """Actor Network"""

        self.actor_local = self.create_nn(
            input_dim=self.environment.get_global_state_size(),
            output_dim=[
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            ],
            key_to_use="Actor_of_DDPG"
        )

        self.actor_target = self.create_nn(
            input_dim=self.environment.get_global_state_size(),
            output_dim=[
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            ],
            key_to_use="Actor_of_DDPG"
        )

        DDPG_Agent.copy_model_over(
            from_model=self.actor_local,
            to_model=self.actor_target
        )

        self.actor_optimizer = optim.Adam(
            params=self.actor_local.parameters(),
            lr=self.hyperparameters["Actor_of_DDPG"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer, 
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

        self.critic_local = self.create_nn(
            input_dim=self.environment.get_global_state_size() + self.action_size,
            output_dim=1,
            key_to_use="Critic_of_DDPG"
        )

        self.critic_target = self.create_nn(
            input_dim=self.environment.get_global_state_size() + self.action_size,
            output_dim=1,
            key_to_use="Critic_of_DDPG"
        )

        DDPG_Agent.copy_model_over(
            from_model=self.critic_local,
            to_model=self.critic_target
        )

        self.critic_optimizer = optim.Adam(
            params=self.critic_local.parameters(),
            lr=self.hyperparameters["Critic_of_DDPG"]["learning_rate"],
            eps=1e-8
        )

        optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, 
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
        _, _, self.observation = self.environment.reset()

    def config_actor_target(self, actor_target):
        self.actor_target = actor_target

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
        
        nodes_start_episode_num = 300 * 2

        with tqdm(total=self.environment.max_episode_length) as my_bar:

            while not self.done:  # when the episode is not over
                self.pick_actions()
                self.conduct_action()
                self.save_experience()
                
                if self.time_to_learn(nodes_start_episode_num):

                    for _ in range(self.hyperparameters["actor_nodes_learning_updates_per_learning_session"]):
                        observations, actions, rewards, next_observations, dones = self.experience_replay_buffer.sample()
                        average_actor_loss_of_edge_node += self.actor_to_learn(observations=observations)
                        average_critic_loss_of_edge_node += self.critic_to_learn(observations=observations, actions=actions, rewards=rewards, next_observations=next_observations, dones=dones)
                    average_actor_loss_of_edge_node /= self.hyperparameters["actor_nodes_learning_updates_per_learning_session"]
                    average_critic_loss_of_edge_node /= self.hyperparameters["actor_nodes_learning_updates_per_learning_session"]

                """Renew by environment"""
                self.observation = self.next_observation.clone().detach()

                my_bar.update(n=1)

        return average_actor_loss_of_sensor_nodes, average_critic_loss_of_sensor_nodes, \
            average_actor_loss_of_edge_node, average_critic_loss_of_edge_node, \
            average_actor_loss_of_reward_node, average_critic_loss_of_reward_node
    

    """
    local and target network to pick actions
    """
    def pick_actions(self):
        observation = self.observation.unsqueeze(0).float().to(self.device)

        self.actor_local.eval()  # set the model to evaluation state
        with torch.no_grad():  # do not compute the gradient
            self.action = self.actor_local(observation)
        self.actor_local.train()  # set the model to training state


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

        edge_nodes_bandwidth = self.action[0][-self.environment.experiment_config.data_types_number:].unsqueeze(0).cpu().data.numpy()
        
        dict_action = {
            "priority": priority,
            "arrival_rate": arrival_rate,
            "bandwidth": edge_nodes_bandwidth[0]
        }
        """Conducts an action in the environment"""
        _, _, self.next_observation, sensor_nodes_reward, \
            self.reward, view_required_number, self.done, sum_age_of_view, sum_timeliness, sum_consistence, sum_completeness, \
            sum_intel_arrival_time, sum_queuing_time, sum_transmitting_time, sum_service_time, sum_service_rate, sum_received_data_number, \
            sum_required_data_number, new_reward = self.environment.step_with_difference_rewards(dict_action)
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

    def save_experience(self):
        if self.experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        self.experience_replay_buffer.add_experience(
            observation=self.observation.clone().detach(),
            action=self.action.clone().detach(),
            reward=self.reward,
            next_observation=self.next_observation.clone().detach(),
            done=self.done)

    def time_to_learn(self, nodes_start_episode_num):
        """Returns boolean indicating whether there are enough experiences to learn from
        and it is time to learn for the actor and critic of sensor nodes and edge node"""
        start_episode_index = nodes_start_episode_num / self.environment.experiment_config.max_episode_length
        if (self.environment.episode_index) >= start_episode_index:
            return self.environment.episode_step % self.hyperparameters["actor_nodes_update_every_n_steps"] == 0
        else:
            return False

    def actor_to_learn(
        self,
        observations: list
    ):
        actions_predicted = self.actor_local(observations)
        
        actor_loss = -self.critic_local(
            torch.cat((observations, actions_predicted), dim=1)
        ).mean()

        self.take_optimisation_step(self.actor_optimizer,
                                    self.actor_local,
                                    actor_loss,
                                    self.hyperparameters["Actor_of_DDPG"]["gradient_clipping_norm"])
        
        self.soft_update_of_target_network(
            self.actor_local,
            self.actor_target,
            self.hyperparameters["Actor_of_DDPG"]["tau"])
        
        return actor_loss.item()

    def critic_to_learn(
        self,
        observations: list,
        actions: list,
        rewards: list,
        next_observations: list,
        dones: Tensor
    ):
        with torch.no_grad():
            actions_next = self.actor_target(next_observations)
            critic_targets_next = self.critic_target(torch.cat((next_observations, actions_next), 1))
            critic_targets = rewards + (0.996 * critic_targets_next * (1.0 - dones))

        critic_expected = self.critic_local(torch.cat((observations, actions), 1))
        critic_loss = functional.mse_loss(critic_expected, critic_targets)
        self.take_optimisation_step(self.critic_optimizer, 
                                    self.critic_local, 
                                    critic_loss, 
                                    self.hyperparameters["Critic_of_DDPG"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(
            self.critic_local, 
            self.critic_target, 
            self.hyperparameters["Critic_of_DDPG"]["tau"])
        
        return critic_loss.item()
        
    @staticmethod
    def take_optimisation_step(optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
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
                save_obj(obj=self.actor_local, name=actor_nodes_name)
                print("save actor targets objectives successful")
            else:
                if self.environment.episode_index <= 100 and self.environment.episode_index % 50 == 0:
                    actor_nodes_name = actor_nodes_name[:-15] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    save_obj(obj=self.actor_local, name=actor_nodes_name)
                    print("save actor targets objectives successful")
                elif self.environment.episode_index > 100 and self.environment.episode_index <= 300 and self.environment.episode_index % 50 == 0 or self.environment.episode_index > 300 and self.environment.episode_index <= 1000 and self.environment.episode_index % 10 ==0:
                    actor_nodes_name = actor_nodes_name[:-16] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    save_obj(obj=self.actor_local, name=actor_nodes_name)
                    print("save actor targets objectives successful")
                elif self.environment.episode_index > 1000 and self.environment.episode_index % 10 == 0:
                    actor_nodes_name = actor_nodes_name[:-17] + "_episode_" + str(self.environment.episode_index) + actor_nodes_name[-4:]
                    save_obj(obj=self.actor_local, name=actor_nodes_name)
                    print("save actor targets objectives successful")

            if self.environment.episode_index <= 10000 and self.environment.episode_index % 100 == 0:
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
        """boolean parameters"""
        self.done = None  # True or False indicate is episode finished
        """float parameters"""
        self.reward = None
        """dict() parameters"""
        self.action = None
        self.next_observation = None


        """Resets the game information so we are ready to play a new episode"""
        _, _, self.observation = self.environment.reset()

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
