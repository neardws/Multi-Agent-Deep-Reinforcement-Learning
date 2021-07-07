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


        self.state = self.environment.reset()
        self.sensor_observations = None
        self.edge_observation = None
        self.next_state = None
        self.reward = None
        self.done = False

        self.action = None
        self.sensor_actions = None
        self.edge_action = None
        self.reward_action = None

        """
        Some parameters
        """
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_index = 0
        self.episode_step = 0
        self.device = "cuda" if self.config.use_GPU else "cpu"
        self.turn_off_exploration = False


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
        self.sensor_observations_size = self.environment.get_sensor_observations_size()
        self.sensor_action_size = self.environment.get_sensor_action_size()
        self.actor_local_of_sensor_nodes = [
            self.create_NN_for_actor_network_of_sensor_node(
                input_dim=self.sensor_observations_size,
                output_dim=self.sensor_action_size
            ) for _ in range(self.environment.vehicle_number)
        ]

        self.actor_target_of_sensor_nodes = [
            self.create_NN_for_actor_network_of_sensor_node(
                input_dim=self.sensor_observations_size,
                output_dim=self.sensor_action_size
            ) for _ in range(self.environment.vehicle_number)
        ]

        for index in range(self.environment.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_sensor_nodes[index],
                                         to_model=self.actor_target_of_sensor_nodes[index])
        self.actor_of_sensor_nodes_optimizer = [
            optim.Adam(params=self.actor_local_of_sensor_nodes[index].parameters(),
                       lr=self.hyperparameters['actor_of_sensor']['learning_rate'],
                       eps=1e-4
            ) for index in range(self.environment.vehicle_number)
        ]

        # self.actor_local_of_sensor_nodes = []
        # for index in range(self.environment.vehicle_number):
        #     self.actor_local_of_sensor_nodes.append(
        #         self.create_NN_for_actor_network_of_sensor_node(
        #             input_dim=self.environment.get_sensor_observations_size(),
        #             output_dim=self.environment.get_sensor_action_size()
        #         )
        #     )
        # self.actor_target_of_sensor_nodes = []
        # for index in range(self.environment.vehicle_number):
        #     self.actor_target_of_sensor_nodes.append(
        #         self.create_NN_for_actor_network_of_sensor_node(
        #             input_dim=self.environment.get_sensor_observations_size(),
        #             output_dim=self.environment.get_sensor_action_size()
        #         )
        #     )
        # for index in range(self.environment.vehicle_number):
        #     HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_sensor_nodes[index],
        #                                  to_model=self.actor_target_of_sensor_nodes[index])
        # self.actor_of_sensor_nodes_optimizer = []
        # for index in range(self.environment.vehicle_number):
        #     self.actor_of_sensor_nodes_optimizer.append(
        #         optim.Adam(params=self.actor_local_of_sensor_nodes[index].parameters(),
        #                    lr=self.hyperparameters['actor_of_sensor']['learning_rate'],
        #                    eps=1e-4)
        #     )

        """Critic Network of Sensor Nodes"""
        self.critic_size_for_sensor = self.environment.get_critic_size_for_sensor()
        self.critic_local_of_sensor_nodes = [
            self.create_NN_for_critic_network_of_sensor_node(
                input_dim=self.critic_size_for_sensor,
                output_dim=1
            ) for _ in range(self.environment.vehicle_number)
        ]

        self.critic_target_of_sensor_nodes = [
            self.create_NN_for_critic_network_of_sensor_node(
                input_dim=self.critic_size_for_sensor,
                output_dim=1
            ) for _ in range(self.environment.vehicle_number)
        ]

        for index in range(self.environment.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_sensor_nodes[index],
                                         to_model=self.critic_target_of_sensor_nodes[index])
        self.critic_of_sensor_nodes_optimizer = [
            optim.Adam(params=self.critic_local_of_sensor_nodes[index].parameters(),
                       lr=self.hyperparameters['critic_of_sensor']['learning_rate'],
                       eps=1e-4
            )for index in range(self.environment.vehicle_number)
        ]


        #
        # self.critic_local_of_sensor_nodes = []
        # for index in range(self.environment.vehicle_number):
        #     self.critic_local_of_sensor_nodes.append(
        #         self.create_NN_for_critic_network_of_sensor_node(
        #             input_dim=self.environment.get_critic_size_for_sensor(),
        #             output_dim=1
        #         )
        #     )
        # self.critic_target_of_sensor_nodes = []
        # for index in range(self.environment.vehicle_number):
        #     self.critic_target_of_sensor_nodes.append(
        #         self.create_NN_for_critic_network_of_sensor_node(
        #             input_dim=self.environment.get_critic_size_for_sensor(),
        #             output_dim=1
        #         )
        #     )
        # for index in range(self.environment.vehicle_number):
        #     HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_sensor_nodes[index],
        #                                  to_model=self.critic_target_of_sensor_nodes[index])
        # self.critic_of_sensor_nodes_optimizer = []
        # for index in range(self.environment.vehicle_number):
        #     self.critic_of_sensor_nodes_optimizer.append(
        #         optim.Adam(params=self.critic_local_of_sensor_nodes[index].parameters(),
        #                    lr=self.hyperparameters['critic_of_sensor']['learning_rate'],
        #                    eps=1e-4)
        #     )

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
            self.sensor_actions = self.sensor_nodes_pick_actions()  # sensor nodes pick actions
            self.edge_action = self.edge_node_pick_action()
            self.global_action = self.combined_action()
            self.conduct_action()
            self.reward_action = self.reward_function_pick_action()
            self.save_experience()
            self.save_reward_experience()
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.sensor_nodes_learn()
                    self.edge_node_learn()
                    self.reward_function_learn()
                    # states, actions, rewards, next_states, dones = self.sample_experiences()
                    # self.critic_learn(states, actions, rewards, next_states, dones)
                    # self.actor_learn(states)
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step += 1
        self.episode_index += 1


    def sensor_nodes_pick_actions(self):
        """Picks an action using the actor network of each sensor node
        and then adds some noise to it to ensure exploration"""
        for index, sensor_observation in enumerate(self.sensor_observations):
            if self.state['action_time'][index][self.episode_index] == 1:
                sensor_node_state = torch.from_numpy(sensor_observation).float().unsqueeze(0).to(
                    self.device)
                self.actor_local_of_sensor_nodes[index].eval()  # set the model to evaluation state
                with torch.no_grad():  # do not compute the gradient
                    sensor_action = self.actor_local_of_sensor_nodes[index](sensor_node_state).cpu().data.numpy()
                self.actor_local_of_sensor_nodes[index].train()  # set the model to training state
                sensor_action = self.exploration_strategy.perturb_action_for_exploration_purposes(
                    {"action": sensor_action})
                self.sensor_actions[index] = sensor_action


    def edge_node_pick_action(self):
        edge_node_state = torch.from_numpy(self.edge_observation).float().unsqueeze(0).to(self.device)
        self.actor_local_of_edge_node.eval()
        with torch.no_grad():
            edge_action = self.actor_local_of_edge_node(edge_node_state).cpu().data.numpy()
        self.actor_local_of_edge_node.train()
        edge_action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": edge_action})
        self.edge_action = edge_action

    def combined_action(self):

        pass

    def conduct_action(self):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(self.action)
        self.total_episode_score_so_far += self.reward
        # TODO what is clip_rewards
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def reward_function_pick_action(self):
        reward_function_state = torch.from_numpy(self.reward_observation).float().unsqueeze(0).to(self.device)
        self.actor_local_of_reward_function.eval()
        with torch.no_grad():
            reward_function_action = self.actor_local_of_reward_function(reward_function_state).cpu().data.numpy()
        self.actor_local_of_reward_function.train()
        reward_function_action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action":reward_function_action})
        self.reward_action = reward_function_action

    def save_experience(self):
        """
        Saves the recent experience to the experience replay buffer
        :param memory: Buffer
        :param experience: self.state, self.action, self.reward, self.next_state, self.done
        :return: None
        """
        if self.experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HMAIMD.py")
        experience = self.state, self.action, self.reward, self.next_state, self.done
        self.experience_replay_buffer.add_experience(*experience)

    def save_reward_experience(self):
        if self.reward_replay_buffer is None:
            raise Exception("reward_replay_buffer is None, function save_experience at HMAIMD.py")
        reward_experience = self.last_state, self.last_action, self.last_reward_action, self.reward, self.state, self.action
        self.reward_replay_buffer.add_experience(*reward_experience)

    def time_for_critic_and_actor_to_learn(self):
        pass

    def sensor_nodes_learn(self):
        pass

    def edge_node_learn(self):
        pass

    def reward_function_learn(self):
        pass