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
from torch import Tensor
from nn_builder.pytorch.NN import NN  # construct a neural network via PyTorch
from Utilities.Data_structures.Config import Agent_Config
from Utilities.Data_structures.Experience_Replay_Buffer import Experience_Replay_Buffer
from Utilities.Data_structures.Reward_Replay_Buffer import Reward_Replay_Buffer
from Exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv


class HMAIMD_Agent(object):
    """
    Workflow of HMAIMD_Agent

    Step.1 Environments reset to get self.reward_state, self.sensor_nodes_observation, self.edge_node_observation
    Step.2 sensor nodes pick actions according to self.sensor_nodes_observation
    Step.3 edge node pick action according to self.edge_node_observation plus sensor actions at step. 2
    Step.4 combine sensor nodes actions and edge node action into one global action, which type is dict
    Step.5 conduct the global action to environment, and return self.next_sensor_nodes_observation,
           self.next_edge_node_observation, self.next_reward_state, self.reward, self.done
    Step.6 reward pick action according to self.reward_state plus the global action
    Step.7 save replay experience
    Step.8 renew self.reward_state, self.sensor_nodes_observation, self.edge_node_observation according to next parameters
           at step.5
    Step.9 replay step.2 - step.8

    @ TODO  reorganize parameters in config
    """

    def __init__(self, agent_config=Agent_Config(), environment=VehicularNetworkEnv()):
        self.config = agent_config
        self.environment = environment
        self.hyperparameters = self.config.hyperparameters

        """float parameters"""
        self.reward = None
        self.done = None                        # 1 or 0 indicate is episode finished
        """dict() parameters"""
        self.action = None
        """torch.Tensor parameters"""
        self.last_reward_state = None
        self.last_global_action = None           # Combine the Tensor sensor nodes action and edge node action
        self.last_reward_action = None
        self.reward_state = None
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
        self.next_reward_state = None

        self.sensor_nodes_observation, self.edge_node_observation, self.reward_state = environment.reset()

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
        self.experience_replay_buffer = Experience_Replay_Buffer(buffer_size=self.config.experience_replay_buffer_buffer_size,
                                                                 batch_size=self.config.experience_replay_buffer_batch_size,
                                                                 seed=self.config.experience_replay_buffer_seed)

        """Reward Replay Buffer"""
        self.reward_replay_buffer = Reward_Replay_Buffer(buffer_size=self.config.reward_replay_buffer_buffer_size,
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
        self.sensor_observations_size = self.environment.get_sensor_observation_size()
        self.sensor_action_size = self.environment.get_sensor_action_size()
        self.actor_local_of_sensor_nodes = [
            self.create_NN(
                input_dim=self.sensor_observations_size,
                output_dim=[self.environment.data_types_number, self.environment.data_types_number],
                key_to_use="actor_of_sensor"
            ) for _ in range(self.environment.vehicle_number)
        ]

        self.actor_target_of_sensor_nodes = [
            self.create_NN(
                input_dim=self.sensor_observations_size,
                output_dim=[self.environment.data_types_number, self.environment.data_types_number],
                key_to_use="actor_of_sensor"
            ) for _ in range(self.environment.vehicle_number)
        ]

        for index in range(self.environment.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_sensor_nodes[index],
                                         to_model=self.actor_target_of_sensor_nodes[index])
        self.actor_optimizer_of_sensor_nodes = [
            optim.Adam(params=self.actor_local_of_sensor_nodes[index].parameters(),
                       lr=self.hyperparameters['actor_of_sensor']['learning_rate'],
                       eps=1e-4
                       ) for index in range(self.environment.vehicle_number)
        ]

        """Critic Network of Sensor Nodes"""
        self.critic_size_for_sensor = self.environment.get_critic_size_for_sensor()
        self.critic_local_of_sensor_nodes = [
            self.create_NN(
                input_dim=self.critic_size_for_sensor,
                output_dim=1,
                key_to_use="critic_of_sensor"
            ) for _ in range(self.environment.vehicle_number)
        ]

        self.critic_target_of_sensor_nodes = [
            self.create_NN(
                input_dim=self.critic_size_for_sensor,
                output_dim=1,
                key_to_use="critic_of_sensor"
            ) for _ in range(self.environment.vehicle_number)
        ]

        for index in range(self.environment.vehicle_number):
            HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_sensor_nodes[index],
                                         to_model=self.critic_target_of_sensor_nodes[index])
        self.critic_optimizer_of_sensor_nodes = [
            optim.Adam(params=self.critic_local_of_sensor_nodes[index].parameters(),
                       lr=self.hyperparameters['critic_of_sensor']['learning_rate'],
                       eps=1e-4
                       ) for index in range(self.environment.vehicle_number)
        ]

        """Actor Network for Edge Node"""
        self.actor_local_of_edge_node = self.create_NN(
            input_dim=self.environment.get_actor_input_size_for_edge(),
            output_dim=self.environment.get_edge_action_size(),
            key_to_use="actor_of_edge"
        )
        self.actor_target_of_edge_node = self.create_NN(
            input_dim=self.environment.get_actor_input_size_for_edge(),
            output_dim=self.environment.get_edge_action_size(),
            key_to_use="actor_of_edge"
        )
        HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_edge_node,
                                     to_model=self.actor_target_of_edge_node)
        self.actor_optimizer_of_edge_node = optim.Adam(
            params=self.actor_local_of_edge_node.parameters(),
            lr=self.hyperparameters['actor_of_edge']['learning_rate'],
            eps=1e-4
        )

        """Critic Network for Edge Node"""
        self.critic_local_of_edge_node = self.create_NN(
            input_dim=self.environment.get_critic_size_for_edge(),
            output_dim=1,
            key_to_use="critic_of_edge"
        )
        self.critic_target_of_edge_node = self.create_NN(
            input_dim=self.environment.get_critic_size_for_edge(),
            output_dim=1,
            key_to_use="critic_of_edge"
        )
        HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_edge_node,
                                     to_model=self.critic_target_of_edge_node)
        self.critic_optimizer_of_edge_node = optim.Adam(
            params=self.critic_local_of_edge_node.parameters(),
            lr=self.hyperparameters['critic_of_edge_node']['learning_rate'],
            eps=1e-4
        )

        """Actor Network for Reward Function"""
        self.actor_local_of_reward_function = self.create_NN(
            input_dim=self.environment.get_actor_input_size_for_reward(),
            output_dim=self.environment.get_reward_action_size(),
            key_to_use="actor_of_reward"
        )
        self.actor_target_of_reward_function = self.create_NN(
            input_dim=self.environment.get_actor_input_size_for_reward(),
            output_dim=self.environment.get_reward_action_size(),
            key_to_use="actor_of_reward"
        )
        HMAIMD_Agent.copy_model_over(from_model=self.actor_local_of_reward_function,
                                     to_model=self.actor_target_of_reward_function)
        self.actor_optimizer_of_reward_function = optim.Adam(
            params=self.actor_local_of_reward_function.parameters(),
            lr=self.hyperparameters['actor_of_reward_function']['learning_rate'],
            eps=1e-4
        )

        """Critic Network for Reward Function"""
        self.critic_local_of_reward_function = self.create_NN(
            input_dim=self.environment.get_critic_size_for_reward(),
            output_dim=1,
            key_to_use="critic_of_reward"
        )
        self.critic_target_of_reward_function = self.create_NN(
            input_dim=self.environment.get_critic_size_for_reward(),
            output_dim=1,
            key_to_use="critic_of_reward"
        )
        HMAIMD_Agent.copy_model_over(from_model=self.critic_local_of_reward_function,
                                     to_model=self.critic_target_of_reward_function)
        self.critic_optimizer_of_reward_function = optim.Adam(
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
        if override_seed:
            seed = override_seed
        else:
            seed = self.config.seed

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
            self.sensor_nodes_pick_actions(sensor_nodes_observation=self.sensor_nodes_observation)  # sensor nodes pick actions
            self.edge_node_pick_action(edge_node_observation=self.edge_node_observation,
                                       sensor_nodes_action=self.sensor_nodes_action)
            self.combined_action()
            self.conduct_action()
            self.reward_function_pick_action(reward_state=self.reward_state,
                                             global_action=self.global_action)
            self.save_experience()
            self.save_reward_experience()
            if self.time_for_critic_and_actor_of_sensor_nodes_and_edge_node_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):

                    sensor_nodes_observations, edge_node_observations, sensor_actions, edge_actions, sensor_nodes_rewards, \
                    edge_node_rewards, next_sensor_nodes_observations, next_edge_node_observations, dones = self.sample_experiences(
                        "experience_replay_buffer")

                    self.sensor_nodes_and_edge_node_to_learn(sensor_nodes_observations=sensor_nodes_observations,
                                                             edge_node_observations=edge_node_observations,
                                                             sensor_actions=sensor_actions,
                                                             edge_actions=edge_actions,
                                                             sensor_nodes_rewards=sensor_nodes_rewards,
                                                             edge_node_rewards=edge_node_rewards,
                                                             next_sensor_nodes_observations=next_sensor_nodes_observations,
                                                             next_edge_node_observations=next_edge_node_observations,
                                                             dones=dones)

            if self.time_for_critic_and_actor_of_reward_function_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    last_reward_states, last_global_actions, last_reward_actions, rewards, reward_states, global_actions, dones = self.sample_experiences("reward_replay_buffer")
                    self.reward_function_to_learn(last_reward_states=last_reward_states, last_global_actions=last_global_actions, last_reward_actions=last_reward_actions,
                                                  rewards=rewards, reward_states=reward_states, global_actions=global_actions, dones=dones)

            """Renew by reward function"""
            self.last_reward_state = self.reward_state
            self.last_global_action = self.global_action
            self.last_reward_action = self.reward_action

            """Renew by environment"""
            self.sensor_nodes_observation = self.next_sensor_nodes_observation
            self.edge_node_observation = self.next_edge_node_observation
            self.reward_state = self.next_reward_state

            self.episode_step += 1
        self.episode_index += 1

    def sample_experiences(self, buffer_name):
        if buffer_name == "experience_replay_buffer":
            return self.experience_replay_buffer.sample()
        elif buffer_name == "reward_replay_buffer":
            return self.reward_replay_buffer.sample()
        else:
            raise Exception("Buffer name is Wrong")

    def sensor_nodes_pick_actions(self, sensor_nodes_observation):
        """Picks an action using the actor network of each sensor node
        and then adds some noise to it to ensure exploration"""
        for sensor_node_index in range(self.environment.vehicle_number):
            if self.environment.state['action_time'][sensor_node_index][self.episode_index] == 1:
                sensor_node_observation = sensor_nodes_observation[sensor_node_index, :].unsqueeze(0)
                self.actor_local_of_sensor_nodes[sensor_node_index].eval()  # set the model to evaluation state
                with torch.no_grad():  # do not compute the gradient
                    sensor_action = self.actor_local_of_sensor_nodes[sensor_node_index](sensor_node_observation)
                self.actor_local_of_sensor_nodes[sensor_node_index].train()  # set the model to training state
                sensor_action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": sensor_action})
                self.sensor_nodes_action[sensor_node_index,:] = sensor_action

    def edge_node_pick_action(self, edge_node_observation, sensor_nodes_action):
        edge_node_state = torch.cat((edge_node_observation, sensor_nodes_action), 1).unsqueeze(0)
        self.actor_local_of_edge_node.eval()
        with torch.no_grad():
            edge_action = self.actor_local_of_edge_node(edge_node_state)
        self.actor_local_of_edge_node.train()
        self.edge_action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": edge_action})

    def combined_action(self, sensor_nodes_action=torch.empty(), edge_node_action=torch.empty()):

        self.global_action = torch.cat(
            (sensor_nodes_action[0,:], sensor_nodes_action[1,:]), dim=1)
        for sensor_node_index in range(self.environment.vehicle_number):
            if sensor_node_index > 1:
                self.global_action = torch.cat(
                    (self.global_action, sensor_nodes_action[sensor_node_index,:]), dim=1)
        self.global_action = torch.cat((self.global_action, edge_node_action), dim=1)

        priority = np.zeros(shape=(self.environment.vehicle_number, self.environment.data_types_number), dtype=np.float)
        arrival_rate = np.zeros(shape=(self.environment.vehicle_number, self.environment.data_types_number), dtype=np.float)

        for sensor_node_index in range(self.environment.vehicle_number):
            sensor_node_action = sensor_nodes_action[sensor_node_index,:]
            sensor_node_action_of_priority = sensor_node_action[0:self.environment.data_types_number-1]
            sensor_node_action_of_arrival_rate = sensor_node_action[self.environment.data_types_number:-1]
            for data_type_index in range(self.environment.data_types_number):
                if self.environment.state['data_types'][sensor_node_index][data_type_index] == 1:
                    priority[sensor_node_index][data_type_index] = sensor_node_action_of_priority[data_type_index]
                    arrival_rate[sensor_node_index][data_type_index] = \
                        float(sensor_node_action_of_arrival_rate[data_type_index]) / self.environment.mean_service_time_of_types[data_type_index]

        edge_nodes_bandwidth = edge_node_action.numpy() * self.environment.bandwidth

        self.action = {"priority": priority,
                       "arrival_rate": arrival_rate,
                       "edge_nodes_bandwidth": edge_nodes_bandwidth}

    def conduct_action(self):
        """Conducts an action in the environment"""
        self.next_reward_state, self.next_sensor_nodes_observation, self.next_edge_node_observation, self.reward, self.done \
            = self.environment.step(self.action)
        self.total_episode_score_so_far += self.reward


    def reward_function_pick_action(self, reward_state, global_action):
        reward_function_state = torch.cat((reward_state, global_action), 1).unsqueeze(0)
        self.actor_local_of_reward_function.eval()
        with torch.no_grad():
            reward_function_action = self.actor_local_of_reward_function(reward_function_state)
        self.actor_local_of_reward_function.train()
        self.reward_action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": reward_function_action})
        self.sensor_nodes_reward = self.reward * self.reward_action[:self.environment.vehicle_number-1]
        self.edge_node_reward = self.reward * self.reward_action[-1]

    def save_experience(self):

        # TODO Renew structure of experience and replay buffer
        """
        sensor_nodes_observations=torch.empty(), sensor_actions=torch.empty(),
                           sensor_nodes_rewards=torch.empty(), next_sensor_nodes_observations=torch.empty(),
                           dones=torch.empty()
        Saves the recent experience to the experience replay buffer
        :param memory: Buffer
        :param experience: self.state, self.action, self.reward, self.next_state, self.done
        :return: None
        """
        if self.experience_replay_buffer is None:
            raise Exception("experience_replay_buffer is None, function save_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        experience = self.sensor_nodes_observation, self.edge_node_observation,\
                     self.sensor_nodes_action, self.edge_node_action, \
                     self.sensor_nodes_reward, self.edge_node_reward,\
                     self.next_sensor_nodes_observation, self.next_edge_node_observation, self.done
        self.experience_replay_buffer.add_experience(*experience)

    def save_reward_experience(self):
        if self.reward_replay_buffer is None:
            raise Exception("reward_replay_buffer is None, function save_experience at HMAIMD.py")
        """Save as torch.Tensor"""
        reward_experience = self.last_reward_state, self.last_global_action, self.last_reward_action, \
                            self.reward, self.reward_state, self.global_action, self.done
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

    def sensor_nodes_and_edge_node_to_learn(self, sensor_nodes_observations=torch.empty(),
                                            edge_node_observations=torch.empty(),
                                            sensor_actions=torch.empty(), edge_actions=torch.empty(),
                                            sensor_nodes_rewards=torch.empty(), edge_node_rewards=torch.empty(),
                                            next_sensor_nodes_observations=torch.empty(),
                                            next_edge_node_observations=torch.empty(),
                                            dones=torch.empty()):

        sensor_nodes_actions_next_list = [
            self.actor_target_of_sensor_nodes[sensor_node_index](next_sensor_nodes_observations[sensor_node_index,:])
            for sensor_node_index in range(self.environment.vehicle_number)]

        sensor_nodes_actions_next_tensor = torch.cat(
            (sensor_nodes_actions_next_list[0], sensor_nodes_actions_next_list[1]), dim=1)
        for index, sensor_nodes_actions_next in enumerate(sensor_nodes_actions_next_list):
            if index > 1:
                sensor_nodes_actions_next_tensor = torch.cat(
                    (sensor_nodes_actions_next_tensor, sensor_nodes_actions_next), dim=1)

        for sensor_node_index in range(self.environment.vehicle_number):
            sensor_node_observations = sensor_nodes_observations[sensor_node_index, :]
            sensor_node_rewards = sensor_nodes_rewards[sensor_node_index, :]
            next_sensor_node_observations = next_sensor_nodes_observations[sensor_node_index, :]

            """Runs a learning iteration for the critic"""
            """Computes the loss for the critic"""
            with torch.no_grad():
                critic_targets_next_of_sensor_node = self.critic_target_of_sensor_nodes[sensor_node_index](
                    torch.cat(next_sensor_node_observations, sensor_nodes_actions_next_tensor), dim=1)
                critic_targets_of_sensor_node = sensor_node_rewards + (
                            self.hyperparameters["discount_rate"] * critic_targets_next_of_sensor_node * (1.0 - dones))
            critic_expected_of_sensor_node = self.critic_local_of_sensor_nodes[sensor_node_index](
                torch.cat((sensor_node_observations, sensor_actions), dim=1))
            critic_loss_of_sensor_node: Tensor = functional.mse_loss(critic_expected_of_sensor_node,
                                                                     critic_targets_of_sensor_node)

            """Update target critic networks"""
            self.take_optimisation_step(self.critic_optimizer_of_sensor_nodes[sensor_node_index],
                                        self.critic_local_of_sensor_nodes[sensor_node_index],
                                        critic_loss_of_sensor_node,
                                        self.hyperparameters["critic_of_sensor"]["gradient_clipping_norm"])
            self.soft_update_of_target_network(self.critic_local_of_sensor_nodes[sensor_node_index],
                                               self.critic_target_of_sensor_nodes[sensor_node_index],
                                               self.hyperparameters["critic_of_sensor"]["tau"])

            """Runs a learning iteration for the actor"""
            if self.done:  # we only update the learning rate at end of each episode
                self.update_learning_rate(self.hyperparameters["actor_of_sensor"]["learning_rate"],
                                          self.actor_optimizer_of_sensor_nodes[sensor_node_index])
            """Calculates the loss for the actor"""
            actions_predicted_of_sensor_node = self.actor_local_of_sensor_nodes[sensor_node_index](
                sensor_node_observations)
            if sensor_node_index == 0:
                sensor_nodes_actions_add_actions_pred = torch.cat(
                    (actions_predicted_of_sensor_node, sensor_actions[1:self.environment.vehicle_number, :]), dim=1)
            elif sensor_node_index == self.environment.vehicle_number - 1:
                sensor_nodes_actions_add_actions_pred = torch.cat(
                    (sensor_actions[0:self.environment.vehicle_number - 1, :], actions_predicted_of_sensor_node), dim=1)
            else:
                sensor_nodes_actions_add_actions_pred = torch.cat((sensor_actions[0:sensor_node_index - 1, :],
                                                                   actions_predicted_of_sensor_node,
                                                                   sensor_actions[sensor_node_index + 1:self.environment.vehicle_number,
                                                                                 :]), dim=1)

            actor_loss_of_sensor_node = -self.critic_local_of_sensor_nodes[sensor_node_index](
                torch.cat((sensor_node_observations, sensor_nodes_actions_add_actions_pred), dim=1)).mean()

            self.take_optimisation_step(self.actor_optimizer_of_sensor_nodes[sensor_node_index],
                                        self.actor_local_of_sensor_nodes[sensor_node_index],
                                        actor_loss_of_sensor_node,
                                        self.hyperparameters["actor_of_sensor"]["gradient_clipping_norm"])
            self.soft_update_of_target_network(self.actor_local_of_sensor_nodes[sensor_node_index],
                                               self.actor_target_of_sensor_nodes[sensor_node_index],
                                               self.hyperparameters["actor_of_sensor"]["tau"])

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
            torch.cat((edge_node_observations, sensor_actions, edge_actions), 1))
        loss_of_edge_node = functional.mse_loss(critic_expected_of_edge_node, critic_targets_of_edge_node)
        self.take_optimisation_step(self.critic_optimizer_of_edge_node,
                                    self.critic_local_of_edge_node,
                                    loss_of_edge_node,
                                    self.hyperparameters["critic_of_edge"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_of_edge_node, self.critic_target_of_edge_node,
                                           self.hyperparameters["critic_of_edge"]["tau"])

        """Runs a learning iteration for the actor of edge node"""
        if self.done:  # we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["actor_of_edge"]["learning_rate"],
                                      self.actor_optimizer_of_edge_node)
        """Calculates the loss for the actor"""
        actions_predicted_of_edge_node = self.actor_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_actions), dim=1))
        actor_loss_of_edge_node = -self.critic_local_of_edge_node(
            torch.cat((edge_node_observations, sensor_actions, actions_predicted_of_edge_node), dim=1)).mean()
        self.take_optimisation_step(self.actor_optimizer_of_edge_node, self.actor_local_of_edge_node,
                                    actor_loss_of_edge_node,
                                    self.hyperparameters["actor_of_edge"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local_of_edge_node, self.actor_target_of_edge_node,
                                           self.hyperparameters["actor_of_edge"]["tau"])

    def reward_function_to_learn(self, last_reward_states=torch.empty(), last_global_actions=torch.empty(),
                              last_reward_actions=torch.empty(), rewards=torch.empty(),
                              reward_states=torch.empty(), global_actions=torch.empty(), dones=torch.empty()):

        """Runs a learning iteration for the critic of reward function"""
        with torch.no_grad():
            reward_actions_next = self.actor_target_of_reward_function(torch.cat((reward_states, global_actions), dim=1))
            critic_targets_next = self.critic_target_of_reward_function(torch.cat((reward_states, global_actions, reward_actions_next), 1))
            critic_targets = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        critic_expected = self.critic_local_of_reward_function(torch.cat((last_reward_states, last_global_actions, last_reward_actions), 1))
        loss = functional.mse_loss(critic_expected, critic_targets)
        self.take_optimisation_step(self.critic_optimizer_of_reward_function,
                                    self.critic_local_of_reward_function, loss,
                                    self.hyperparameters["critic_of_reward"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_of_reward_function, self.critic_target_of_reward_function,
                                           self.hyperparameters["critic_of_reward"]["tau"])

        """Runs a learning iteration for the actor"""
        if self.done:  # we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["actor_of_reward"]["learning_rate"], self.actor_optimizer_of_reward_function)
        """Calculates the loss for the actor"""
        actions_predicted = self.actor_local_of_reward_function(torch.cat((last_reward_states, last_global_actions), dim=1))
        actor_loss = -self.critic_local_of_reward_function(torch.cat((last_reward_states, last_global_actions, actions_predicted), dim=1)).mean()
        self.take_optimisation_step(self.actor_optimizer_of_reward_function, self.actor_local_of_reward_function, actor_loss,
                                    self.hyperparameters["actor_of_reward"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local_of_reward_function, self.actor_target_of_reward_function,
                                           self.hyperparameters["actor_of_reeard"]["tau"])


    def update_learning_rate(self, starting_lr, optimizer):
        """
        Lowers the learning rate according to how close we are to the solution
        The learning rate is smaller when closer the solution
        However, we must determine the average score required to win
        :param starting_lr:  learning rate of starting
        :param optimizer:
        :return:
        """
        new_lr = starting_lr
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if np.random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()  # this applies the gradients

    def soft_update_of_target_network(self, local_model, target_model, tau):
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

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.seed)
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))