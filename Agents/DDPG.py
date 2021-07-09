# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：DDPG.py
@Author  ：Neardws
@Date    ：6/17/21 1:58 下午 
"""
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim
from nn_builder.pytorch.NN import NN  # construct a neural network via PyTorch
from Utilities.Data_structures.Replay_Buffer import Replay_Buffer
from Exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration

class DDPG(object):
    """A DDPG Agent"""

    def __init__(self, config):
        self.config = config
        self.hyperparameters = self.config.hyperparameters
        self.action_size = self.config.action_size
        self.state_size = self.state_size


        """Replay Buffer"""
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        """Critic Network"""
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        DDPG.copy_model_over(self.critic_local, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"],
                                           eps=1e-4)
        """Actor Network"""
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        DDPG.copy_model_over(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"],
                                          eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.config)

    def step(self):
        """Runs a step in the game"""
        while not self.done:  # when the episode is not over
            # print("State ", self.state.shape)
            self.action = self.pick_action()  # pick a action
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None: state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()  # set the model to evaluation state
        with torch.no_grad():  # do not compute the gradient
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()  # set the model to training state
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        return action.squeeze(0)

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]: self.reward =  max(min(self.reward, 1.0), -1.0)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        return self.memory.sample()

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def update_learning_rate(self, starting_lr,  optimizer):
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


    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    """"""
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



    def save_experience(self, memory=None, experience=None):
        """
        Saves the recent experience to the memory buffer
        :param memory: Buffer
        :param experience: self.state, self.action, self.reward, self.next_state, self.done
        :return: None
        """
        if memory is None:
            # memory = self.memory
            raise Exception("Memory is None, function save_experience at Base_Agent.py")
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)



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

    def enough_experiences_to_learn_from(self):
        """
        Boolean indicated whether there are enough experiences in the memory buffer to learn from
        :return: Boolean , True or False
        The memory must be overridden
        """
        return len(self.memory) > self.hyperparameters["batch_size"]



