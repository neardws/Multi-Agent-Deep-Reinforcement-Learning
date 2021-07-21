# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：ExperienceReplayBuffer.py
@Author  ：Neardws
@Date    ：7/7/21 7:35 下午 
"""
from collections import namedtuple, deque
import random
import torch
import numpy as np


class ExperienceReplayBuffer(object):
    """Replay buffer to store past reward experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, device=None):
        """
        Init Replay_buffer
        :param buffer_size: buffer size
        :param batch_size: batch number
        :param seed: seed of random number
        :param device: GPU or CPU
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["sensor_nodes_observation", "edge_node_observation",
                                                                "sensor_nodes_action", "edge_node_action",
                                                                "sensor_nodes_reward", "edge_node_reward",
                                                                "next_sensor_nodes_observation",
                                                                "next_edge_node_observation", "done"])
        random.seed(seed)  # setup random number seed
        # if the device is not settle, then use available GPU, if not, the cpu
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_experience(self, sensor_nodes_observation, edge_node_observation, sensor_nodes_action, edge_node_action,
                       sensor_nodes_reward, edge_node_reward, next_sensor_nodes_observation, next_edge_node_observation,
                       done):
        """
        Adds experience(s) into the replay buffer
        :param sensor_nodes_observation:
        :param edge_node_observation:
        :param sensor_nodes_action:
        :param edge_node_action:
        :param sensor_nodes_reward:
        :param edge_node_reward:
        :param next_sensor_nodes_observation:
        :param next_edge_node_observation:
        :param done:
        :return: None
        """
        experience = self.experience(sensor_nodes_observation, edge_node_observation,
                                     sensor_nodes_action, edge_node_action,
                                     sensor_nodes_reward, edge_node_reward,
                                     next_sensor_nodes_observation, next_edge_node_observation,
                                     done)
        self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """
        Draws a random sample of experience from the replay buffer
        :param num_experiences: the number of experience
        :param separate_out_data_types: True or False, indicate is the return separate
        :return:
        """
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            return self.separate_out_data_types(experiences)
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """
        Puts the sampled experience into the correct format for a PyTorch neural network
        :param experiences:
        :return:
        """
        sensor_nodes_observations = torch.from_numpy(
            np.vstack([e.sensor_nodes_observation for e in experiences if e is not None])).float().to(self.device)
        edge_node_observations = torch.from_numpy(
            np.vstack([e.edge_node_observation for e in experiences if e is not None])).float().to(self.device)
        sensor_nodes_actions = torch.from_numpy(
            np.vstack([e.sensor_nodes_action for e in experiences if e is not None])).float().to(self.device)
        edge_node_actions = torch.from_numpy(
            np.vstack([e.edge_node_action for e in experiences if e is not None])).float().to(self.device)
        sensor_nodes_rewards = torch.from_numpy(
            np.vstack([e.sensor_nodes_reward for e in experiences if e is not None])).float().to(self.device)
        edge_node_rewards = torch.from_numpy(
            np.vstack([e.edge_node_reward for e in experiences if e is not None])).float().to(self.device)
        next_sensor_nodes_observations = torch.from_numpy(
            np.vstack([int(e.next_sensor_nodes_observation) for e in experiences if e is not None])).float().to(
            self.device)
        next_edge_node_observations = torch.from_numpy(
            np.vstack([e.next_edge_node_observation for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        return sensor_nodes_observations, edge_node_observations, sensor_nodes_actions, edge_node_actions, \
            sensor_nodes_rewards, edge_node_rewards, next_sensor_nodes_observations, next_edge_node_observations, dones

    def pick_experiences(self, num_experiences=None):
        """
        random pick experience from memory
        :param num_experiences: the number of experiences
        :return: random samples
        """
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """
        The length of Replay_Buffer equal to the length of memory, i.e., buffer_size
        :return: length of Replay_Buffer
        """
        return len(self.memory)
