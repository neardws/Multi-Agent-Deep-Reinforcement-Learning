# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：Reward_Replay_Buffer.py
@Author  ：Neardws
@Date    ：7/7/21 7:35 下午 
"""
from collections import namedtuple, deque
import random
import torch
import numpy as np


class RewardReplayBuffer(object):
    """Replay buffer to store past reward experiences that the agent can then use for training data"""

    # module-level type definition
    experience = namedtuple("Experience", field_names=["last_reward_observation", "last_global_action",
                                                       "last_reward_action", "reward", "reward_observation",
                                                       "global_action", "done"])
    experience.__qualname__ = 'RewardReplayBuffer.experience'

    def __init__(self, buffer_size, batch_size, seed, dropout, device=None):
        """
        Init Replay_buffer
        :param buffer_size: buffer size
        :param batch_size: batch number
        :param seed: seed of random number
        :param device: GPU or CPU
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dropout = dropout
        self.memory = deque(maxlen=self.buffer_size)

        random.seed(seed)  # setup random number seed
        # if the device is not settle, then use available GPU, if not, the cpu
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_experience(self, last_reward_observation, last_global_action, last_reward_action,
                       reward, reward_observation, global_action, done):
        """
        Adds experience(s) into the replay buffer
        :param last_reward_observation:
        :param last_global_action:
        :param last_reward_action:
        :param reward:
        :param reward_observation:
        :param global_action:
        :param done:
        :return:
        """
        experience = self.experience(last_reward_observation, last_global_action, last_reward_action,
                                     reward, reward_observation, global_action, done)
        if self.__len__() == self.buffer_size:
            if self.dropout != 0:
                size = self.buffer_size * self.dropout
                for i in range(int(size)):
                    self.memory.pop()
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
            last_reward_observation, last_global_actions, last_reward_actions, \
                rewards, reward_observation, global_actions, dones = self.separate_out_data_types(experiences)
            return last_reward_observation, last_global_actions, last_reward_actions, \
                rewards, reward_observation, global_actions, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """
        Puts the sampled experience into the correct format for a PyTorch neural network
        :param experiences: Input
        :return:/
        """
        last_reward_observations = torch.from_numpy(np.vstack([e.last_reward_observation.cpu().data for e in experiences if e is not None]))\
            .float().to(self.device)
        last_global_actions = torch.from_numpy(np.vstack([e.last_global_action.cpu().data for e in experiences if e is not None]))\
            .float().to(self.device)
        last_reward_actions = torch.from_numpy(np.vstack([e.last_reward_action.cpu().data for e in experiences if e is not None]))\
            .float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        reward_observations = torch.from_numpy(np.vstack([e.reward_observation.cpu().data for e in experiences if e is not None]))\
            .float().to(self.device)
        global_actions = torch.from_numpy(np.vstack([e.global_action.cpu().data for e in experiences if e is not None])).float()\
            .to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return last_reward_observations, last_global_actions, last_reward_actions, \
            rewards, reward_observations, global_actions, dones

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
