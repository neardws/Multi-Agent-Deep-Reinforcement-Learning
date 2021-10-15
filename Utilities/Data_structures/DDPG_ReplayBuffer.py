#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DDPG_ReplayBuffer.py
@Time    :   2021/09/16 17:06:10
@Author  :   Neardws
@Version :   1.0
@Contact :   neard.ws@gmail.com
@Desc    :   None
'''

from collections import namedtuple, deque
import random
import torch
import numpy as np


class DDPG_ReplayBuffer(object):
    """Replay buffer to store past reward experiences that the agent can then use for training data"""

    experience = namedtuple("Experience", field_names=["observation", "action", "reward", "next_observation", "done"])
    experience.__qualname__ = 'DDPG_ReplayBuffer.experience'

    def __init__(self, buffer_size, batch_size, seed, device=None):
        """
        Init Replay_buffer
        :param buffer_size: buffer size
        :param batch_size: batch number
        :param seed: seed of random number
        :param device: GPU or CPU
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)

        random.seed(seed)  # setup random number seed
        # if the device is not settle, then use available GPU, if not, the cpu
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add_experience(self, observation, action, reward, next_observation, done):
        """
        Adds experience(s) into the replay buffer
        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        :return: None
        """
        experience = self.experience(observation, action, reward, next_observation, done)
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
        observations = torch.from_numpy(np.vstack([e.observation.cpu().data for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action.cpu().data for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_observations = torch.from_numpy(np.vstack([e.next_observation.cpu().data for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        return observations, actions, rewards, next_observations, dones

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
