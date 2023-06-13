# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：Gaussian_Exploration.py
@Author  ：Neardws
@Date    ：9/3/21 2:17 下午 
"""
import torch
from torch.distributions.normal import Normal


class Gaussian_Exploration(object):

    """Gaussian noise exploration strategy"""
    def __init__(self, size, hyperparameters, key_to_use=None, device=None):
        self.hyperparameters = hyperparameters[key_to_use]
        self.action_noise_std = self.hyperparameters["action_noise_std"]
        self.action_noise_distribution = Normal(torch.FloatTensor([0.0]).to(device), torch.FloatTensor([self.action_noise_std]).to(device))
        self.action_noise_clipping_range = self.hyperparameters["action_noise_clipping_range"]

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action_noise = self.action_noise_distribution.sample(sample_shape=action.shape)
        action_noise = action_noise.squeeze(-1)
        clipped_action_noise = torch.clamp(action_noise, min=-self.action_noise_clipping_range,
                                           max=self.action_noise_clipping_range)
        action += clipped_action_noise
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        pass
