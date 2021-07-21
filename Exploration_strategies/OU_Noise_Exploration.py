# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：OU_Noise_Exploration.py
@Author  ：Neardws
@Date    ：7/2/21 2:29 下午 
"""

import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), '..'))
from Utilities.OU_Noise import OU_Noise
from Exploration_strategies.BaseExplorationStrategy import BaseExplorationStrategy


# noinspection PyPep8Naming
class OU_Noise_Exploration(BaseExplorationStrategy):
    """Ornstein-Uhlenbeck noise process exploration strategy"""

    def __init__(self, config):
        super().__init__(config)
        self.noise = OU_Noise(self.config.noise_action_size, self.config.noise_seed, self.config.noise_mu,
                              self.config.noise_theta, self.config.noise_sigma)

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
