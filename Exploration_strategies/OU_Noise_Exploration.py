# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：OU_Noise_Exploration.py
@Author  ：Neardws
@Date    ：7/2/21 2:29 下午 
"""
from Utilities.OU_Noise import OU_Noise


# noinspection PyPep8Naming
class OU_Noise_Exploration(object):
    """Ornstein-Uhlenbeck noise process exploration strategy"""

    def __init__(self, size, hyperparameters, key_to_use=None):
        self.noise = OU_Noise(size,
                              hyperparameters[key_to_use]['noise_seed'],
                              hyperparameters[key_to_use]['mu'],
                              hyperparameters[key_to_use]['theta'],
                              hyperparameters[key_to_use]['sigma'])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        # raise ValueError("Must be implemented")
        pass

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
