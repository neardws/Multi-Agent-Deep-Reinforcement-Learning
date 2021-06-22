# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：env_stu.py
@Author  ：Neardws
@Date    ：6/20/21 11:10 下午 
"""
import gym
import numpy as np
from gym import spaces
from gym.vector.tests.utils import HEIGHT, WIDTH


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, N_DISCRETE_ACTIONS=None, N_CHANNELS=None):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
        pass
    # Execute one time step within the environment...

    def reset(self):
        pass
    # Reset the state of the environment to an initial state...

    def render(self, mode='human', close=False):
        pass
    # Render the environment to the screen