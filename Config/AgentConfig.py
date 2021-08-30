# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AgentConfig.py
@Author  ：Neardws
@Date    ：7/27/21 10:43 上午 
"""
import numpy as np


class AgentConfig(object):
    """
    Object to hold the config requirements for an agent/game
    :arg
    experience_replay_buffer_buffer_size: buffer size for experiment replay buffer
    experience_replay_buffer_batch_size: batch size for experiment replay buffer
    experience_replay_buffer_seed: random seed for experiment replay buffer
    reward_replay_buffer_buffer_size: buffer size for reward replay buffer
    reward_replay_buffer_batch_size: batch size for reward replay buffer
    reward_replay_buffer_seed: random seed for reward replay buffer
    use_gpu: is the data on the GPU, select devices to run
    nn_seed: random seed for neural network
    environment_seed: random seed for VehicularNetworkEnv
    hyperparameters: hyperparameters for neural network
    file_to_save_data_results: file name to sava data results
    """

    def __init__(self):
        self.experience_replay_buffer_buffer_size = None
        self.experience_replay_buffer_batch_size = None
        self.experience_replay_buffer_seed = None

        self.reward_replay_buffer_buffer_size = None
        self.reward_replay_buffer_batch_size = None
        self.reward_replay_buffer_seed = None

        self.nn_seed = None

        self.hyperparameters = None

        self.file_to_save_data_results = None

    def config(self,
               experience_replay_buffer_buffer_size=100000,
               experience_replay_buffer_batch_size=256,
               experience_replay_buffer_seed=np.random.randint(0, 2 ** 32 - 2),
               reward_replay_buffer_buffer_size=100000,
               reward_replay_buffer_batch_size=256,
               reward_replay_buffer_seed=np.random.randint(0, 2 ** 32 - 2),
               nn_seed=np.random.randint(0, 2 ** 32 - 2),
               hyperparameters=None,
               file_to_save_data_results="Results/HMAIMD_RESULTS.pkl"):
        self.experience_replay_buffer_buffer_size = experience_replay_buffer_buffer_size
        self.experience_replay_buffer_batch_size = experience_replay_buffer_batch_size
        self.experience_replay_buffer_seed = experience_replay_buffer_seed
        self.reward_replay_buffer_buffer_size = reward_replay_buffer_buffer_size
        self.reward_replay_buffer_batch_size = reward_replay_buffer_batch_size
        self.reward_replay_buffer_seed = reward_replay_buffer_seed
        self.nn_seed = nn_seed
        self.hyperparameters = hyperparameters
        self.file_to_save_data_results = file_to_save_data_results
