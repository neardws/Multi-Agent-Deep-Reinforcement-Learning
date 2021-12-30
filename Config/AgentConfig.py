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
        self.actor_experience_replay_buffer_buffer_size = None
        self.actor_experience_replay_buffer_batch_size = None
        self.actor_experience_replay_buffer_seed = None
        self.actor_experience_replay_buffer_dropout = None

        self.critic_experience_replay_buffer_buffer_size = None
        self.critic_experience_replay_buffer_batch_size = None
        self.critic_experience_replay_buffer_seed = None
        self.critic_experience_replay_buffer_dropout = None

        self.actor_reward_replay_buffer_buffer_size = None
        self.actor_reward_replay_buffer_batch_size = None
        self.actor_reward_replay_buffer_seed = None
        self.actor_reward_replay_buffer_dropout = None

        self.critic_reward_replay_buffer_buffer_size = None
        self.critic_reward_replay_buffer_batch_size = None
        self.critic_reward_replay_buffer_seed = None
        self.critic_reward_replay_buffer_dropout = None

        self.nn_seed = None

        self.hyperparameters = None

        self.file_to_save_data_results = None

    def config(
        self,
        actor_experience_replay_buffer_buffer_size=100000,
        actor_experience_replay_buffer_batch_size=512,
        actor_experience_replay_buffer_seed=2801641275,
        # actor_experience_replay_buffer_seed=np.random.randint(0, 2 ** 32 - 2),
        actor_experience_replay_buffer_dropout=0,
        critic_experience_replay_buffer_buffer_size=200000,
        critic_experience_replay_buffer_batch_size=512,
        critic_experience_replay_buffer_seed=1097855444,
        # critic_experience_replay_buffer_seed=np.random.randint(0, 2 ** 32 - 2),
        critic_experience_replay_buffer_dropout=0,
        actor_reward_replay_buffer_buffer_size=100000,
        actor_reward_replay_buffer_batch_size=512,
        actor_reward_replay_buffer_seed=1328027828,
        # actor_reward_replay_buffer_seed=np.random.randint(0, 2 ** 32 - 2),
        actor_reward_replay_buffer_dropout=0,
        critic_reward_replay_buffer_buffer_size=200000,
        critic_reward_replay_buffer_batch_size=512,
        critic_reward_replay_buffer_seed=2419977517,
        # critic_reward_replay_buffer_seed=np.random.randint(0, 2 ** 32 - 2),
        critic_reward_replay_buffer_dropout=0,
        # nn_seed=3523186978,
        nn_seed=197538830,
        # nn_seed=np.random.randint(0, 2 ** 32 - 2),
        hyperparameters=None,
        file_to_save_data_results="Results/HMAIMD_RESULTS.pkl"):

        self.actor_experience_replay_buffer_buffer_size = actor_experience_replay_buffer_buffer_size
        self.actor_experience_replay_buffer_batch_size = actor_experience_replay_buffer_batch_size
        self.actor_experience_replay_buffer_seed = actor_experience_replay_buffer_seed
        self.actor_experience_replay_buffer_dropout = actor_experience_replay_buffer_dropout

        self.critic_experience_replay_buffer_buffer_size = critic_experience_replay_buffer_buffer_size
        self.critic_experience_replay_buffer_batch_size = critic_experience_replay_buffer_batch_size
        self.critic_experience_replay_buffer_seed = critic_experience_replay_buffer_seed
        self.critic_experience_replay_buffer_dropout = critic_experience_replay_buffer_dropout

        self.actor_reward_replay_buffer_buffer_size = actor_reward_replay_buffer_buffer_size
        self.actor_reward_replay_buffer_batch_size = actor_reward_replay_buffer_batch_size
        self.actor_reward_replay_buffer_seed = actor_reward_replay_buffer_seed
        self.actor_reward_replay_buffer_dropout = actor_reward_replay_buffer_dropout

        self.critic_reward_replay_buffer_buffer_size = critic_reward_replay_buffer_buffer_size
        self.critic_reward_replay_buffer_batch_size = critic_reward_replay_buffer_batch_size
        self.critic_reward_replay_buffer_seed = critic_reward_replay_buffer_seed
        self.critic_reward_replay_buffer_dropout = critic_reward_replay_buffer_dropout

        self.nn_seed = nn_seed
        self.hyperparameters = hyperparameters
        self.file_to_save_data_results = file_to_save_data_results
