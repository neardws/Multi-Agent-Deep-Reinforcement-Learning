# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_AVVO.py
@Author  ：Neardws
@Date    ：7/27/21 3:12 下午 
"""
import numpy as np
import torch
from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Agents.Trainer import Trainer


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)


def test_init():
    experiment_config = ExperimentConfig()
    experiment_config.config()

    # for item in experiment_config.__dict__:
    #     print(item)
    #     print(experiment_config.__getattribute__(item))
    #     print(type(experiment_config.__getattribute__(item)))
    #     print()

    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)
    vehicularNetworkEnv.reset()

    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=500))
    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=1000))
    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=1250))
    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=1500))
    a = np.random.uniform(low=vehicularNetworkEnv.config.noise_uncertainty_low_bound,
                          high=vehicularNetworkEnv.config.noise_uncertainty_up_bound)
    print(a)
    print(vehicularNetworkEnv.computer_SNR_wall_by_noise_uncertainty(noise_uncertainty=a))
    # vehicularNetworkEnv.get_mean_and_second_moment_service_time_of_types()

    # for item in vehicularNetworkEnv.__dict__:
    #     print(item)
    #     print(vehicularNetworkEnv.__getattribute__(item))
    #     print(type(vehicularNetworkEnv.__getattribute__(item)))
    #     print()


if __name__ == '__main__':
    test_init()
