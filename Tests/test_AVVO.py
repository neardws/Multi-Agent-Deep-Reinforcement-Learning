# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_AVVO.py
@Author  ：Neardws
@Date    ：7/27/21 3:12 下午 
"""
import numpy as np
from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Agents.Trainer import Trainer
np.set_printoptions(threshold=np.inf)


def test_init():
    experiment_config = ExperimentConfig()
    experiment_config.config()

    for item in experiment_config.__dict__:
        print(item)
        print(experiment_config.__getattribute__(item))
        print(type(experiment_config.__getattribute__(item)))
        print()

    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)
    vehicularNetworkEnv.reset()
    for item in vehicularNetworkEnv.__dict__:
        print(item)
        print(vehicularNetworkEnv.__getattribute__(item))
        print(type(vehicularNetworkEnv.__getattribute__(item)))
        print()


if __name__ == '__main__':
    test_init()
