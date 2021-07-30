# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_Config.py
@Author  ：Neardws
@Date    ：7/27/21 2:51 下午 
"""
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Environments.VehicularNetworkEnv.envs.VehicularNetworkEnv import VehicularNetworkEnv


def test_AgentConfig():
    a = AgentConfig()
    a.config()
    print(a.__dict__)


def test_ExperimentConfig():
    print(VehicularNetworkEnv.cover_dBm_to_W(-70))
    print(VehicularNetworkEnv.cover_dBm_to_W(20))

    print(VehicularNetworkEnv.cover_dB_to_ratio(2))
    print(VehicularNetworkEnv.cover_dB_to_ratio(-20))
    print(VehicularNetworkEnv.cover_dB_to_ratio(-14))


if __name__ == '__main__':
    # test_AgentConfig()
    test_ExperimentConfig()
