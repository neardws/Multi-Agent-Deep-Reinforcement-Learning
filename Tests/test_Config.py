# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_Config.py
@Author  ：Neardws
@Date    ：7/27/21 2:51 下午 
"""
from Config.AgentConfig import AgentConfig


def test_AgentConfig():
    a = AgentConfig()
    a.config()
    print(a.__dict__)


if __name__ == '__main__':
    test_AgentConfig()