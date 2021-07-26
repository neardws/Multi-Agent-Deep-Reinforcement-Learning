# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_VehicularNetworkEnv.py
@Author  ：Neardws
@Date    ：7/26/21 2:45 下午 
"""
from torch import Tensor
import torch

if __name__ == '__main__':

    a = torch.rand(3, 4)
    b = torch.rand(3, 4)
    print(a)
    print(b)

    a = torch.cat((a, b), dim=0)
    print(a)
    print(a.shape)

    a = torch.cat((a, b), dim=0)
    print(a)
    print(a.shape)
