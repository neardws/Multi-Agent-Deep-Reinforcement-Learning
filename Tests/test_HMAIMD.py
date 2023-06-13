# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning
@File    ：test_HMAIMD.py
@Author  ：Neardws
@Date    ：7/26/21 2:45 下午
"""
import torch
import numpy as np


def test_1():
    tensor_list = [torch.rand(2, 3) for _ in range(4)]
    print(tensor_list)

    new_tensor = torch.from_numpy(
        np.vstack([e for e in tensor_list if e is not None])).float()
    print(new_tensor)

    new_tensor[0, :] = 1
    print(new_tensor)
    print(type(new_tensor))
    print(new_tensor.shape)
