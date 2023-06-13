# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：show_noise.py
@Author  ：Neardws
@Date    ：8/15/21 2:33 下午 
"""
from Exploration_strategies.Gaussian_Exploration import Gaussian_Exploration
from Utilities.OU_Noise import OU_Noise
from Exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
import torch
import numpy as np


if __name__ == '__main__':

    hyperparameters = {
        "Actor_of_Sensor": {
            "action_noise_std": 0.05,
            "action_noise_clipping_range": 1.0
        }}

    # sensor_exploration_strategy = OU_Noise_Exploration(size=20,
    #                                                    hyperparameters=hyperparameters,
    #                                                    key_to_use="Actor_of_Sensor")
    # sensor_action = torch.rand(1, 20)
    # print(sensor_action)
    # sensor_action = sensor_exploration_strategy.perturb_action_for_exploration_purposes(
    #     {"action": sensor_action.cpu().data.numpy()})
    # print(sensor_action)
    # print(sensor_action[0][0:10])
    # print(sensor_action[0][10:20])
    # # print(sensor_action[0][9, 19])
    # log_softmax = torch.nn.Softmax(dim=0)
    # print(log_softmax(torch.cuda.FloatTensor(sensor_action[0][0:10])))
    # print(log_softmax(torch.cuda.FloatTensor(sensor_action[0][10:20])))
    # print(torch.cat((log_softmax(torch.cuda.FloatTensor(sensor_action[0][0:10])),
    #                  log_softmax(torch.cuda.FloatTensor(sensor_action[0][10:20]))),
    #                 dim=-1).unsqueeze(0))

    sensor_exploration_strategy = Gaussian_Exploration(size=20,
                                                       hyperparameters=hyperparameters,
                                                       key_to_use="Actor_of_Sensor")
    sensor_action = torch.rand(1, 20).to("cuda")
    print(sensor_action)
    sensor_action = sensor_exploration_strategy.perturb_action_for_exploration_purposes(
        {"action": sensor_action})

    print(sensor_action)

    # noise = OU_Noise(size=100, seed=1, mu=0., theta=0.001, sigma=0.5)
    # for i in range(100):
    #     print(noise.sample())
