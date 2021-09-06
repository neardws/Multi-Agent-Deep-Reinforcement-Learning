# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：show_guassian.py
@Author  ：Neardws
@Date    ：9/5/21 11:58 上午 
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal


def demo2():
    mu, sigma = 0.0, 0.005
    sampleNo = 1000
    noise_distribution = Normal(torch.cuda.FloatTensor([mu]),torch.cuda.FloatTensor([sigma]))

    np.random.seed(0)
    action_noise = noise_distribution.sample(sample_shape=[sampleNo]).tolist()

    plt.scatter(list(range(1000)), action_noise, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()


if __name__ == '__main__':
    demo2()
