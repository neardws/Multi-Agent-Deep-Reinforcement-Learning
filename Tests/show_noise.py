# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：show_noise.py
@Author  ：Neardws
@Date    ：8/15/21 2:33 下午 
"""
from Utilities.OU_Noise import OU_Noise

if __name__ == '__main__':
    noise = OU_Noise(size=100, seed=1, mu=0., theta=0.001, sigma=0.5)
    print(noise.sample())
