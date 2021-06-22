# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：envCartPole.py
@Author  ：Neardws
@Date    ：6/20/21 8:55 下午 
"""
import gym
env = gym.make('FetchReach-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()