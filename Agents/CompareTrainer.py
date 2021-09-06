# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：CompareTrainer.py
@Author  ：Neardws
@Date    ：9/6/21 3:06 下午 
"""
from Utilities.FileOperator import init_compare_file_name


class CompareTrainer(object):
    def __init__(self,
                 num_episodes,
                 init_vehicular_environment,
                 compare_agents):
        self.num_episodes = num_episodes
        self.vehicular_environment = init_vehicular_environment
        self.agents = compare_agents

    def run_games_for_agents(self):
        file_name = init_compare_file_name()
        for agent_class in self.agents:
            agent = agent_class(self.vehicular_environment)
            agent_name = agent_class.name
            temple_result_name = file_name + agent_name + ".csv"
            agent.run_n_episodes(num_episodes=self.num_episodes,
                                 temple_result_name=temple_result_name)
