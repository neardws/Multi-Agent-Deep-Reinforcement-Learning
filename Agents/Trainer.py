# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：Trainer.py
@Author  ：Neardws
@Date    ：7/1/21 10:11 上午 
"""
import pickle
from Utilities.Data_structures.Config import Agent_Config
from Agents.HMAIMD import HMAIMD_Agent

class Trainer(object):
    """
    Runs game for given agent
    """
    def __init__(self,
                 agent_config=Agent_Config(),
                 agent = HMAIMD_Agent()):
        self.config = agent_config
        self.agent = agent


    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

    def save_obj(self, obj, name):
        """
        Saves given object as a pickle file
        :param obj:
        :param name:
        :return:
        """
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """
        Loads a pickle file object
        :param name:
        :return:
        """
        with open(name, 'rb') as f:
            return pickle.load(f)

    def run_games_for_agent(self):

        game_scores, rolling_scores, time_taken = self.agent.run_n_episodes()
        print("Time taken: {}".format(time_taken), flush=True)

        self.print_two_empty_lines()
        agent_result = [game_scores, rolling_scores, len(rolling_scores), max(rolling_scores), time_taken]
        self.save_obj(agent_result, self.config.file_to_save_data_results)
