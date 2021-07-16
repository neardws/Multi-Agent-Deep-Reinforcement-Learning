# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：Trainer.py
@Author  ：Neardws
@Date    ：7/1/21 10:11 上午 
"""
import copy
import random
import pickle
import os
from gym import wrappers
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

    def create_object_to_store_results(self):
        """
        Creates a dictionary that we will store the results in
        if it doesn't exist, otherwise it loads it up
        """
        if self.config.overwrite_existing_results_file \
                or not self.config.file_to_save_data_results \
                or not os.path.isfile(
                self.config.file_to_save_data_results):
            results = {}
        else:
            results = self.load_obj(self.config.file_to_save_data_results)
        return results

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
        agent_config = copy.deepcopy(self.config)

        if self.config.randomise_random_seed: agent_config.seed = random.randint(0, 2 ** 32 - 2)
        print("RANDOM SEED ", agent_config.seed)

        game_scores, rolling_scores, time_taken = self.agent.run_n_episodes()
        print("Time taken: {}".format(time_taken), flush=True)

        self.print_two_empty_lines()
        agent_result = [game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken]

        pass
