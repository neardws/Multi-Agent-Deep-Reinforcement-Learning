# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：Trainer.py
@Author  ：Neardws
@Date    ：7/1/21 10:11 上午 
"""
from Config.AgentConfig import AgentConfig
from Agents.HMAIMD import HMAIMD_Agent
from Utilities.FileOperator import save_obj


class Trainer(object):
    """
    Runs game for given agent
    """
    def __init__(self,
                 agent_config: AgentConfig,
                 agent: HMAIMD_Agent):
        self.config = agent_config
        self.agent = agent

    @staticmethod
    def print_two_empty_lines():
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

    def run_games_for_agent(self, temple_agent_config_name, temple_agent_name, temple_result_name, temple_loss_name):

        game_scores, rolling_scores, time_taken = self.agent.run_n_episodes(temple_agent_config_name=temple_agent_config_name,
                                                                            temple_agent_name=temple_agent_name,
                                                                            temple_result_name=temple_result_name,
                                                                            temple_loss_name=temple_loss_name)
        print("Time taken: {}".format(time_taken), flush=True)

        self.print_two_empty_lines()
        agent_result = [game_scores, rolling_scores, len(rolling_scores), max(rolling_scores), time_taken]
        save_obj(agent_result, self.config.file_to_save_data_results)
