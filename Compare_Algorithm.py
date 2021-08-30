# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：Compare_Algorithm.py
@Author  ：Neardws
@Date    ：8/27/21 9:26 下午 
"""
from file_name import project_dir
from Utilities.FileOperator import load_obj


def run(rerun=False, given_list_file_name=None):
    correct_list_file_name = project_dir + '/Data/' + given_list_file_name
    list_file = load_obj(name=correct_list_file_name)

    if rerun:
        vehicularNetworkEnv = load_obj(list_file[3])
        trainer = Trainer(agent_config, agent)
        trainer.run_games_for_agent(temple_agent_config_name=list_file[4],
                                    temple_agent_name=list_file[5],
                                    temple_result_name=list_file[6])


if __name__ == '__main__':
    # run(first=True)
    # run(rerun=True, given_list_file_name='2021-08-10-10-19-22-list_file_name.pkl')
    run(given_list_file_name='2021-08-21-05-56-04-list_file_name.pkl')
