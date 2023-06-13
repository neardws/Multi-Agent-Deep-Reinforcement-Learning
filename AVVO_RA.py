#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Experiments.py
@Time    :   2021/09/23 16:05:02
@Author  :   Neardws
@Version :   1.0
@Contact :   neard.ws@gmail.com
'''

# from Agents.Trainer import Trainer
from multiprocessing import Pool

import numpy as np

from Agents.HMAIMD import HMAIMD_Agent
from Agents.Random_Agent import Random_Agent
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from File_Name import project_dir
from Utilities.FileOperator import (init_file_name, load_name, load_obj,
                                    save_init_files)


def run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name):
    environment = load_obj(name=environment_file_name)
    ra_agent = Random_Agent(environment=environment)
    ra_agent.run_n_episodes_as_results(num_episodes=num_episodes, result_name=ra_result_name)

if __name__ == '__main__':

    # run_ra_algorithms_for_results(
    #     num_episodes=2000, 
    #     environment_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl",
    #     ra_result_name="/home/neardws/Hierarchical-Reinforcement-Learning/Results/ra_results_1116_0800_bandwidth_3_datasize_1024_02.csv"
    # )
    # test_name = "0204_1039" + "_"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_05.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # # list_environment_file_name.append(environment_file_name)
    # # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    p = Pool(processes=2)
    test_name = "0505_1517" + "_"

    list_environment_file_name = []
    list_result_name = []

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_2350_bandwidth_3_dataszie_1024_02.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1127_0800_bandwidth_3_dataszie_1024_02.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_05.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_10.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_15.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_20.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_25.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)
    
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_4_threshold_15.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_5_threshold_15.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    list_num_episodes = [100 for _ in range(len(list_environment_file_name))]

    for i in range(len(list_num_episodes)):
        p.apply_async(run_ra_algorithms_for_results, args=(list_num_episodes[i], list_environment_file_name[i], list_result_name[i],))

    p.close()
    p.join()

