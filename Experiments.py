#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Experiments.py
@Time    :   2021/09/23 16:05:02
@Author  :   Neardws
@Version :   1.0
@Contact :   neard.ws@gmail.com
'''

import numpy as np
from File_Name import project_dir
from Utilities.FileOperator import load_obj
from Utilities.FileOperator import init_file_name
from Utilities.FileOperator import save_init_files
from Utilities.FileOperator import load_name
from Agents.HMAIMD import HMAIMD_Agent
from Agents.Random_Agent import Random_Agent
# from Agents.DDPG_Agent import DDPG_Agent
# from Agents.IDDPG import IDDPG_Agent
# from Agents.IDPG_Agent import IDPG_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
# from Agents.Trainer import Trainer
from multiprocessing import Pool

def show_old_env(environment_file_name):
    environment = load_obj(name=environment_file_name)
    # environment.get_mean_and_second_moment_service_time_of_types()
    print('bandwidth: ', environment.bandwidth)
    
    print('data_size_lower: ', environment.experiment_config.data_size_low_bound)
    print('data_size_upper: ', environment.experiment_config.data_size_up_bound)

    print('data_size_of_types: ', np.mean(environment.data_size_of_types) / 1024)
    print("noise_uncertainty_low_bound: ", environment.experiment_config.noise_uncertainty_low_bound)
    print("noise_uncertainty_up_bound: ", environment.experiment_config.noise_uncertainty_up_bound)
    sum_of_data_types_in_vehicles = 0
    for vehicle_index in range(environment.experiment_config.vehicle_number):
        data_types_in_vehicle = 0
        for data_types_index in range(environment.experiment_config.data_types_number):
            if environment.data_types_in_vehicles[vehicle_index][data_types_index] == 1:
                data_types_in_vehicle += 1
        print('data_types_in_vehicle: ', data_types_in_vehicle)
        sum_of_data_types_in_vehicles += data_types_in_vehicle
    print('sum_of_data_types_in_vehicles: ', sum_of_data_types_in_vehicles / environment.experiment_config.vehicle_number)

    sum_edge_view_required_data = 0
    for edge_view_index in range(environment.experiment_config.edge_views_number):
        edge_view_required_data = 0
        for data_types_index in range(environment.experiment_config.data_types_number):
            for vehicle_index in range(environment.experiment_config.vehicle_number):
                if environment.view_required_data[edge_view_index][vehicle_index][data_types_index] == 1:
                    edge_view_required_data += 1
        print('edge_view_required_data:', edge_view_required_data)
        sum_edge_view_required_data += edge_view_required_data
    edge_view_required_data = sum_edge_view_required_data / environment.experiment_config.edge_views_number
    print("edge_view_required_data: ", edge_view_required_data)

    sum_edge_view_in_edge_node = 0
    for time_slot_index in range(environment.experiment_config.time_slots_number):
        edge_view_in_edge_node = 0
        for edge_view_index in range(environment.experiment_config.edge_views_number):
            if environment.edge_views_in_edge_node[edge_view_index][time_slot_index] == 1:
                edge_view_in_edge_node += 1
        # print('edge_view_in_edge_node: ', edge_view_in_edge_node)
        sum_edge_view_in_edge_node += edge_view_in_edge_node
    print('view requirements: ', sum_edge_view_in_edge_node)
    print('sum_edge_view_in_edge_node: ', sum_edge_view_in_edge_node / environment.experiment_config.time_slots_number)
    
    # print("bandwidth: ", environment.bandwidth)
    print("mean_service_time_of_types: \n", np.mean(environment.experiment_config.mean_service_time_of_types))
    # print("mean_service_time_of_types: \n", environment.experiment_config.mean_service_time_of_types)
    print("second_moment_service_time_of_types: \n", np.mean(environment.experiment_config.second_moment_service_time_of_types))
    # print("second_moment_service_time_of_types: \n", environment.experiment_config.second_moment_service_time_of_types)
    # print("threshold_edge_views_in_edge_node: ", environment.experiment_config.threshold_edge_views_in_edge_node)
    print("\n")


def show_env(environment_file_name):
    environment = load_obj(name=environment_file_name)
    # environment.get_mean_and_second_moment_service_time_of_types()
    print("seed_data_types_in_vehicles: ", environment.seed_data_types_in_vehicles)
    print("seed_data_size_of_types: ", environment.seed_data_size_of_types)
    print("seed_edge_views_in_edge_node: ", environment.seed_edge_views_in_edge_node)
    print("seed_view_required_data: ", environment.seed_view_required_data)
    print('bandwidth: ', environment.bandwidth)
    
    print('data_size_lower: ', environment.experiment_config.data_size_low_bound)
    print('data_size_upper: ', environment.experiment_config.data_size_up_bound)

    print('data_size_of_types: ', np.mean(environment.data_size_of_types) / 1024)
    print("noise_uncertainty_low_bound: ", environment.experiment_config.noise_uncertainty_low_bound)
    print("noise_uncertainty_up_bound: ", environment.experiment_config.noise_uncertainty_up_bound)
    sum_of_data_types_in_vehicles = 0
    for vehicle_index in range(environment.experiment_config.vehicle_number):
        data_types_in_vehicle = 0
        for data_types_index in range(environment.experiment_config.data_types_number):
            if environment.data_types_in_vehicles[vehicle_index][data_types_index] == 1:
                data_types_in_vehicle += 1
        print('data_types_in_vehicle: ', data_types_in_vehicle)
        sum_of_data_types_in_vehicles += data_types_in_vehicle
    print('sum_of_data_types_in_vehicles: ', sum_of_data_types_in_vehicles / environment.experiment_config.vehicle_number)

    sum_edge_view_required_data = 0
    for edge_view_index in range(environment.experiment_config.edge_views_number):
        edge_view_required_data = 0
        for data_types_index in range(environment.experiment_config.data_types_number):
            if environment.view_required_data[edge_view_index][data_types_index] == 1:
                edge_view_required_data += 1
        print('edge_view_required_data:', edge_view_required_data)
        sum_edge_view_required_data += edge_view_required_data
    edge_view_required_data = sum_edge_view_required_data / environment.experiment_config.edge_views_number
    print("edge_view_required_data: ", edge_view_required_data)


    sum_edge_view_in_edge_node = 0
    for time_slot_index in range(environment.experiment_config.time_slots_number):
        edge_view_in_edge_node = 0
        for edge_view_index in range(environment.experiment_config.edge_views_number):
            if environment.edge_views_in_edge_node[edge_view_index][time_slot_index] == 1:
                edge_view_in_edge_node += 1
        # print('edge_view_in_edge_node: ', edge_view_in_edge_node)
        sum_edge_view_in_edge_node += edge_view_in_edge_node
    print('view requirements: ', sum_edge_view_in_edge_node)
    print('sum_edge_view_in_edge_node: ', sum_edge_view_in_edge_node / environment.experiment_config.time_slots_number)
    
    # print("bandwidth: ", environment.bandwidth)
    print("mean_service_time_of_types: \n", np.mean(environment.experiment_config.mean_service_time_of_types))
    # print("mean_service_time_of_types: \n", environment.experiment_config.mean_service_time_of_types)
    print("second_moment_service_time_of_types: \n", np.mean(environment.experiment_config.second_moment_service_time_of_types))
    # print("second_moment_service_time_of_types: \n", environment.experiment_config.second_moment_service_time_of_types)
    # print("threshold_edge_views_in_edge_node: ", environment.experiment_config.threshold_edge_views_in_edge_node)
    print("\n")


def show_all_environments():
    print("-" * 64)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    print("-" * 64)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_1_threshold_15.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_2_threshold_15.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_15.pkl"
    show_env(environment_file_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_4_threshold_15.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_5_threshold_15.pkl"
    show_env(environment_file_name)

    print("-" * 64)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_05.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_10.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_15.pkl"
    show_env(environment_file_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_20.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_25.pkl"
    show_env(environment_file_name)


def show_all_old_environments():

    print("-" * 64)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    print("-" * 64)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_256_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_512_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_2048_01.pkl"
    show_old_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_4096_01.pkl"
    show_old_env(environment_file_name)


if __name__ == '__main__':

    # run_ra_algorithms_for_results(
    #     num_episodes=2000, 
    #     environment_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl",
    #     ra_result_name="/home/neardws/Hierarchical-Reinforcement-Learning/Results/ra_results_1116_0800_bandwidth_3_datasize_1024_02.csv"
    # )
    # get_hmaimd_results()
    # get_hmaimd_results_changing_theshold()
    # get_nomal_iddpg_results_1()
    # get_nomal_iddpg_results_2()
    # get_nomal_iddpg_results_3()
    # get_rr_iddpg_results_1()
    # get_rr_iddpg_results_2()
    # get_rr_iddpg_results_3()

    # get_ddpg_results()
    # get_iddpg_results()
    # run(first=True)
    # run_ra_algorithms_for_results()
    # run(given_list_file_name='2021-09-22-00-47-47-list_file_name.pkl')

    # run_ddpg_algorithms(given_list_file_name='2021-10-19-16-54-31-list_file_name.pkl')

    # data = '/Data/Data1110_Agents/1116/0800/ddpg/bandwidth_3_threshold_05_01/'
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl"
    # data = '/Data/Data1110_Agents/1116/0800/bandwidth_4_threshold_05_01/'
    # ddpg_result_name = project_dir + data + "ddpg_result_10.csv"
    # ddpg_agent_name = project_dir + data + "ddpg_10.pkl"
    # num_episodes = 2500
    # run_ddpg_algorithms(environment_file_name, ddpg_result_name, ddpg_agent_name, num_episodes)
    
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl"
    # data = '/Data/Data1110_Agents/1116/0800/bandwidth_5_threshold_05_01/'
    # ddpg_result_name = project_dir + data + "ddpg_result_10.csv"
    # ddpg_agent_name = project_dir + data + "ddpg_10.pkl"
    # num_episodes = 2500
    # run_ddpg_algorithms(environment_file_name, ddpg_result_name, ddpg_agent_name, num_episodes)

    # num_episodes = 10
    # get_hmaimd_results()
    # gerenate_random_results()
    # get_hmaimd_results_changing_theshold()
    # get_iddpg_results_changing_theshold()
    
[236.87664323697, 318.508683985828, 315.034869116743, 354.157492999339, 373.360464430689]
    # get_hmaimd_results_changing_scenarios()

    # gerenate_random_results()
    # gerenate_random_results_changing_bandwidth()
    # show_all_environments()
    show_all_old_environments()
    # p = Pool(processes=5)
    # test_name = "0104_5000" + "_"

    # list_environment_file_name = []
    # list_result_name = []

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_datasize_1024_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(100, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_datasize_1024_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_256_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(2000, environment_file_name, ra_result_name)
    
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_512_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(5000, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_2048_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_4096_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_datasize_1024_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_datasize_1024_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # list_num_episodes = [5000 for _ in range(len(list_environment_file_name))]

    # for i in range(len(list_num_episodes)):
    #     p.apply_async(run_ra_algorithms_for_results, args=(list_num_episodes[i], list_environment_file_name[i], list_result_name[i],))

    # p.close()
    # p.join()

