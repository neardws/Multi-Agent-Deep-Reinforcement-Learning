# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""
import json
import numpy as np
from File_Name import project_dir, data
from Utilities.FileOperator import load_obj, save_obj
from Utilities.FileOperator import init_file_name
from Utilities.FileOperator import save_init_files
from Utilities.FileOperator import load_name
from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig

# import sys
# print(sys.path)

def show_environment(environments_file_name):
    vehicularNetworkEnv = load_obj(name=environments_file_name)
    print(vehicularNetworkEnv.__dict__)

def show_environment_config(file_name):
    vehicularNetworkEnv_config = load_obj(name=file_name)
    print(vehicularNetworkEnv_config.__dict__)

def save_environment(trajectories_file_name, environments_file_name):
    experiment_config = ExperimentConfig()
    experiment_config.config()
    env = VehicularNetworkEnv(experiment_config, trajectories_file_name)
    env.reset()
    save_obj(env, environments_file_name)


def generate_environment():
    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/scenario/vehicle_1116_23.csv"
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1116_2350_bandwidth_3_dataszie_1024_02.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

    trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/scenario/vehicle_1127_12.csv"
    environments_file_name = project_dir + "/Environments/Data/vehicle_1127_1200_bandwidth_3_dataszie_1024_02.pkl"
    save_environment(trajectories_file_name, environments_file_name)

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_08_scen_03.csv"
    # environments_file_name = project_dir + "/Environments/Data/Scenario3/vehicle_1116_0800_bandwidth_3_threshold_15.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_08_scen_04.csv"
    # environments_file_name = project_dir + "/Environments/Data/Scenario4/vehicle_1116_0800_bandwidth_3_threshold_15.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

def change_environment(environment_file_name):
    obj_file_name = environment_file_name.replace('vehicle_1116_0800_bandwidth_3_threshold_15.pkl', '')
    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_bandwidth(new_bandwidth=1)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_1_threshold_15.pkl")
    
    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_bandwidth(new_bandwidth=2)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_2_threshold_15.pkl")

    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_bandwidth(new_bandwidth=4)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_4_threshold_15.pkl")

    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_bandwidth(new_bandwidth=5)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_5_threshold_15.pkl")

    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_data_types_in_vehicles(new_threshold_data_types_in_vehicles=0.09)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_3_threshold_05.pkl")
    
    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_data_types_in_vehicles(new_threshold_data_types_in_vehicles=0.13)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_3_threshold_10.pkl")

    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_data_types_in_vehicles(new_threshold_data_types_in_vehicles=0.21)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_3_threshold_20.pkl")

    environment = load_obj(name=environment_file_name)
    print(environment.seed_data_types_in_vehicles)
    environment.config_data_types_in_vehicles(new_threshold_data_types_in_vehicles=0.25)
    save_obj(obj=environment, name=obj_file_name+"vehicle_1116_0800_bandwidth_3_threshold_25.pkl")
    
    



if __name__ == '__main__':

    # show_environment_config("/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl")
    # show_environment("/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_2350_bandwidth_3_dataszie_1024_01.pkl")
    # show_environment("/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1127_0800_bandwidth_3_dataszie_1024_01.pkl")
    generate_environment()
    # change_environment(environment_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_15.pkl")



    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_08_scen_01.csv"
    # experiment_config = ExperimentConfig()
    # experiment_config.config()
    # env = VehicularNetworkEnv(experiment_config, trajectories_file_name)
    # print(env.compute_SNR_by_distance(distance=460))
    # for _ in range(10):
    #     print(VehicularNetworkEnv.computer_SNR_wall_by_noise_uncertainty(
    #                         noise_uncertainty=np.random.uniform(low=0,
    #                                                             high=3)))