# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""
import json

import numpy as np

from Agents.DR_GA import DR_GA_Agent
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from File_Name import data, project_dir
from Utilities.FileOperator import (init_file_name, load_name, load_obj,
                                    save_init_files, save_obj)


def show_environment(environments_file_name):
    vehicularNetworkEnv = load_obj(name=environments_file_name)
    print(vehicularNetworkEnv.__dict__)

def show_environment_config(file_name):
    vehicularNetworkEnv_config = load_obj(name=file_name)
    print(vehicularNetworkEnv_config.__dict__)

def show_agent_config(file_name):
    agent_config = load_obj(name=file_name)
    print(agent_config.__dict__)

def save_environment(trajectories_file_name, environments_file_name):
    experiment_config = ExperimentConfig()
    experiment_config.config()
    env = VehicularNetworkEnv(experiment_config, trajectories_file_name)
    env.reset()
    save_obj(env, environments_file_name)

def init(environments_file_name):
    # experiment_config = ExperimentConfig()
    # experiment_config.config()

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_08.csv"
    vehicularNetworkEnv = load_obj(name=environments_file_name)
    
    vehicularNetworkEnv.reset()

    agent_config = AgentConfig()

    hyperparameters = {
        # Builds a NN with 2 output heads. The first output heads has data_types_number hidden units and
        # uses a softmax activation function and the second output head has data_types_number hidden units and
        # uses a softmax activation function

        "Actor_of_Sensor": {
            "learning_rate": 1e-3,
            "linear_hidden_units": [64, 32],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25,
            "action_noise_std": 0.001,
            "action_noise_clipping_range": 1.0
        },

        "Critic_of_Sensor": {
            "learning_rate": 1e-2,
            "linear_hidden_units": [128, 64],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 1e-5,
            "linear_hidden_units": [256, 128],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25,
            "action_noise_std": 0.001,
            "action_noise_clipping_range": 1.0
        },

        "Critic_of_Edge": {
            "learning_rate": 1e-4,
            "linear_hidden_units": [256, 128], 
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 1e-5,
            "linear_hidden_units": [256, 128],
            "final_layer_activation": "softmax",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

        "Critic_of_Reward": {
            "learning_rate": 1e-4,
            "linear_hidden_units": [256, 128],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "discount_rate": 0.996,
        "actor_nodes_update_every_n_steps": 300,  # 10 times in one episode
        "critic_nodes_update_every_n_steps": 300,  # 15 times in one episode
        "actor_reward_update_every_n_steps": 300,  # 20 times in one episode
        "critic_reward_update_every_n_steps": 300,  # 20 times in one episode
        "actor_nodes_learning_updates_per_learning_session": 16,
        "critic_nodes_learning_updates_per_learning_session": 16,
        "actor_reward_learning_updates_per_learning_session": 160,
        "critic_reward_learning_updates_per_learning_session": 160,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)
    return vehicularNetworkEnv.experiment_config, agent_config, vehicularNetworkEnv



def run(first=False, rerun=False, environments_file_name=None, given_list_file_name=None):
    if first:  # run in the first time
        experiment_config, agent_config, vehicularNetworkEnv = init(environments_file_name)

        # correct_list_file_name = project_dir + data + '2021-09-01-03-58-12-list_file_name.pkl'
        # list_file = load_obj(name=correct_list_file_name)
        # vehicularNetworkEnv = load_obj(name=load_name(list_file, 'init_environment_name'))

        list_file_name = init_file_name()
        save_init_files(list_file_name, experiment_config, agent_config, vehicularNetworkEnv)
        agent = DR_GA_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

        agent.run_n_episodes(temple_agent_config_name=load_name(list_file_name, 'temple_agent_config_name'),
                            temple_agent_name=load_name(list_file_name, 'temple_agent_name'),
                            temple_result_name=load_name(list_file_name, 'temple_result_name'),
                            temple_loss_name=load_name(list_file_name, 'temple_loss_name'),
                            actor_nodes_name=load_name(list_file_name, 'actor_nodes_name'), 
                            actor_edge_name=load_name(list_file_name, 'actor_edge_name'))

    else:
        if rerun:
            correct_list_file_name = project_dir + data + given_list_file_name
            list_file = load_obj(name=correct_list_file_name)
            # init_experiment_config = load_obj(load_name(list_file, 'init_experiment_config_name'))
            # init_agent_config = load_obj(load_name(list_file, 'init_agent_config_name'))
            init_vehicularNetworkEnv = load_obj(load_name(list_file, 'init_environment_name'))
            experiment_config, agent_config, _ = init(load_name(list_file, 'init_environment_name'))
            new_list_file_name = init_file_name()
            save_init_files(new_list_file_name, experiment_config, agent_config, init_vehicularNetworkEnv)
            agent = DR_GA_Agent(agent_config=agent_config, environment=init_vehicularNetworkEnv)
            agent.run_n_episodes(temple_agent_config_name=load_name(new_list_file_name, 'temple_agent_config_name'),
                                temple_agent_name=load_name(new_list_file_name, 'temple_agent_name'),
                                temple_result_name=load_name(new_list_file_name, 'temple_result_name'),
                                temple_loss_name=load_name(new_list_file_name, 'temple_loss_name'),
                                actor_nodes_name=load_name(new_list_file_name, 'actor_nodes_name'), 
                                actor_edge_name=load_name(new_list_file_name, 'actor_edge_name'))
        else:
            correct_list_file_name = given_list_file_name
            list_file = load_obj(name=correct_list_file_name)
            temple_agent_config = load_obj(name=load_name(list_file, 'temple_agent_config_name'))
            temple_agent = load_obj(name=load_name(list_file, 'temple_agent_name'))
            temple_agent.run_n_episodes(num_episodes=5000,
                                        temple_agent_config_name=load_name(list_file, 'temple_agent_config_name'),
                                        temple_agent_name=load_name(list_file, 'temple_agent_name'),
                                        temple_result_name=load_name(list_file, 'temple_result_name'),
                                        temple_loss_name=load_name(list_file, 'temple_loss_name'),
                                        actor_nodes_name=load_name(list_file, 'actor_nodes_name'), 
                                        actor_edge_name=load_name(list_file, 'actor_edge_name'))


def generate_environment():
    trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_10.csv"
    environments_file_name = project_dir + "/Environments/Data/vehicle_1116_1000_bandwidth_3_threshold_05_01.pkl"
    save_environment(trajectories_file_name, environments_file_name)

    trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_12.csv"
    environments_file_name = project_dir + "/Environments/Data/vehicle_1116_1200_bandwidth_3_threshold_05_01.pkl"
    save_environment(trajectories_file_name, environments_file_name)

    trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_18.csv"
    environments_file_name = project_dir + "/Environments/Data/vehicle_1116_1800_bandwidth_3_threshold_05_01.pkl"
    save_environment(trajectories_file_name, environments_file_name)

    trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_22.csv"
    environments_file_name = project_dir + "/Environments/Data/vehicle_1116_2200_bandwidth_3_threshold_05_01.pkl"
    save_environment(trajectories_file_name, environments_file_name)

    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_0800_bandwidth_3_threshold_015_02.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1117_10.csv"
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_1000_bandwidth_3_threshold_015_01.pkl"
    # save_environment(trajectories_file_name, environments_file_name)
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_1000_bandwidth_3_threshold_015_02.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1117_12.csv"
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_1200_bandwidth_3_threshold_015_01.pkl"
    # save_environment(trajectories_file_name, environments_file_name)
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_1200_bandwidth_3_threshold_015_02.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1117_18.csv"
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_1800_bandwidth_3_threshold_015_01.pkl"
    # save_environment(trajectories_file_name, environments_file_name)
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_1800_bandwidth_3_threshold_015_02.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

    # trajectories_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1117_22.csv"
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_2200_bandwidth_3_threshold_015_01.pkl"
    # save_environment(trajectories_file_name, environments_file_name)
    # environments_file_name = project_dir + "/Environments/Data/vehicle_1117_2200_bandwidth_3_threshold_015_02.pkl"
    # save_environment(trajectories_file_name, environments_file_name)

def change_environment():
    environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl")
    environment.config_datasize_of_types(new_data_size_of_types=256 * 1024)
    save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_256_01.pkl")
    
    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=2)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=4)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=5)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_threshold_view_required_data(new_threshold_view_required_data=0.175)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_threshold_view_required_data(new_threshold_view_required_data=0.33)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_threshold_view_required_data(new_threshold_view_required_data=0.54)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_threshold_view_required_data(new_threshold_view_required_data=0.64)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl")
    
    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_01.pkl")
    # environment.config_bandwidth(new_bandwidth=5)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_01.pkl")
    # environment.config_bandwidth(new_bandwidth=5.5)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_5_threshold_015_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_01.pkl")
    # environment.config_bandwidth(new_bandwidth=6)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_6_threshold_015_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl")
    # environment.config_views_required_at_each_time_slot(threshold_edge_views_in_edge_node=0.05)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_005_02.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl")
    # environment.config_views_required_at_each_time_slot(threshold_edge_views_in_edge_node=0.10)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_010_02.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl")
    # environment.config_views_required_at_each_time_slot(threshold_edge_views_in_edge_node=0.20)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_020_02.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl")
    # environment.config_views_required_at_each_time_slot(threshold_edge_views_in_edge_node=0.25)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_025_02.pkl")

if __name__ == '__main__':

    # show_environment_config("/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1209_Agents/bandwidth_3_datasize_1024_01/2021-12-07-19-11-36/init_experiment_config_8f3e0dd35b3f41e2bbc0e06896ada216.pkl")
    # show_environment("/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1209_Agents/bandwidth_3_datasize_1024_01/2021-12-07-19-11-36/init_environment_8f3e0dd35b3f41e2bbc0e06896ada216.pkl")
    # generate_environment()
    # change_environment()
    # show_agent_config("/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1213_Agents/bandwidth_3_datasize_1024_01/2021-12-13-15-45-10/init_agent_config_777591997d204dc2bc98f329347a7263.pkl")
    # run_iddpg(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl")
    # show_agent_config("/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1213_Agents/bandwidth_3_datasize_1024_01/2021-12-13-09-44-46/init_agent_config_6304b8bfb4bf4295a4f289cfefa89e3c.pkl")
    
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/Scenario1/vehicle_1116_0800_bandwidth_3_threshold_15.pkl")
    
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1127_0800_bandwidth_3_dataszie_1024_02.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_2350_bandwidth_3_dataszie_1024_02.pkl")

    # DR_GA
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0205_MDR_GBA/bandwidth_3_threshold_15/2022-05-05-21-26-26-list_file_name.pkl")

    run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0205_MDR_GBA/bandwidth_3_threshold_15/2022-05-05-21-28-32-list_file_name.pkl")
    
    
    
    
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0205_MDR_GBA/bandwidth_3_threshold_15/2022-02-05-11-24-21-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0205_MDR_GBA/bandwidth_3_threshold_15/2022-02-05-15-53-16-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0205_MDR_GBA/bandwidth_3_threshold_15/2022-02-05-16-16-57-list_file_name.pkl")
    
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_2_datasize_1024_01/2021-12-16-19-35-30-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_4_datasize_1024_01/2021-12-16-19-35-58-list_file_name.pkl")
    
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_5_datasize_1024_01/2021-12-16-19-36-41-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_3_datasize_256_01/2021-12-16-19-34-19-list_file_name.pkl")
    
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_3_datasize_512_01/2021-12-16-19-33-44-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_3_datasize_2048_01/2021-12-16-19-31-15-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1216_Agents/bandwidth_3_datasize_4096_01/2021-12-16-19-32-59-list_file_name.pkl")

    
    