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
from Agents.DDPG import DDPG_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig


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

        "Actor_of_DDPG": {
            "learning_rate": 1e-6,
            "linear_hidden_units": [512, 256],
            "final_layer_activation": [
                    "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax",
                    "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax", "softmax",
                    "softmax"
                ],
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

        "Critic_of_DDPG": {
            "learning_rate": 1e-5,
            "linear_hidden_units": [512, 256],
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
        "actor_nodes_learning_updates_per_learning_session": 1,
        "critic_nodes_learning_updates_per_learning_session": 1,
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
        agent = DDPG_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

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
            agent = DDPG_Agent(agent_config=agent_config, environment=init_vehicularNetworkEnv)
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
            temple_agent.run_n_episodes(num_episodes=10000,
                                        temple_agent_config_name=load_name(list_file, 'temple_agent_config_name'),
                                        temple_agent_name=load_name(list_file, 'temple_agent_name'),
                                        temple_result_name=load_name(list_file, 'temple_result_name'),
                                        temple_loss_name=load_name(list_file, 'temple_loss_name'),
                                        actor_nodes_name=load_name(list_file, 'actor_nodes_name'), 
                                        actor_edge_name=load_name(list_file, 'actor_edge_name'))



if __name__ == '__main__':
    
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_datasize_1024_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_datasize_1024_01.pkl")
    run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_datasize_1024_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_datasize_1024_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_256_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_512_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_2048_01.pkl")
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_4096_01.pkl")


    # C-DDPG

    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_1_datasize_1024/2022-01-09-18-51-08-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_2_datasize_1024/2022-01-09-18-52-50-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_3_datasize_256/2022-01-09-18-57-06-list_file_name.pkl")
    
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_3_datasize_512/2022-01-09-18-57-44-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_3_datasize_1024/2022-01-09-18-53-21-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_3_datasize_2048/2022-01-09-18-58-30-list_file_name.pkl")
    
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_3_datasize_4096/2022-01-09-18-59-01-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_4_datasize_1024/2022-01-09-18-54-03-list_file_name.pkl")
    # run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data0109_DDPG/bandwidth_5_datasize_1024/2022-01-09-18-55-31-list_file_name.pkl")

