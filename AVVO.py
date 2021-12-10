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
from Agents.IDDPG import IDDPG_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig

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
            "learning_rate": 1e-5,
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
            "learning_rate": 1e-4,
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
        "actor_nodes_learning_updates_per_learning_session": 40,
        "critic_nodes_learning_updates_per_learning_session": 40,
        "actor_reward_learning_updates_per_learning_session": 160,
        "critic_reward_learning_updates_per_learning_session": 160,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)
    return vehicularNetworkEnv.experiment_config, agent_config, vehicularNetworkEnv


def run_iddpg(first=False, rerun=False, environments_file_name=None, given_list_file_name=None):
    if first:  # run in the first time
        experiment_config, agent_config, vehicularNetworkEnv = init(environments_file_name)
        list_file_name = init_file_name()
        save_init_files(list_file_name, experiment_config, agent_config, vehicularNetworkEnv)
        agent = IDDPG_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)
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
            agent = HMAIMD_Agent(agent_config=agent_config, environment=init_vehicularNetworkEnv)
            agent.run_n_episodes(temple_agent_config_name=load_name(new_list_file_name, 'temple_agent_config_name'),
                                temple_agent_name=load_name(new_list_file_name, 'temple_agent_name'),
                                temple_result_name=load_name(new_list_file_name, 'temple_result_name'),
                                temple_loss_name=load_name(new_list_file_name, 'temple_loss_name'),
                                actor_nodes_name=load_name(new_list_file_name, 'actor_nodes_name'), 
                                actor_edge_name=load_name(new_list_file_name, 'actor_edge_name'))
        else:
            list_file = load_obj(name=given_list_file_name)
            temple_agent_config = load_obj(name=load_name(list_file, 'temple_agent_config_name'))
            temple_agent = load_obj(name=load_name(list_file, 'temple_agent_name'))
            temple_agent.run_n_episodes(num_episodes=5000,
                                        temple_agent_config_name=load_name(list_file, 'temple_agent_config_name'),
                                        temple_agent_name=load_name(list_file, 'temple_agent_name'),
                                        temple_result_name=load_name(list_file, 'temple_result_name'),
                                        temple_loss_name=load_name(list_file, 'temple_loss_name'),
                                        actor_nodes_name=load_name(list_file, 'actor_nodes_name'), 
                                        actor_edge_name=load_name(list_file, 'actor_edge_name'))


def run(first=False, rerun=False, environments_file_name=None, given_list_file_name=None):
    if first:  # run in the first time
        experiment_config, agent_config, vehicularNetworkEnv = init(environments_file_name)

        # correct_list_file_name = project_dir + data + '2021-09-01-03-58-12-list_file_name.pkl'
        # list_file = load_obj(name=correct_list_file_name)
        # vehicularNetworkEnv = load_obj(name=load_name(list_file, 'init_environment_name'))

        list_file_name = init_file_name()
        save_init_files(list_file_name, experiment_config, agent_config, vehicularNetworkEnv)
        agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

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
            agent = HMAIMD_Agent(agent_config=agent_config, environment=init_vehicularNetworkEnv)
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
            temple_agent.run_n_episodes(num_episodes=2000,
                                        temple_agent_config_name=load_name(list_file, 'temple_agent_config_name'),
                                        temple_agent_name=load_name(list_file, 'temple_agent_name'),
                                        temple_result_name=load_name(list_file, 'temple_result_name'),
                                        temple_loss_name=load_name(list_file, 'temple_loss_name'),
                                        actor_nodes_name=load_name(list_file, 'actor_nodes_name'), 
                                        actor_edge_name=load_name(list_file, 'actor_edge_name'))

def run_again(temple_agent_config_file_name, temple_agent_file_name):
    new_list_file_name = init_file_name()
    temple_agent_config = load_obj(temple_agent_config_file_name)
    hyperparameters = {
        "Actor_of_Sensor": {
            "learning_rate": 1e-2,
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
            "learning_rate": 1e-1,
            "linear_hidden_units": [128, 64],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 1e-2,
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
            "learning_rate": 1e-1,
            "linear_hidden_units": [256, 128], 
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 1e-4,
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
            "learning_rate": 1e-3,
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
        "actor_nodes_learning_updates_per_learning_session": 1,
        "critic_nodes_learning_updates_per_learning_session": 1,
        "actor_reward_learning_updates_per_learning_session": 160,
        "critic_reward_learning_updates_per_learning_session": 160,
        "clip_rewards": False}

    temple_agent_config.config(hyperparameters=hyperparameters)

    temple_agent = load_obj(temple_agent_file_name)
    temple_agent.config_hyperparameters(hyperparameters=hyperparameters)

    trainer = Trainer(temple_agent_config, temple_agent)
    trainer.run_games_for_agent(temple_agent_config_name=load_name(new_list_file_name, 'temple_agent_config_name'),
                                temple_agent_name=load_name(new_list_file_name, 'temple_agent_name'),
                                temple_result_name=load_name(new_list_file_name, 'temple_result_name'),
                                temple_loss_name=load_name(new_list_file_name, 'temple_loss_name'),
                                agent_name=load_name(new_list_file_name, 'agent_name'))

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
    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=1)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_05_01.pkl")
    
    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=2)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=4)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl")

    # environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    # environment.config_bandwidth(new_bandwidth=5)
    # save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl")

    environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    environment.config_threshold_view_required_data(new_threshold_view_required_data=0.175)
    save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl")

    environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    environment.config_threshold_view_required_data(new_threshold_view_required_data=0.33)
    save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl")

    environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    environment.config_threshold_view_required_data(new_threshold_view_required_data=0.54)
    save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl")

    environment = load_obj(name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl")
    environment.config_threshold_view_required_data(new_threshold_view_required_data=0.64)
    save_obj(obj=environment, name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl")
    
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
    
    # run_iddpg(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl")
    
    # run(first=True, environments_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_datasize_1024_01.pkl")
    run(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1210_Agents/bandwidth_3_datasize_1024_01/2021-12-09-16-38-26-list_file_name.pkl")

    # run(rerun=True, given_list_file_name='2021-10-25-22-33-35-list_file_name.pkl')

    # run_iddpg(given_list_file_name="2021-11-15-16-08-37-list_file_name.pkl")

    # nomal bandwidth = 1
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_1_threshold_05_01/2021-11-15-22-24-33-list_file_name.pkl")
    # nomal bandwidth = 2
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_2_threshold_05_01/2021-11-15-22-31-59-list_file_name.pkl")
    # nomal bandwidth = 4
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_4_threshold_05_01/2021-11-15-22-33-35-list_file_name.pkl")
    # nomal bandwidth = 5
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_5_threshold_05_01/2021-11-15-22-35-59-list_file_name.pkl")

    # nomal threshold = 0.3
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_03_01/2021-11-15-22-38-16-list_file_name.pkl")
    # nomal threshold = 0.4
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_04_01/2021-11-15-22-38-45-list_file_name.pkl")
    # nomal threshold = 0.6
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_06_01/2021-11-15-22-39-05-list_file_name.pkl")
    # nomal threshold = 0.7
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_07_01/2021-11-15-22-39-27-list_file_name.pkl")

    # random bandwidth = 1
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_1_threshold_05_01/2021-11-15-22-47-48-list_file_name.pkl")
    # random bandwidth = 2
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_2_threshold_05_01/2021-11-15-22-48-30-list_file_name.pkl")
    # random bandwidth = 4
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_4_threshold_05_01/2021-11-15-22-49-02-list_file_name.pkl")
    # random bandwidth = 5
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_5_threshold_05_01/2021-11-15-22-49-31-list_file_name.pkl")

    # random threshold = 0.3
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_03_01/2021-11-15-22-52-48-list_file_name.pkl")
    # random threshold = 0.4
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_04_01/2021-11-15-22-53-17-list_file_name.pkl")
    # random threshold = 0.6
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_06_01/2021-11-15-22-53-41-list_file_name.pkl")
    # random threshold = 0.7
    # run_iddpg(given_list_file_name="/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_07_01/2021-11-15-22-54-05-list_file_name.pkl")


    
    # run(given_list_file_name='2021-09-29-16-16-31-list_file_name.pkl')
    # run(given_list_file_name='2021-09-29-20-11-11-list_file_name.pkl')

    # run(given_list_file_name='2021-10-01-14-54-33-list_file_name.pkl')
    # run(given_list_file_name='2021-10-02-11-34-43-list_file_name.pkl')
    # run(given_list_file_name='2021-10-08-12-53-05-list_file_name.pkl')
    
    # run(given_list_file_name='2021-10-19-16-54-31-list_file_name.pkl')
    
    # IDDPG
    # run(given_list_file_name='2021-10-10-15-40-29-list_file_name.pkl')

    # run_again(
    #     temple_agent_config_file_name='/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1021/2021-10-21-15-05-30/temple_agent_config_d8b2a3ad18cf457c854f792ffd109a87.pkl',
    #     temple_agent_file_name='/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1021/2021-10-21-15-05-30/temple_agent_7ff8d462fac44731ae264ed69a1421df.pkl'
    #     )

    