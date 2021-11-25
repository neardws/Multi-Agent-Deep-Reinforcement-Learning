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
from Agents.DDPG_Agent import DDPG_Agent
from Agents.IDDPG import IDDPG_Agent
# from Agents.IDPG_Agent import IDPG_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Agents.Trainer import Trainer
from multiprocessing import Pool


def init():
    experiment_config = ExperimentConfig()
    experiment_config.config()

    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)
    vehicularNetworkEnv.reset()

    agent_config = AgentConfig()

    hyperparameters = {
        # Builds a NN with 2 output heads. The first output heads has data_types_number hidden units and
        # uses a softmax activation function and the second output head has data_types_number hidden units and
        # uses a softmax activation function

        "Actor_of_Sensor": {
            "learning_rate": 1e-7,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size())),
                 int(0.75 * (
                         vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size()))
                 ],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.00001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25,
            "action_noise_std": 0.001,
            "action_noise_clipping_range": 1.0
        },

        "Critic_of_Sensor": {
            "learning_rate": 1e-7,
            "linear_hidden_units":
                [int(0.2 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1)),
                 int(0.2 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1))],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.00001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 1e-5,
            "linear_hidden_units":
                [int(0.3 * (
                        vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size())),
                 int(0.2 * (
                         vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size()))
                 ],
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
            "learning_rate": 1e-5,
            "linear_hidden_units":
                [int(0.3 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1)),
                 int(0.2 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1))],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 1e-6,
            "linear_hidden_units":
                [int(0.5 * (
                        vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size())),
                 int(0.5 * (
                         vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size()))],
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
            "learning_rate": 1e-6,
            "linear_hidden_units":
                [int(0.4 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1)),
                 int(0.4 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1))],
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
        "actor_nodes_learning_updates_per_learning_session": 8,
        "critic_nodes_learning_updates_per_learning_session": 8,
        "actor_reward_learning_updates_per_learning_session": 8,
        "critic_reward_learning_updates_per_learning_session": 8,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)
    return experiment_config, agent_config, vehicularNetworkEnv


def run(first=False, rerun=False, given_list_file_name=None):
    if first:  # run in the first time
        experiment_config, agent_config, vehicularNetworkEnv = init()

        # correct_list_file_name = project_dir + data + '2021-09-01-03-58-12-list_file_name.pkl'
        # list_file = load_obj(name=correct_list_file_name)
        # vehicularNetworkEnv = load_obj(name=load_name(list_file, 'init_environment_name'))

        list_file_name = init_file_name()
        save_init_files(list_file_name, experiment_config, agent_config, vehicularNetworkEnv)
        agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

        trainer = Trainer(agent_config, agent)
        trainer.run_games_for_agent(temple_agent_config_name=load_name(list_file_name, 'temple_agent_config_name'),
                                    temple_agent_name=load_name(list_file_name, 'temple_agent_name'),
                                    temple_result_name=load_name(list_file_name, 'temple_result_name'),
                                    temple_loss_name=load_name(list_file_name, 'temple_loss_name'))
    else:
        if rerun:
            correct_list_file_name = project_dir + data + given_list_file_name
            list_file = load_obj(name=correct_list_file_name)
            # init_experiment_config = load_obj(load_name(list_file, 'init_experiment_config_name'))
            # init_agent_config = load_obj(load_name(list_file, 'init_agent_config_name'))
            init_vehicularNetworkEnv = load_obj(load_name(list_file, 'init_environment_name'))
            experiment_config, agent_config, _ = init()
            new_list_file_name = init_file_name()
            save_init_files(new_list_file_name, experiment_config, agent_config, init_vehicularNetworkEnv)
            agent = HMAIMD_Agent(agent_config=agent_config, environment=init_vehicularNetworkEnv)
            trainer = Trainer(agent_config, agent)
            trainer.run_games_for_agent(temple_agent_config_name=load_name(new_list_file_name, 'temple_agent_config_name'),
                                        temple_agent_name=load_name(new_list_file_name, 'temple_agent_name'),
                                        temple_result_name=load_name(new_list_file_name, 'temple_result_name'),
                                        temple_loss_name=load_name(new_list_file_name, 'temple_loss_name'))
        else:
            correct_list_file_name = project_dir + data + given_list_file_name
            list_file = load_obj(name=correct_list_file_name)
            temple_agent_config = load_obj(name=load_name(list_file, 'temple_agent_config_name'))
            temple_agent = load_obj(name=load_name(list_file, 'temple_agent_name'))
            trainer = Trainer(temple_agent_config, temple_agent)
            trainer.run_games_for_agent(temple_agent_config_name=load_name(list_file, 'temple_agent_config_name'),
                                        temple_agent_name=load_name(list_file, 'temple_agent_name'),
                                        temple_result_name=load_name(list_file, 'temple_result_name'),
                                        temple_loss_name=load_name(list_file, 'temple_loss_name'))

def show_env(environment_file_name):
    environment = load_obj(name=environment_file_name)
    # environment.get_mean_and_second_moment_service_time_of_types()

    print('data_size_of_types: ', np.mean(environment.data_size_of_types) / (1024 * 1024))

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
        for vehicle_index in range(environment.experiment_config.vehicle_number):
            for data_types_index in range(environment.experiment_config.data_types_number):
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
    print('sum_edge_view_in_edge_node: ', sum_edge_view_in_edge_node / environment.experiment_config.time_slots_number)
    
    # print("bandwidth: ", environment.bandwidth)
    print("mean_service_time_of_types: \n", np.mean(environment.experiment_config.mean_service_time_of_types))
    # print("mean_service_time_of_types: \n", environment.experiment_config.mean_service_time_of_types)
    print("second_moment_service_time_of_types: \n", np.mean(environment.experiment_config.second_moment_service_time_of_types))
    # print("second_moment_service_time_of_types: \n", environment.experiment_config.second_moment_service_time_of_types)
    # print("threshold_edge_views_in_edge_node: ", environment.experiment_config.threshold_edge_views_in_edge_node)
    print("\n")

def run_ra_algorithms(given_list_file_name, ra_result_name, num_episodes):
    vehicularNetworkEnv = load_obj(name=environemt_file_name)
    ra_agent = Random_Agent(environment=vehicularNetworkEnv)
    ra_agent.run_n_episodes(num_episodes=num_episodes, temple_result_name=ra_result_name)

def run_ddpg_algorithms(environment_file_name, ddpg_result_name, ddpg_agent_name, num_episodes):
    vehicularNetworkEnv = load_obj(name=environment_file_name)
    ddpg_agent = DDPG_Agent(environment=vehicularNetworkEnv)
    ddpg_agent.run_n_episodes(num_episodes=num_episodes, temple_result_name=ddpg_result_name, agent_name=ddpg_agent_name)

def run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name):
    environment = load_obj(name=environment_file_name)
    ra_agent = Random_Agent(environment=environment)
    ra_agent.run_n_episodes_as_results(num_episodes=num_episodes, result_name=ra_result_name)

def run_ddpg_algorithms_for_results(num_episodes, environment_file_name, agent_name, ddpg_result_name):
    ddpg_agent = load_obj(name=agent_name)
    environment = load_obj(name=environment_file_name)
    ddpg_agent.config_environment(environment=environment)
    ddpg_agent.run_n_episodes_as_results(num_episodes=num_episodes, result_name=ddpg_result_name)

def run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name):
    environment = load_obj(name=environment_file_name)
    agent_config = AgentConfig()
    hyperparameters = {

        "Actor_of_Sensor": {
            "learning_rate": 1e-4,
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
            "learning_rate": 1e-3,
            "linear_hidden_units": [128, 64],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 1e-4,
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
            "learning_rate": 1e-3,
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
        "actor_nodes_learning_updates_per_learning_session": 10,
        "critic_nodes_learning_updates_per_learning_session": 10,
        "actor_reward_learning_updates_per_learning_session": 160,
        "critic_reward_learning_updates_per_learning_session": 160,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)
    agent = IDDPG_Agent(agent_config=agent_config, environment=environment)

    agent.config_environment(environment=environment)
    agent.run_n_episodes_as_results(
        num_episodes=num_episodes, 
        result_name=hmaimd_result_name, 
        environment=environment, 
        actor_nodes_name=actor_nodes_name,
        actor_edge_name=actor_edge_name)


def run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name):
    environment = load_obj(name=environment_file_name)
    agent_config = AgentConfig()
    hyperparameters = {

        "Actor_of_Sensor": {
            "learning_rate": 1e-4,
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
            "learning_rate": 1e-3,
            "linear_hidden_units": [128, 64],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 1e-4,
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
            "learning_rate": 1e-3,
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
        "actor_nodes_learning_updates_per_learning_session": 10,
        "critic_nodes_learning_updates_per_learning_session": 10,
        "actor_reward_learning_updates_per_learning_session": 160,
        "critic_reward_learning_updates_per_learning_session": 160,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)
    hmaimd_agent = HMAIMD_Agent(agent_config=agent_config, environment=environment)

    hmaimd_agent.config_environment(environment=environment)
    hmaimd_agent.run_n_episodes_as_results(
        num_episodes=num_episodes, 
        result_name=hmaimd_result_name, 
        environment=environment, 
        actor_nodes_name=actor_nodes_name,
        actor_edge_name=actor_edge_name)

def show_all_environments():
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    # show_env(environment_file_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_02.pkl"
    # show_env(environment_file_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_03.pkl"
    # show_env(environment_file_name)
    
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_04.pkl"
    # show_env(environment_file_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_05.pkl"
    # show_env(environment_file_name)


    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_05_01.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    show_env(environment_file_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl"
    show_env(environment_file_name)

    print("-" * 64)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    show_env(environment_file_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl"
    show_env(environment_file_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl"
    show_env(environment_file_name)



def gerenate_random_results_changing_bandwidth():
    num_episodes = 100
    test_name = "test_2149" + "_"

    list_environment_file_name = []
    list_result_name = []

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_0_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_5_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_6_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    list_environment_file_name.append(environment_file_name)
    list_result_name.append(ra_result_name)

    list_num_episodes = [100 for _ in range(len(list_environment_file_name))]
    with Pool(5) as p:
        p.map(run_ra_algorithms_for_results, list_num_episodes, list_environment_file_name, list_result_name)

def gerenate_random_results():
    num_episodes = 10
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_1000_bandwidth_3_threshold_015_02.pkl"
    ra_result_name = project_dir + "/Results/" + "ra_results" + environment_file_name[-43:-4] + ".csv"
    run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_1200_bandwidth_3_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + "ra_results" + environment_file_name[-43:-4] + ".csv"
    run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_1800_bandwidth_3_threshold_015_01.pkl"
    ra_result_name = project_dir + "/Results/" + "ra_results" + environment_file_name[-43:-4] + ".csv"
    run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name)
    
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_2200_bandwidth_3_threshold_015_02.pkl"
    ra_result_name = project_dir + "/Results/" + "ra_results" + environment_file_name[-43:-4] + ".csv"
    run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name)

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_02.pkl"
    ra_result_name = project_dir + "/Results/" + "ra_results" + environment_file_name[-43:-4] + ".csv"
    run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name)

def get_hmaimd_results_changing_scenarios():
    num_episodes = 10
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/1000/bandwidth_3_threshold_015_02/2021-10-26-09-10-33/actor_nodes_61e26cb8880640f2848a55b38bce53a0_episode_810.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/1000/bandwidth_3_threshold_015_02/2021-10-26-09-10-33/actor_edge_61e26cb8880640f2848a55b38bce53a0_episode_810.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_1000_bandwidth_3_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/1200/bandwidth_3_threshold_015_01/2021-10-27-18-34-25/actor_nodes_5f4941aebe8b4b4d9a15c43af5e82991_episode_1950.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/1200/bandwidth_3_threshold_015_01/2021-10-27-18-34-25/actor_edge_5f4941aebe8b4b4d9a15c43af5e82991_episode_1950.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_1200_bandwidth_3_threshold_015_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/1800/bandwidth_3_threshold_015_01/2021-10-27-18-37-42/actor_nodes_c7c2d228a35840d39940bcec4bfbc848_episode_1990.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/1800/bandwidth_3_threshold_015_01/2021-10-27-18-37-42/actor_edge_c7c2d228a35840d39940bcec4bfbc848_episode_1990.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_1800_bandwidth_3_threshold_015_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/2200/bandwidth_3_threshold_015_02/2021-10-26-09-17-28/actor_nodes_a13c599945e24185be02d242b7c37947_episode_510.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/2200/bandwidth_3_threshold_015_02/2021-10-26-09-17-28/actor_edge_a13c599945e24185be02d242b7c37947_episode_510.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_2200_bandwidth_3_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)



def get_hmaimd_results_changing_theshold():
    num_episodes = 1
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_03_01/2021-11-12-14-08-18/actor_nodes_6695398cfcad452fa09c3f48d35da3eb_episode_300.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_03_01/2021-11-12-14-08-18/actor_edge_6695398cfcad452fa09c3f48d35da3eb_episode_300.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_04_01/2021-11-12-17-57-29/actor_nodes_3cc0be67b14144b6a330a64cd16497de_episode_380.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_04_01/2021-11-12-17-57-29/actor_edge_3cc0be67b14144b6a330a64cd16497de_episode_380.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_06_01/2021-11-12-23-10-29/actor_nodes_01373c855b444804b2822601ca561eb8_episode_780.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_06_01/2021-11-12-23-10-29/actor_edge_01373c855b444804b2822601ca561eb8_episode_780.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_07_01/2021-11-13-09-04-08/actor_nodes_13d25c4f6ed8416eb892d5cebb719d30_episode_680.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_07_01/2021-11-13-09-04-08/actor_edge_13d25c4f6ed8416eb892d5cebb719d30_episode_680.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_iddpg_results_changing_theshold():
    num_episodes = 10
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_005_02/2021-10-27-12-13-47/actor_nodes_45b1ed479c434fd5bdddd87d793d56f1_episode_540.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_005_02/2021-10-27-12-13-47/actor_edge_45b1ed479c434fd5bdddd87d793d56f1_episode_540.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_005_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_010_02/2021-10-27-12-16-16/actor_nodes_f904b34b659442f6b157387ac3e3ee32_episode_530.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_010_02/2021-10-27-12-16-16/actor_edge_f904b34b659442f6b157387ac3e3ee32_episode_530.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_010_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_020_02/2021-10-27-12-17-41/actor_nodes_afbc676cf6a44fd1a71e2cbed0346b06_episode_530.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_020_02/2021-10-27-12-17-41/actor_edge_afbc676cf6a44fd1a71e2cbed0346b06_episode_530.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_020_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_025_02/2021-10-27-12-19-26/actor_nodes_2973a9a9eac5464da1804731cc1d9beb_episode_520.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_025_02/2021-10-27-12-19-26/actor_edge_2973a9a9eac5464da1804731cc1d9beb_episode_520.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_025_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_nomal_iddpg_results_1():
    num_episodes = 1
    iddpg_name = "iddpg_nomal"
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_1_threshold_05_01/2021-11-15-22-24-33/actor_nodes_8c21ee58474641e09bf2db1134246d45_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_1_threshold_05_01/2021-11-15-22-24-33/actor_edge_8c21ee58474641e09bf2db1134246d45_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_2_threshold_05_01/2021-11-15-22-31-59/actor_nodes_e7d20fc76277454bb252903d2a55a15a_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_2_threshold_05_01/2021-11-15-22-31-59/actor_edge_e7d20fc76277454bb252903d2a55a15a_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_05_01/2021-11-15-16-06-46/actor_nodes_f0480ada534d4672a5dd12f893f142a1_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_05_01/2021-11-15-16-06-46/actor_edge_f0480ada534d4672a5dd12f893f142a1_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_nomal_iddpg_results_2():
    num_episodes = 1
    iddpg_name = "iddpg_nomal"
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_4_threshold_05_01/2021-11-15-22-33-35/actor_nodes_01d67df2e39141088cde7f34658628af_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_4_threshold_05_01/2021-11-15-22-33-35/actor_edge_01d67df2e39141088cde7f34658628af_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_5_threshold_05_01/2021-11-15-22-35-59/actor_nodes_adab626b117744078d0c78156108f19c_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_5_threshold_05_01/2021-11-15-22-35-59/actor_edge_adab626b117744078d0c78156108f19c_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_03_01/2021-11-15-22-38-16/actor_nodes_8a8e81bb589e4048bcd8e1eb41dc4484_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_03_01/2021-11-15-22-38-16/actor_edge_8a8e81bb589e4048bcd8e1eb41dc4484_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_nomal_iddpg_results_3():
    num_episodes = 1
    iddpg_name = "iddpg_nomal"
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_04_01/2021-11-15-22-38-45/actor_nodes_4180eb9dcf98443bba4f0398afc706b9_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_04_01/2021-11-15-22-38-45/actor_edge_4180eb9dcf98443bba4f0398afc706b9_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_06_01/2021-11-15-22-39-05/actor_nodes_6a9f8ed9b86a4ef2b7d669085da62a74_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_06_01/2021-11-15-22-39-05/actor_edge_6a9f8ed9b86a4ef2b7d669085da62a74_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_07_01/2021-11-15-22-39-27/actor_nodes_a9ca65831f9049e3846f913c2f07cba6_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_07_01/2021-11-15-22-39-27/actor_edge_a9ca65831f9049e3846f913c2f07cba6_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_rr_iddpg_results_1():
    num_episodes = 1
    iddpg_name = "iddpg_rr"
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_1_threshold_05_01/2021-11-15-22-47-48/actor_nodes_a26391f6bf6e410282602fefb11e3f41_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_1_threshold_05_01/2021-11-15-22-47-48/actor_edge_a26391f6bf6e410282602fefb11e3f41_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_2_threshold_05_01/2021-11-15-22-48-30/actor_nodes_df5f030faf58488da2f5f10e8a504435_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_2_threshold_05_01/2021-11-15-22-48-30/actor_edge_df5f030faf58488da2f5f10e8a504435_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_05_01/2021-11-15-16-08-37/actor_nodes_a14be6912cfb42d4a473b7d3111dd683_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_05_01/2021-11-15-16-08-37/actor_edge_a14be6912cfb42d4a473b7d3111dd683_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_rr_iddpg_results_2():
    num_episodes = 1
    iddpg_name = "iddpg_rr"
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_4_threshold_05_01/2021-11-15-22-49-02/actor_nodes_68ccde64d15f4f86bb0c473edcf6dacf_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_4_threshold_05_01/2021-11-15-22-49-02/actor_edge_68ccde64d15f4f86bb0c473edcf6dacf_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_5_threshold_05_01/2021-11-15-22-49-31/actor_nodes_3052bf107c49452fae74e2552368b0ae_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_5_threshold_05_01/2021-11-15-22-49-31/actor_edge_3052bf107c49452fae74e2552368b0ae_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_03_01/2021-11-15-22-52-48/actor_nodes_9a6de95fc5b64ffcbd18d3acbe21cf6c_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_03_01/2021-11-15-22-52-48/actor_edge_9a6de95fc5b64ffcbd18d3acbe21cf6c_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_rr_iddpg_results_3():
    num_episodes = 1
    iddpg_name = "iddpg_rr"
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_04_01/2021-11-15-22-53-17/actor_nodes_b3b24bf249f146879eb4240e4fdff0b1_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_04_01/2021-11-15-22-53-17/actor_edge_b3b24bf249f146879eb4240e4fdff0b1_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_06_01/2021-11-15-22-53-41/actor_nodes_d2f93da3492b451ab2c436ead33042c7_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_06_01/2021-11-15-22-53-41/actor_edge_d2f93da3492b451ab2c436ead33042c7_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_07_01/2021-11-15-22-54-05/actor_nodes_b36bb516ced94cc2bc2fdeced84080ca_episode_2000.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/iddpg/bandwidth_3_threshold_07_01/2021-11-15-22-54-05/actor_edge_b36bb516ced94cc2bc2fdeced84080ca_episode_2000.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + iddpg_name + environment_file_name[-43:-4] + ".csv"
    run_iddpg_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_hmaimd_results():
    num_episodes = 1540
    test_name = "test_1907" + "_"
    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_1_threshold_05_01/2021-11-11-12-17-33/actor_nodes_00e7eea6ad7e4bbfb34b78462c1255e1_episode_540.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_1_threshold_05_01/2021-11-11-12-17-33/actor_edge_00e7eea6ad7e4bbfb34b78462c1255e1_episode_540.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_05_01.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + test_name + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_2_threshold_05_01/2021-11-11-12-21-18/actor_nodes_1e01fb5f95ff4eaca515_episode_570.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_2_threshold_05_01/2021-11-11-12-21-18/actor_edge_1e01fb5f95ff4eaca515_episode_570.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + test_name + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_05_01/2021-11-10-20-13-19/actor_nodes_58c99a5453df443089b2df70ce7bf87c_episode_460.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_3_threshold_05_01/2021-11-10-20-13-19/actor_edge_58c99a5453df443089b2df70ce7bf87c_episode_460.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    hmaimd_result_name = project_dir + "/Results/" + test_name + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_4_threshold_05_01/2021-11-11-12-26-04/actor_nodes_95c7ca3043de46c2b8f9_episode_560.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_4_threshold_05_01/2021-11-11-12-26-04/actor_edge_95c7ca3043de46c2b8f9_episode_560.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + test_name + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_5_threshold_05_01/2021-11-11-12-28-23/actor_nodes_6d13ba79e681477e934f_episode_600.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1110_Agents/1116/0800/bandwidth_5_threshold_05_01/2021-11-11-12-28-23/actor_edge_6d13ba79e681477e934f_episode_600.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + test_name + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)


def get_ddpg_results():
    num_episodes = 10
    ddpg_agent_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_01/ddpg.pkl"

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl"
    ddpg_result_name = project_dir + "/Results/" + "ddpg_results" + environment_file_name[-43:-4] + ".csv"
    run_ddpg_algorithms_for_results(num_episodes, environment_file_name, ddpg_agent_name, ddpg_result_name)

if __name__ == '__main__':
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

    # get_hmaimd_results_changing_scenarios()

    # gerenate_random_results()
    # gerenate_random_results_changing_bandwidth()
    # show_all_environments()

    # p = Pool(processes=5)
    # test_name = "test_1911" + "_"

    # list_environment_file_name = []
    # list_result_name = []

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_05_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_05_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_05_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # # list_environment_file_name.append(environment_file_name)
    # # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(2000, environment_file_name, ra_result_name)
    
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_05_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_05_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_03_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_04_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_06_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_07_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    # run_ra_algorithms_for_results(1, environment_file_name, ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_5_threshold_015_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)
    
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_6_threshold_015_01.pkl"
    # ra_result_name = project_dir + "/Results/" + test_name + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # list_environment_file_name.append(environment_file_name)
    # list_result_name.append(ra_result_name)

    # list_num_episodes = [1 for _ in range(len(list_environment_file_name))]

    # for i in range(len(list_num_episodes)):
    #     p.apply_async(run_ra_algorithms_for_results, args=(list_num_episodes[i], list_environment_file_name[i], list_result_name[i],))

    # p.close()
    # p.join()

