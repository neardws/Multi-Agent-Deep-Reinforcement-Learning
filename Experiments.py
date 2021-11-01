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
from File_Name import project_dir, data
from Utilities.FileOperator import load_obj
from Utilities.FileOperator import init_file_name
from Utilities.FileOperator import save_init_files
from Utilities.FileOperator import load_name
from Agents.HMAIMD import HMAIMD_Agent
from Agents.Random_Agent import Random_Agent
from Agents.DDPG_Agent import DDPG_Agent
# from Agents.IDPG_Agent import IDPG_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Agents.Trainer import Trainer


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

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_02.pkl"
    # ra_result_name = project_dir + "/Results/" + "ra_results" + environment_file_name[-43:-4] + ".csv"
    # run_ra_algorithms_for_results(num_episodes, environment_file_name, ra_result_name)

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
    num_episodes = 10
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_005_02/2021-10-25-22-29-31/actor_nodes_2514213c7df94f44b15743517ed784c4_episode_700.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_005_02/2021-10-25-22-29-31/actor_edge_2514213c7df94f44b15743517ed784c4_episode_700.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_005_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_threshold_015_02/2021-10-25-17-38-13/actor_nodes_b2077735496c4872832e7b35bdc6c5e5_episode_470.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_threshold_015_02/2021-10-25-17-38-13/actor_edge_b2077735496c4872832e7b35bdc6c5e5_episode_470.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_010_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_020_02/2021-10-25-22-34-19/actor_nodes_8df1cd6e22a8492292dc9a954e99b4d4_episode_250.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_020_02/2021-10-25-22-34-19/actor_edge_8df1cd6e22a8492292dc9a954e99b4d4_episode_250.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_020_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_025_02/2021-10-25-22-35-33/actor_nodes_2b7bf5953a7f4ed6966973848e498234_episode_430.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_025_02/2021-10-25-22-35-33/actor_edge_2b7bf5953a7f4ed6966973848e498234_episode_430.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_025_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

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

def get_iddpg_results():
    num_episodes = 10
    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_threshold_015_02/2021-10-26-22-27-57/actor_nodes_81c8110d904546588b53741559b01e23_episode_650.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_threshold_015_02/2021-10-26-22-27-57/actor_edge_81c8110d904546588b53741559b01e23_episode_650.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_5_threshold_015_02/2021-10-26-22-30-10/actor_nodes_eff006f478af46d48c4e9ac7462be2dd_episode_650.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_5_threshold_015_02/2021-10-26-22-30-10/actor_edge_eff006f478af46d48c4e9ac7462be2dd_episode_650.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_5_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_02/2021-10-26-22-13-39/actor_nodes_5c78edb56f29466bbbcd1da47fba757e_episode_500.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_02/2021-10-26-22-13-39/actor_edge_5c78edb56f29466bbbcd1da47fba757e_episode_500.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_5_threshold_015_02/2021-10-26-22-31-43/actor_nodes_12a801d3d95a40e9a9794a2534f246b0_episode_640.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_5_threshold_015_02/2021-10-26-22-31-43/actor_edge_12a801d3d95a40e9a9794a2534f246b0_episode_640.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_5_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_4_threshold_015_02/2021-10-26-22-33-18/actor_nodes_9563768effd24bc7b909c3645d23bb65_episode_640.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_4_threshold_015_02/2021-10-26-22-33-18/actor_edge_9563768effd24bc7b909c3645d23bb65_episode_640.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_015_02.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "iddpg_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)



def get_hmaimd_results():
    num_episodes = 10
    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_1_threshold_015_02/2021-10-25-17-37-00/actor_nodes_0f9bfe27e8e1435a9d1c770e1551bb5a_episode_460.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_1_threshold_015_02/2021-10-25-17-37-00/actor_edge_0f9bfe27e8e1435a9d1c770e1551bb5a_episode_460.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_1_threshold_015_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_threshold_015_02/2021-10-25-17-38-13/actor_nodes_b2077735496c4872832e7b35bdc6c5e5_episode_470.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_2_threshold_015_02/2021-10-25-17-38-13/actor_edge_b2077735496c4872832e7b35bdc6c5e5_episode_470.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_threshold_015_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_02/2021-10-25-15-08-41/actor_nodes_bc146d79a8984ca88734da363812fd79_episode_700.pkl"
    actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_02/2021-10-25-15-08-41/actor_edge_bc146d79a8984ca88734da363812fd79_episode_700.pkl"
    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_2_5_threshold_015_02_03.pkl"
    hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_5_threshold_015_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_4_threshold_015_02/2021-10-25-17-42-19/actor_nodes_04f41676570e4f90b4613e5a5d2b64d2_episode_440.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_4_threshold_015_02/2021-10-25-17-42-19/actor_edge_04f41676570e4f90b4613e5a5d2b64d2_episode_440.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_4_threshold_015_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

    # actor_nodes_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_01/2021-10-25-10-22-14/actor_nodes_e65a583eb5fa4fffbcb9_episode_570.pkl"
    # actor_edge_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_01/2021-10-25-10-22-14/actor_edge_e65a583eb5fa4fffbcb9_episode_570.pkl"
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_5_threshold_015_02.pkl"
    # hmaimd_result_name = project_dir + "/Results/" + "hmaimd_result" + environment_file_name[-43:-4] + ".csv"
    # run_hmaimd_algorithms_for_results(num_episodes, environment_file_name, actor_nodes_name, actor_edge_name, hmaimd_result_name)

def get_ddpg_results():
    num_episodes = 10
    ddpg_agent_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Agents/1116/0800/bandwidth_3_threshold_015_01/ddpg.pkl"

    environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl"
    ddpg_result_name = project_dir + "/Results/" + "ddpg_results" + environment_file_name[-43:-4] + ".csv"
    run_ddpg_algorithms_for_results(num_episodes, environment_file_name, ddpg_agent_name, ddpg_result_name)

if __name__ == '__main__':

    # get_ddpg_results()
    # get_iddpg_results()
    # run(first=True)

    # run(given_list_file_name='2021-09-22-00-47-47-list_file_name.pkl')

    # run_ddpg_algorithms(given_list_file_name='2021-10-19-16-54-31-list_file_name.pkl')

    # data = '/Data/Agents/1116/0800/bandwidth_3_threshold_015_01/'
    # environment_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Environments/Data/vehicle_1116_0800_bandwidth_3_threshold_015_02.pkl"
    # ddpg_result_name = project_dir + data + "ddpg_result_5.csv"
    # ddpg_agent_name = project_dir + data + "ddpg_5.pkl"
    # num_episodes = 5000
    # run_ddpg_algorithms(environment_file_name, ddpg_result_name, ddpg_agent_name, num_episodes)
    
    # num_episodes = 10
    # get_hmaimd_results()
    # gerenate_random_results()
    # get_hmaimd_results_changing_theshold()
    # get_iddpg_results_changing_theshold()

    # get_hmaimd_results_changing_scenarios()

    gerenate_random_results()

