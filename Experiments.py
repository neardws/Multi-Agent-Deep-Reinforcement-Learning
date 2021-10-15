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


def run_ra_algorithms(given_list_file_name, num_episodes=5000):
    correct_list_file_name = project_dir + data + given_list_file_name
    time = given_list_file_name.replace("-list_file_name.pkl", "")
    ra_result_name = project_dir + data + time + "-ra_result.csv"
    ddpg_result_name = project_dir + data + time + "-ddpg_result.csv"
    idpg_result_name = project_dir + data + time + "-idpg_result.csv"
    list_file = load_obj(name=correct_list_file_name)
    init_vehicularNetworkEnv = load_obj(load_name(list_file, 'init_environment_name'))
    ra_agent = Random_Agent(environment=init_vehicularNetworkEnv)
    ra_agent.run_n_episodes(num_episodes=num_episodes, temple_result_name=ra_result_name)
    # ddpg_agent = DDPG_Agent(environment=init_vehicularNetworkEnv)
    # ddpg_agent.run_n_episodes(num_episodes=num_episodes, temple_result_name=None)
    # idpg_agent = IDPG_Agent(environment=init_vehicularNetworkEnv)
    # idpg_agent.run_n_episodes(num_episodes=num_episodes, temple_result_name=None)


def run_ddpg_algorithms(given_list_file_name, num_episodes=5000):
    correct_list_file_name = project_dir + data + given_list_file_name
    time = given_list_file_name.replace("-list_file_name.pkl", "")
    ddpg_result_name = project_dir + data + time + "-ddpg_result_e6.csv"
    list_file = load_obj(name=correct_list_file_name)
    init_vehicularNetworkEnv = load_obj(load_name(list_file, 'init_environment_name'))
    ddpg_agent = DDPG_Agent(environment=init_vehicularNetworkEnv)
    ddpg_agent.run_n_episodes(num_episodes=num_episodes, temple_result_name=ddpg_result_name)


if __name__ == '__main__':
    # run(first=True)

    # run(given_list_file_name='2021-09-22-00-47-47-list_file_name.pkl')

    run_ddpg_algorithms(given_list_file_name='2021-09-14-04-49-44-list_file_name.pkl')
