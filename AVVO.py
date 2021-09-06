# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""
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
            "learning_rate": 0.00001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size())),
                 int(0.5 * (
                         vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size()))
                 ],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25,
            "action_noise_std": 0.005,
            "action_noise_clipping_range": 1.0
        },

        "Critic_of_Sensor": {
            "learning_rate": 0.00001,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1)),
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1))],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 0.00001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size())),
                 int(0.5 * (
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
            "action_noise_std": 0.005,
            "action_noise_clipping_range": 1.0
        },

        "Critic_of_Edge": {
            "learning_rate": 0.00001,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1)),
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1))],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.0001,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 0.0001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size())),
                 int(0.5 * (
                         vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size()))],
            "final_layer_activation": "softmax",
            "batch_norm": False,
            "tau": 0.001,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

        "Critic_of_Reward": {
            "learning_rate": 0.0001,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1)),
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1))],
            "final_layer_activation": "tanh",
            "batch_norm": False,
            "tau": 0.001,
            "gradient_clipping_norm": 5
        },

        "discount_rate": 0.996,
        "update_every_n_steps": 300,  # 30 times in one episode
        "learning_updates_per_learning_session": 32,
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


if __name__ == '__main__':
    # run(first=True)
    # run(rerun=True, given_list_file_name='2021-08-10-10-19-22-list_file_name.pkl')

    # run(rerun=True, given_list_file_name='2021-08-21-05-56-04-list_file_name.pkl')
    # run(given_list_file_name='2021-08-28-09-11-22-list_file_name.pkl')
    # run(given_list_file_name='2021-08-29-23-03-17-list_file_name.pkl')

    # run(given_list_file_name='2021-08-30-02-18-29-list_file_name.pkl')

    # run(given_list_file_name='2021-08-30-02-32-49-list_file_name.pkl')

    # run(given_list_file_name='2021-08-30-04-01-44-list_file_name.pkl')

    # run(rerun=True, given_list_file_name='2021-08-30-04-45-25-list_file_name.pkl')
    # run(given_list_file_name='2021-08-30-04-45-25-list_file_name.pkl')

    # run(given_list_file_name='2021-08-31-07-23-38-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-00-58-25-list_file_name.pkl')

    # run(rerun=True, given_list_file_name='2021-09-01-02-10-58-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-02-22-26-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-02-35-26-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-03-36-37-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-03-58-12-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-04-42-32-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-06-43-51-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-07-14-31-list_file_name.pkl')

    # run(given_list_file_name='2021-09-01-22-23-27-list_file_name.pkl')

    # run(given_list_file_name='2021-09-02-03-20-11-list_file_name.pkl')

    # run(given_list_file_name='2021-09-03-03-59-01-list_file_name.pkl')
    #
    # run(given_list_file_name='2021-09-04-09-37-24-list_file_name.pkl')
    #
    # run(given_list_file_name='2021-09-05-00-29-24-list_file_name.pkl')
    #
    #
    run(given_list_file_name='2021-09-05-06-52-00-list_file_name.pkl')





