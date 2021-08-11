# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""
import numpy as np
import uuid
import os
import datetime
from file_name import project_dir
from Utilities.FileSaver import save_obj
from Utilities.FileSaver import load_obj
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
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size()))
                 ],
            "final_layer_activation": ["softmax", "softmax"],
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

        "Critic_of_Sensor": {
            "learning_rate": 0.01,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1)),
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size()))],
            "final_layer_activation": "softmax",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

        "Critic_of_Edge": {
            "learning_rate": 0.01,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1)),
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size())),
                 int(0.5 * (
                         vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size()))],
            "final_layer_activation": "softmax",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5,
            "noise_seed": np.random.randint(0, 2 ** 32 - 2),
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

        "Critic_of_Reward": {
            "learning_rate": 0.01,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1)),
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "discount_rate": 0.9,
        "update_every_n_steps": 10,  # 30 times in one episode
        "learning_updates_per_learning_session": 8,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)
    return experiment_config, agent_config, vehicularNetworkEnv


def init_file_name():
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    hourTime = datetime.datetime.now().strftime('%H-%M-%S')
    pwd = project_dir + '/Data/' + dayTime + '-' + hourTime

    if not os.path.exists(pwd):
        os.makedirs(pwd)

    list_file_name = project_dir + '/Data/' + dayTime + '-' + hourTime + '-' + 'list_file_name.pkl'

    uuid_str = uuid.uuid4().hex
    init_experiment_config_name = pwd + '/' + 'init_experiment_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    init_agent_config_name = pwd + '/' + 'init_agent_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    init_environment_name = pwd + '/' + 'init_environment_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_agent_config_name = pwd + '/' + 'temple_agent_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_agent_name = pwd + '/' + 'temple_agent_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_result_name = pwd + '/' + 'temple_result_%s.csv' % uuid_str

    return [list_file_name, init_experiment_config_name, init_agent_config_name, init_environment_name,
            temple_agent_config_name, temple_agent_name, temple_result_name]


def run(first=False, rerun=False, given_list_file_name=None):

    if first:    # run in the first time
        experiment_config, agent_config, vehicularNetworkEnv = init()
        list_file_name = init_file_name()
        save_obj(obj=list_file_name, name=list_file_name[0])
        save_obj(obj=experiment_config, name=list_file_name[1])
        save_obj(obj=agent_config, name=list_file_name[2])
        save_obj(obj=vehicularNetworkEnv, name=list_file_name[3])
        print("save init files successful")
        agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)
        trainer = Trainer(agent_config, agent)
        trainer.run_games_for_agent(temple_agent_config_name=list_file_name[4],
                                    temple_agent_name=list_file_name[5],
                                    temple_result_name=list_file_name[6])

    else:
        if rerun:
            correct_list_file_name = project_dir + '/Data/' + given_list_file_name
            list_file = load_obj(name=correct_list_file_name)
            agent_config = load_obj(list_file[2])
            vehicularNetworkEnv = load_obj(list_file[3])
            agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)
            trainer = Trainer(agent_config, agent)
            trainer.run_games_for_agent(temple_agent_config_name=list_file[4],
                                        temple_agent_name=list_file[5],
                                        temple_result_name=list_file[6])
        else:
            correct_list_file_name = project_dir + '/Data/' + given_list_file_name
            list_file = load_obj(name=correct_list_file_name)
            temple_agent_config = load_obj(name=list_file[4])
            temple_agent = load_obj(name=list_file[5])
            trainer = Trainer(temple_agent_config, temple_agent)
            trainer.run_games_for_agent(temple_agent_config_name=list_file[4],
                                        temple_agent_name=list_file[5],
                                        temple_result_name=list_file[6])


if __name__ == '__main__':
    run(first=True)
    # run(rerun=True, given_list_file_name='2021-08-10-10-19-22-list_file_name.pkl')
    # run(given_list_file_name='2021-08-10-10-42-50-list_file_name.pkl')
