# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_AVVO.py
@Author  ：Neardws
@Date    ：7/27/21 3:12 下午 
"""
import yaml
import numpy as np
import torch
from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Agents.Trainer import Trainer


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)


def test_SNR():
    experiment_config = ExperimentConfig()
    experiment_config.config()

    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)
    vehicularNetworkEnv.reset()

    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=500))
    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=1000))
    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=1250))
    print(vehicularNetworkEnv.compute_SNR_by_distance(distance=1500))
    a = np.random.uniform(low=vehicularNetworkEnv.config.noise_uncertainty_low_bound,
                          high=vehicularNetworkEnv.config.noise_uncertainty_up_bound)
    print(a)
    b = vehicularNetworkEnv.computer_SNR_wall_by_noise_uncertainty(noise_uncertainty=1)
    print(b)
    print(vehicularNetworkEnv.cover_ratio_to_dB(b))
    # vehicularNetworkEnv.get_mean_and_second_moment_service_time_of_types()


def test_init():
    experiment_config = ExperimentConfig()
    experiment_config.config()

    # for item in experiment_config.__dict__:
    #     print(item)
    #     print(experiment_config.__getattribute__(item))
    #     print(type(experiment_config.__getattribute__(item)))
    #     print()

    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)
    vehicularNetworkEnv.reset()

    # for item in vehicularNetworkEnv.__dict__:
    #     print(item)
    #     print(vehicularNetworkEnv.__getattribute__(item))
    #     print(type(vehicularNetworkEnv.__getattribute__(item)))
    #     print()

    agent_config = AgentConfig()

    hyperparameters = {
        # Builds a NN with 2 output heads. The first output heads has data_types_number hidden units and
        # uses a softmax activation function and the second output head has data_types_number hidden units and
        # uses a softmax activation function

        "Actor_of_Sensor": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size())),
                 int(0.5 * (vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size())),
                 int(0.25 * (vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size()))
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
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1)),
                 int(0.25 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size())),
                 int(0.5 * (vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size())),
                 int(0.25 * (vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size()))],
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
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1)),
                 int(0.25 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size())),
                 int(0.5 * (vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size())),
                 int(0.25 * (vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size()))],
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
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1)),
                 int(0.25 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "discount_rate": 0.9,
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)

    for item in agent_config.__dict__:
        if isinstance(agent_config.__getattribute__(item), dict):
            print(item)
            print(yaml.dump(agent_config.__getattribute__(item), sort_keys=False, default_flow_style=False))
            print()
        else:
            print(item)
            print(agent_config.__getattribute__(item))
            print(type(agent_config.__getattribute__(item)))
            print()


def test_HMAIMD():
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
                        vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size())),
                 int(0.5 * (
                         vehicularNetworkEnv.get_sensor_observation_size() + vehicularNetworkEnv.get_sensor_action_size())),
                 int(0.25 * (
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
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1)),
                 int(0.25 * (vehicularNetworkEnv.get_critic_size_for_sensor() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(0.75 * (
                        vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size())),
                 int(0.5 * (
                         vehicularNetworkEnv.get_actor_input_size_for_edge() + vehicularNetworkEnv.get_edge_action_size())),
                 int(0.25 * (
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
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1)),
                 int(0.25 * (vehicularNetworkEnv.get_critic_size_for_edge() + 1))],
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
                         vehicularNetworkEnv.get_actor_input_size_for_reward() + vehicularNetworkEnv.get_reward_action_size())),
                 int(0.25 * (
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
                 int(0.5 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1)),
                 int(0.25 * (vehicularNetworkEnv.get_critic_size_for_reward() + 1))],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "discount_rate": 0.9,
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)

    agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

    agent.reset_game()


if __name__ == '__main__':
    test_HMAIMD()
