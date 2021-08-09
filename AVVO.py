# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""
import numpy as np
from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Config.AgentConfig import AgentConfig
from Config.ExperimentConfig import ExperimentConfig
from Agents.Trainer import Trainer


if __name__ == '__main__':
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
        "update_every_n_steps": 50,
        "learning_updates_per_learning_session": 1,
        "clip_rewards": False}

    agent_config.config(hyperparameters=hyperparameters)

    agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

    trainer = Trainer(agent_config, agent)
    trainer.run_games_for_agent()
