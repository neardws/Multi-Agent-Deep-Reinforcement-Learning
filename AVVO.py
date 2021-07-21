# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""

from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Utilities.Data_structures.Config import AgentConfig
from Utilities.Data_structures.Config import ExperimentConfig
from Agents.Trainer import Trainer

if __name__ == '__main__':
    experiment_config = ExperimentConfig()
    experiment_config.config()

    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)

    agent_config = AgentConfig()

    noise_action_size = vehicularNetworkEnv.get_global_action_size()

    hyperparameters = {
        # Builds a NN with 2 output heads. The first output heads has data_types_number hidden units and
        # uses a softmax activation function and the second output head has data_types_number hidden units and
        # uses a softmax activation function

        "Actor_of_Sensor": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(vehicularNetworkEnv.get_sensor_observation_size() * 1.5),
                 int(vehicularNetworkEnv.get_sensor_observation_size() * 1.5),
                 int(vehicularNetworkEnv.get_sensor_observation_size())],
            "final_layer_activation": ["softmax", "softmax"],
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic_of_Sensor": {
            "learning_rate": 0.01,
            "linear_hidden_units":
                [int(vehicularNetworkEnv.get_critic_size_for_sensor() * 1.5),
                 int(vehicularNetworkEnv.get_critic_size_for_sensor() * 1.5),
                 int(vehicularNetworkEnv.get_critic_size_for_sensor())],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Edge": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(vehicularNetworkEnv.get_actor_input_size_for_edge() * 1.5),
                 int(vehicularNetworkEnv.get_actor_input_size_for_edge() * 1.5),
                 int(vehicularNetworkEnv.get_actor_input_size_for_edge())],
            "final_layer_activation": "softmax",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic_of_Edge": {
            "learning_rate": 0.01,
            "linear_hidden_units":
                [int(vehicularNetworkEnv.get_critic_size_for_edge() * 1.5),
                 int(vehicularNetworkEnv.get_critic_size_for_edge() * 1.5),
                 int(vehicularNetworkEnv.get_critic_size_for_edge())],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Actor_of_Reward": {
            "learning_rate": 0.001,
            "linear_hidden_units":
                [int(vehicularNetworkEnv.get_actor_input_size_for_reward() * 1.5),
                 int(vehicularNetworkEnv.get_actor_input_size_for_reward() * 1.5),
                 int(vehicularNetworkEnv.get_actor_input_size_for_reward())],
            "final_layer_activation": "softmax",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic_of_Reward": {
            "learning_rate": 0.01,
            "linear_hidden_units":
                [int(vehicularNetworkEnv.get_critic_size_for_reward() * 1.5),
                 int(vehicularNetworkEnv.get_critic_size_for_reward() * 1.5),
                 int(vehicularNetworkEnv.get_critic_size_for_reward())],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "discount_rate": 0.9,
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10,
        "clip_rewards": False}

    agent_config.config(noise_action_size=noise_action_size,
                        hyperparameters=hyperparameters)

    agent = HMAIMD_Agent(agent_config=agent_config, environment=vehicularNetworkEnv)

    trainer = Trainer(experiment_config, agent)
    trainer.run_games_for_agent()
