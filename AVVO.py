# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：AVVO.py
@Author  ：Neardws
@Date    ：7/11/21 3:25 下午 
"""

from Agents.HMAIMD import HMAIMD_Agent
from Environments.VehicularNetworkEnv.envs import VehicularNetworkEnv
from Utilities.Data_structures.Config import Agent_Config
from Utilities.Data_structures.Config import Experiment_Config
from Agents.Trainer import Trainer


if __name__== '__main__':
    experiment_config = Experiment_Config()
    vehicularNetworkEnv = VehicularNetworkEnv(experiment_config)
    agent_config = Agent_Config()

    agent_config.seed = 1
    agent_config.num_episodes_to_run = 1000
    agent_config.file_to_save_data_results = None
    agent_config.file_to_save_results_graph = None
    agent_config.show_solution_score = False
    agent_config.visualise_individual_results = False
    agent_config.visualise_overall_agent_results = True
    agent_config.standard_deviation_results = 1.0
    agent_config.runs_per_agent = 3
    agent_config.use_GPU = True
    agent_config.overwrite_existing_results_file = False
    agent_config.randomise_random_seed = True
    agent_config.save_model = False

    agent_config.hyperparameters = {
        # Builds a NN with 2 output heads. The first output heads has data_types_number hidden units and
        # uses a softmax activation function and the second output head has data_types_number hidden units and
        # uses a softmax activation function

        "Actor_of_Sensor": {
            "learning_rate": 0.001,
            "linear_hidden_units": [int(vehicularNetworkEnv.get_sensor_observations_size() / 2), int(vehicularNetworkEnv.get_sensor_observations_size() / 2)],
            "final_layer_activation": ["softmax", "softmax"],
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.01,
            "linear_hidden_units": [50, 50, 50],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 30000,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "batch_size": 256,
        "discount_rate": 0.9,
        "mu": 0.0,
        "theta": 0.15,
        "sigma": 0.25,
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10,
        "clip_rewards": False}

    AGENTS = [DDPG]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()