# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning
@File    ：Config.py
@Author  ：Neardws
@Date    ：6/16/21 4:53 下午
"""
import numpy as np

class Experiment_Config(object):
    """
    Object to hold the config requirements for an experiment
    :arg
        vehicle_number: number of sensor nodes
        data_types_number: number of data types in the system
        time_slots_number: number of time-slots in one experiment
        edge_views_number: number of edge views in the system
        communication_range: communication range of edge node
        transmission_power: transmission power of edge node
        bandwidth: bandwidth of edge node
        additive_white_gaussian_noise: additive white gaussian noise of transmission
        channel_fading_gain: channel fading gain of transmission
        path_loss_exponent: path loss exponent of transmission
        data_types_in_vehicles： data types in sensor node， which is randomly generated
        view_required_data： view required data at each time-slot， which is randomly generated
    """
    def __init__(self):
        """Experiment Config"""
        self.episode_number = None
        self.max_episode_length = None
        """Some constant number"""
        self.vehicle_number = None
        self.data_types_number = None
        self.time_slots_number = None
        self.edge_views_number = None
        """The parameters related with transmission queue"""
        self.seed_data_size_of_types = None
        self.data_size_low_bound = None
        self.data_size_up_bound = None
        self.data_size_of_types = None
        self.mean_service_time_of_types = None
        self.second_moment_service_time_of_types = None
        """The parameters related with wireless transmission"""
        self.communication_range = None
        self.transmission_power = None
        self.bandwidth = None

        self.mean_additive_white_gaussian_noise = None  # white gaussian noise according to Gauss Distribution
        self.second_moment_additive_white_gaussian_noise = None
        self.mean_channel_fading_gain = None  # channel fading gain according to Gauss Distribution
        self.second_moment_channel_fading_gain = None

        self.path_loss_exponent = None

        """Some parameters of sensor node"""
        self.arrival_rate_low_bound = None
        self.arrival_rate_up_bound = None
        """Random generated value, the relationship of data types, edge views, vehicles, and edge node"""
        self.seed_data_types_in_vehicles = None
        self.seed_edge_views_in_edge_node = None
        self.seed_view_required_data = None
        self.threshold_data_types_in_vehicles = None
        self.threshold_edge_views_in_edge_node = None
        self.threshold_view_required_data = None
        self.data_types_in_vehicles = None
        self.edge_views_in_edge_node = None
        self.view_required_data = None
        """State varying with time"""
        self.trajectories = None
        self.data_in_edge_node = None

    def config(self,
               vehicle_number,
               data_types_number,
               time_slots_number,
               edge_views_number,

               seed_data_size_of_types,
               data_size_low_bound,
               data_size_up_bound,

               communication_range,
               transmission_power,
               bandwidth,
               mean_additive_white_gaussian_noise,
               second_moment_additive_white_gaussian_noise,
               mean_channel_fading_gain,
               second_moment_channel_fading_gain,

               channel_fading_gain,
               path_loss_exponent,

               threshold_data_types_in_vehicles,
               threshold_edge_views_in_edge_node,
               threshold_view_required_data):
        """The setup number"""
        self.vehicle_number = vehicle_number
        self.data_types_number = data_types_number
        self.time_slots_number = time_slots_number
        self.edge_views_number = edge_views_number

        self.data_size_low_bound = data_size_low_bound
        self.data_size_up_bound = data_size_up_bound

        self.seed_data_size_of_types = seed_data_size_of_types
        np.random.seed(self.seed_data_size_of_types)
        self.data_size_of_types = np.random.uniform(low=self.data_size_low_bound,
                                                    high=self.data_size_up_bound,
                                                    size=self.data_types_number)

        self.communication_range = communication_range
        self.transmission_power = transmission_power
        self.bandwidth = bandwidth

        self.mean_additive_white_gaussian_noise = mean_additive_white_gaussian_noise
        self.second_moment_additive_white_gaussian_noise = second_moment_additive_white_gaussian_noise
        self.mean_channel_fading_gain = mean_channel_fading_gain
        self.second_moment_channel_fading_gain = second_moment_channel_fading_gain

        self.path_loss_exponent = path_loss_exponent


        """Random generated of data types in all vehicles"""
        self.seed_data_types_in_vehicles = np.random.randint(0, 2**32 - 2)
        np.random.seed(self.seed_data_types_in_vehicles)
        self.data_types_in_vehicles = np.random.rand(vehicle_number, data_types_number)
        for value in np.nditer(self.data_types_in_vehicles, op_flags=['readwrite']):
            if value <= threshold_data_types_in_vehicles:
                value[...] = 1
            else:
                value[...] = 0
        """Random generated of edge views requirement at each time-slot in one edge node"""
        self.seed_edge_views_in_edge_node = np.random.randint(0, 2**32 - 2)
        np.random.seed(self.seed_edge_views_in_edge_node)
        self.edge_views_in_edge_node = np.random.rand(edge_views_number, time_slots_number)
        for value in np.nditer(self.edge_views_in_edge_node, op_flags=['readwrite']):
            if value <= threshold_edge_views_in_edge_node:
                value[...] = 1
            else:
                value[...] = 0
        """Random generated of view required data"""
        self.seed_view_required_data = np.random.randint(0, 2**32 - 2)
        np.random.seed(self.seed_view_required_data)
        self.view_required_data = np.random.rand(vehicle_number, data_types_number, edge_views_number)
        for value in np.nditer(self.view_required_data, flags=['multi_index'], op_flags=['readwrite']):
            if self.data_types_in_vehicles[tuple(value.multi_index)[0]][tuple(value.multi_index)[1]] == 1 and\
                    value[...] <= threshold_view_required_data:
                value[...] = 1
            else:
                value[...] = 0
        """Trajectories and data in edge node"""
        self.trajectories = np.zeros(shape=(self.vehicle_number, self.time_slots_number), dtype=np.float32)
        self.data_in_edge_node = np.zeros(shape=(vehicle_number, data_types_number, time_slots_number))

class Agent_Config(object):
    """
    Object to hold the config requirements for an agent/game
    :arg
        seed: seed for random number, to make sure the result can be recurrent
        environment: environment where agent interact with
        requirement_to_solve_game: # TODO fix the meaning
        num_episodes_to_run: the number of episodes
        file_to_save_data_results: file to save the result
        file_to_save_results_graph: # TODO may need more graph for the experiment
        runs_per_agents: #TODO Fix the meaning
        visualise_overall_results: show the overall results or not
        visualise_individual_results: show the results individual or not
        hyperparameters: the parameters of NN, neural network
        use_GPU: is the data on the GPU, select devices to run
        overwrite_existing_results_file: overwrite the result file or not
        save_model: save the model or not
        standard_deviation_results: # TODO fix the meaning
        randomise_random_seed: # TODO fix the meaning
        show_solution_score: show the solution score or not
        debug_mode: in debug mode or not
    """
    def __init__(self):
        self.seed = None   # 随机数种子
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False

    def config(self,
               seed,
               environment,
               requirements_to_solve_game,
               num_episodes_to_run,
               file_to_save_data_results,
               file_to_save_results_graph,
               runs_per_agent,
               visualise_overall_results,
               visualise_individual_results,
               hyperparameters,
               use_GPU,
               overwrite_existing_results_file,
               save_model = False,
               standard_deviation_results = 1.0,
               randomise_random_seed = True,
               show_solution_score = False,
               debug_mode = False):
        self.seed = seed
        self.environment = environment
        self.requirements_to_solve_game = requirements_to_solve_game
        self.num_episodes_to_run = num_episodes_to_run
        self.file_to_save_data_results = file_to_save_data_results
        self.file_to_save_results_graph = file_to_save_results_graph
        self.runs_per_agent = runs_per_agent
        self.visualise_overall_results = visualise_overall_results
        self.visualise_individual_results = visualise_individual_results
        self.hyperparameters = hyperparameters
        self.use_GPU = use_GPU
        self.overwrite_existing_results_file = overwrite_existing_results_file
        self.save_model = save_model
        self.standard_deviation_results = standard_deviation_results
        self.randomise_random_seed = randomise_random_seed
        self.show_solution_score = show_solution_score
        self.debug_mode = debug_mode



if __name__ == '__main__':
    # np.random.seed(1)
    # print(np.random.rand(10))
    # np.random.seed(1)
    # print(np.random.rand(10))
    # print(np.random.randint(0,2,10))
    # print(np.random.randint(0,2,10))

    # TODO test this code
    action_time_of_sensor_nodes = np.zeros((2, 3))
    print(action_time_of_sensor_nodes.shape)
    print(action_time_of_sensor_nodes)
    action_time_of_sensor_nodes[:, 0] = 1
    print(action_time_of_sensor_nodes)