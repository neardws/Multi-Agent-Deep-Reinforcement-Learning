# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：VehicularNetworkEnv.py
@Author  ：Neardws
@Date    ：6/22/21 4:09 下午 
"""
import gym
import numpy as np
from gym import spaces
from Utilities.Data_structures.Config import Experiment_Config


class VehicularNetworkEnv(gym.Env):
    """Vehicular Network Environment that follows gym interface"""
    metadata = {'render.modes': []}

    def __init__(self, experiment_config=Experiment_Config()):
        """
        init the environment via Experiment_Config
        :param experiment_config:
        """
        super(VehicularNetworkEnv, self).__init__()

        assert experiment_config is not None

        self.episode_number = experiment_config.episode_number
        self.max_episode_length = experiment_config.max_episode_length

        self.state = None  # global state
        self.action = None
        self.observation_states = None  # individually observation state for sensor node
        self.reward = None  # external reward
        self.next_state = None
        self.done = False
        self.episode_steps = 0

        self.vehicle_number = experiment_config.vehicle_number
        self.data_types_number = experiment_config.data_types_number
        self.time_slots_number = experiment_config.time_slots_number

        self.arrival_rate_low_bound = experiment_config.arrival_rate_low_bound
        self.arrival_rate_up_bound = experiment_config.arrival_rate_up_bound

        self.mean_service_time_of_types = experiment_config.mean_service_time_of_types
        self.second_moment_service_time_of_types = experiment_config.second_moment_service_time_of_types

        self.data_types_in_vehicles = experiment_config.data_types_in_vehicles
        self.edge_views_in_edge_node = experiment_config.edge_views_in_edge_node
        self.view_required_data = experiment_config.view_required_data

        self.trajectories = experiment_config.trajectories
        self.data_in_edge_node = experiment_config.data_in_edge_node

        """Communication parameters"""
        self.communication_range = experiment_config.communication_range

        self.white_gaussian_noise = experiment_config.additive_white_gaussian_noise
        self.mean_channel_fading_gain = experiment_config.mean_channel_fading_gain
        self.second_moment_channel_fading_gain = experiment_config.second_moment_channel_fading_gain

        self.path_loss_exponent = experiment_config.path_loss_exponent
        self.transmission_power = experiment_config.transmission_power

        self.action_experiences = None

        self.action_time_of_sensor_nodes = None
        self.waiting_time_in_queue = None

        """
        Define action and observation space
        They must be gym.spaces objects
        Action:
            top-level controller:
                [priority, arrival rate] of all sensor node
            bottom-level controller：
                [bandwidth] of all sensor node 
        """
        #  TODO: need to defined activation function
        #  Cause: the sum of bandwidth may exceed 1, it should be minus the average value
        #         and the constraint of arrival rate
        #  Solution: use 'softmax' may work
        #  Fixed Data:  TBD
        self.action_space = spaces.Dict({
            'priority': spaces.Box(low=0, high=1,
                                   shape=(self.vehicle_number, self.data_types_number),
                                   dtype=np.float32),
            'arrival_rate': spaces.Box(low=self.arrival_rate_low_bound,
                                       high=self.arrival_rate_up_bound,
                                       shape=(self.vehicle_number, self.data_types_number),
                                       dtype=np.float32),
            'edge_nodes_bandwidth': spaces.Box(low=0, high=1,
                                               shape=self.vehicle_number,
                                               dtype=np.float32)
        })

        """
        State:
            [Time-slot, Data_types_in_vehicles, Action_time_of_vehicles Edge_views_in_edge_node, 
            View_required_data, Trajectories， Data_in_edge_node]
            View:
            Trajectories:
            Location:
            Bandwidth:
        """
        self.observation_space = spaces.Dict({
            'time': spaces.Discrete(int(self.time_slots_number)),
            'data_types': spaces.MultiBinary(list(self.data_types_in_vehicles.shape)), # the MultiBinary require list
            'action_time': spaces.MultiBinary([self.vehicle_number, self.time_slots_number]),
            'edge_view': spaces.MultiBinary(list(self.edge_views_in_edge_node.shape)),
            'view': spaces.MultiBinary(list(self.view_required_data.shape)),
            'trajectories': spaces.Box(low=0, high=self.communication_range,
                                       shape=(self.vehicle_number, self.time_slots_number), dtype=np.float32),
            'data_in_edge': spaces.MultiBinary([self.data_types_number, self.time_slots_number])
        })


    def reset(self):

        """
        Reset the state of the environment to an initial state
        :return:
        """

        self.episode_steps = 0
        self.state ={
            'time': self.episode_steps,
            'data_types': self.data_types_in_vehicles,
            'action_time': self.action_time_of_sensor_nodes,
            'edge_view': self.edge_views_in_edge_node,
            'view': self.view_required_data,
            'trajectories': self.trajectories,
            'data_in_edge': self.data_in_edge_node
        }
        self.observation_states = None  # individually observation state for sensor node
        self.action = None
        self.reward = None  # external reward
        self.next_state = None
        self.done = False

        """Save action of each time-slot into action experiences"""
        self.action_experiences = list()

        """Init the action time of sensor nodes"""
        self.action_time_of_sensor_nodes = np.zeros((self.vehicle_number, self.time_slots_number))
        self.action_time_of_sensor_nodes[:,0] = 1
        """Init the waiting time of sensor nodes"""
        self.waiting_time_in_queue = np.zeros(shape=(self.vehicle_number, self.data_types_number),
                                         dtype=np.float32)

        return self.state

    def state_to_individually_observation_state(self):

        for vehicle_index in self.vehicle_number:
            observation = np.zeros(shape=(1
                                          + self.data_types_number
                                          + self.time_slots_number
                                          + int(self.edge_views_number * self.time_slots_number)
                                          + int(self.vehicle_number * self.data_types_number * self.edge_views_number)))


    def step(self, action):
        """
        Execute one time step within the environment
        :param action:
        :return: state, action, reward, next_state, done
        """
        self.action = action
        pass

    def compute_reward(self):
        pass

    def update_action_time_and_waiting_time_in_queue(self):
        """
        When the sensor node conduct the action, update the action time
        and the average waiting time in the queue in stable system
        :return:
        """

        for vehicle_index in range(self.vehicle_number):
            """When the action time equal to now time"""
            if self.action_time_of_sensor_nodes[vehicle_index] == self.time_slots_number:
                vehicle_action = list()
                for data_type_index in range(self.data_types_number):
                    if self.data_types_in_vehicles[vehicle_index][data_type_index] == 1:
                        priority = self.action['priority'][vehicle_index][data_type_index]
                        arrival_rate = self.action['arrival_rate'][vehicle_index][data_type_index]
                        vehicle_action.append(
                            {'priority': priority, 'arrival_rate': arrival_rate, 'data_type': data_type_index})
                vehicle_action.sort(key=lambda value: value['priority'])
                max_average_waiting_time = 0
                for index, action in enumerate(vehicle_action):
                    work_load_before_type = 0
                    mu_before_type = 0
                    if index != 0:
                        for i in range(index):
                            work_load_before_type += vehicle_action[i]['arrival_rate'] * \
                                                     self.mean_service_time_of_types[vehicle_action[i]['data_type']]
                            mu_before_type += vehicle_action[i]['arrival_rate'] * \
                                              self.second_moment_service_time_of_types[vehicle_action[i]['data_type']]
                    average_sojourn_time = 1 / (1 - work_load_before_type + action['arrival_rate'] *
                                                self.mean_service_time_of_types[action['data_type']])
                    if index != 0:
                        average_sojourn_time *= self.mean_service_time_of_types[action['data_type']] + \
                                                (mu_before_type / (2 * (1 - work_load_before_type)))
                    else:
                        average_sojourn_time *= self.mean_service_time_of_types[action['data_type']]
                    average_waiting_time = average_sojourn_time - self.mean_service_time_of_types[action['data_type']]

                    """Update the waiting time in queue"""
                    self.waiting_time_in_queue[vehicle_index][action['data_type']] += average_waiting_time

                    if average_waiting_time > max_average_waiting_time:
                        max_average_waiting_time = average_waiting_time

                """Update the action time"""
                self.action_time_of_sensor_nodes[vehicle_index] += int(max_average_waiting_time)

    @staticmethod
    def cover_ratio_to_dB(ratio):
        return 10 * np.log10(ratio)

    @staticmethod
    def cover_dB_to_ratio(dB):
        return np.power(10, (dB / 10))

    @staticmethod
    def cover_dBm_to_W(dBm):
        return np.power(10, (dBm / 10)) / 1000

    @staticmethod
    def cover_W_to_dBm(W):
        return 10 * np.log10(W * 1000)

    @staticmethod
    def cover_W_to_mW(W):
        return W * 1000

    @staticmethod
    def cover_mW_to_W(mW):
        return mW / 1000

    def compute_SNR(self, vehicle_index, time_slot):
        pass

    def compute_SNR_by_distance(self, distance):
        white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(self.white_gaussian_noise)
        channel_fading_gain = np.random.normal(loc=self.mean_channel_fading_gain,
                                               scale=self.second_moment_channel_fading_gain)
        SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
              np.power(distance, -self.path_loss_exponent) * self.transmission_power
        return SNR

    @staticmethod
    def compute_transmission_rate(SNR, bandwidth):
        return VehicularNetworkEnv.cover_MHz_to_Hz(bandwidth) * np.log2(1 + SNR) / (8 * 1024 * 1024)

    @staticmethod
    def cover_MHz_to_Hz(MHz):
        return MHz * 10e6

    @staticmethod
    def computer_SNR_wall_by_noise_uncertainty(noise_uncertainty):
        return (np.power(VehicularNetworkEnv.cover_dB_to_ratio(noise_uncertainty), 2) - 1) / \
               VehicularNetworkEnv.cover_dB_to_ratio(noise_uncertainty)


    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        :param mode:
        :param close:
        :return:
        """
        pass


if __name__ == '__main__':

    # ratio = 2
    # print(10 * np.log10(ratio))
    # dB = 2
    # print(np.power(10, (dB / 10)))

    # mean_additive_white_gaussian_noise = 1 / np.power(10,10)
    # mean_additive_white_gaussian_noise = 10e-11
    # second_moment_additive_white_gaussian_noise = 0

    mean_channel_fading_gain = 2
    second_moment_channel_fading_gain = 0.4

    distance = 100
    path_loss_exponent = 3
    transmission_power = 0.001

    # white_gaussian_noise = np.random.normal(loc=mean_additive_white_gaussian_noise,
    #                                         scale=second_moment_additive_white_gaussian_noise)

    white_gaussian_noise = VehicularNetworkEnv.cover_dBm_to_W(-70)
    channel_fading_gain = np.random.normal(loc=mean_channel_fading_gain,
                                           scale=second_moment_channel_fading_gain)
    SNR = (1 / white_gaussian_noise) * np.power(np.abs(channel_fading_gain), 2) * \
          (1 / np.power(distance, path_loss_exponent)) * transmission_power

    print(white_gaussian_noise)
    # print(VehicularNetworkEnv.cover_W_to_dBm(mean_additive_white_gaussian_noise))

    print("Transmission power")
    print(VehicularNetworkEnv.cover_W_to_dBm(transmission_power))

    print("SNR")
    print(SNR)
    print(str(10 * np.log10(SNR)) + "dB")

    print(VehicularNetworkEnv.compute_transmission_rate(SNR, bandwidth=0.1))

    # print(VehicularNetworkEnv.cover_dB_to_ratio(2))
    #
    # noise_uncertainty = np.random.random() * (3 - 1) + 1
    # # noise_uncertainty = 1
    # print("noise_uncertainty:" + str(noise_uncertainty))
    # SNR_wall = VehicularNetworkEnv.computer_SNR_wall_by_noise_uncertainty(noise_uncertainty)
    # print(SNR_wall)
    # print(VehicularNetworkEnv.cover_ratio_to_dB(SNR_wall))

    # dBm = 23
    # print(np.power(10, (dBm / 10)) / 1000)
    #
    # W = 10e-11
    # print(10 * np.log10(W * 1000))
    #
    # mW = 10e-12
    # print(VehicularNetworkEnv.cover_mW_to_W(mW))
    #
    # print(VehicularNetworkEnv.cover_W_to_dBm(10e-11))
    #
    # print(VehicularNetworkEnv.cover_dBm_to_W(10))
    #
    # print(VehicularNetworkEnv.cover_dBm_to_W(-70))


    # print(VehicularNetworkEnv.cover_dBm_to_W(-70) * 10e6)