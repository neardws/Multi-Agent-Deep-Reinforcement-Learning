# Hierarchical-Reinforcement-Learning
Hierarchical Reinforcement Learning to solve Minimizing the Average Age of View (MAAV) Problem

![](https://img.shields.io/github/issues/neardws/Hierarchical-Reinforcement-Learning)
![](https://img.shields.io/github/forks/neardws/Hierarchical-Reinforcement-Learning)
![](https://img.shields.io/github/stars/neardws/Hierarchical-Reinforcement-Learning)
![](https://img.shields.io/github/license/neardws/Hierarchical-Reinforcement-Learning)

### TODO

- [ ] Unit testing
- [ ] Add some comparison algorithm

#### Analytics

- [ ] Remove the noise when testing
- [ ] The relationship between the number of received packets and the bandwidth
- [ ] Show more details in results
- [ ] 


### Done

- [x] Update the readme.md 2021-07-03
- [x] Renew Configuration 2021-07-04
- [x] NN Input and Output Dimensions 2021-07-05
- [x] Initialization of Actor Network and Critic Network for sensor nodes, edge node, and reward function, respectively 2021-07-06
- [x] Workflow of HMAIMD_Agent 2021-07-06
- [x] Read DDPG in detail 2021-07-06
- [x] Transplant implementation code from DDPG 2021-07-09
- [x] Create NN functions for actor and critic networks 2021-07-11
- [x] Renew buffer input and output 2021-07-12
- [x] Save NN input and output tensor into self.Parameters 2021-07-12
- [x] Init and update NN input and output 2021-07-13
- [x] Normalization the input of NN 2021-07-13
- [x] combined_action function realization 2021-07-13
- [x] Transfer NN output to action 2021-07-14
- [x] step in VehicularNetworkEnv 2021-07-15
- [x] auto adjust learning rate 2021-07-16
- [x] Trajectories related, such as define global trajectory within the experiment, and add trajectory predicted time in experiment parameters 2021-07-19
- [x] Reorganize parameters in config according to HMAIMD_Agent 2021-07-20
- [x] sensor_nodes_observations dimensions have some issues，need add unit testing 2021-07-26
- [x] Code review 2021-07-27
- [x] bug found on global trajectory process with zero values, and view required data maybe all zero in one view which means it may require no data 2021-07-29
- [x] Unit testing finished in VehicularNetworkEnv.py 2021-07-30
- [x] Add multi outputs heads in Actor of Sensor node 2021-08-07
- [x] Add predicted trajectories into global trajectory 2021-08-07
- [x] Save configuration and others to recurring results 2021-08-10
- [x] Interruption and recovery the training process 2021-08-10
- [x] Add draw results 2021-08-11
- [x] Remove noise in action 2021-08-11
- [x] Test Replay Buffer 2021-08-11

### Env Info

data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 26
edge_view_required_data: 19
edge_view_required_data: 26
edge_view_required_data: 24
edge_view_required_data: 27
edge_view_required_data: 33
edge_view_required_data: 28
edge_view_required_data: 29
edge_view_required_data: 25
edge_view_required_data: 18
edge_view_required_data:  25.5
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 19.956368780622327
second_moment_service_time_of_types: 
 0.0028811580669187834


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 26
edge_view_required_data: 19
edge_view_required_data: 26
edge_view_required_data: 24
edge_view_required_data: 27
edge_view_required_data: 33
edge_view_required_data: 28
edge_view_required_data: 29
edge_view_required_data: 25
edge_view_required_data: 18
edge_view_required_data:  25.5
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 9.978144334364275
second_moment_service_time_of_types: 
 0.0007203110297611713


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 26
edge_view_required_data: 19
edge_view_required_data: 26
edge_view_required_data: 24
edge_view_required_data: 27
edge_view_required_data: 33
edge_view_required_data: 28
edge_view_required_data: 29
edge_view_required_data: 25
edge_view_required_data: 18
edge_view_required_data:  25.5
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 6.652092629515526
second_moment_service_time_of_types: 
 0.00032013733955035025


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 26
edge_view_required_data: 19
edge_view_required_data: 26
edge_view_required_data: 24
edge_view_required_data: 27
edge_view_required_data: 33
edge_view_required_data: 28
edge_view_required_data: 29
edge_view_required_data: 25
edge_view_required_data: 18
edge_view_required_data:  25.5
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 4.9890690535959
second_moment_service_time_of_types: 
 0.0001800954070329668


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 26
edge_view_required_data: 19
edge_view_required_data: 26
edge_view_required_data: 24
edge_view_required_data: 27
edge_view_required_data: 33
edge_view_required_data: 28
edge_view_required_data: 29
edge_view_required_data: 25
edge_view_required_data: 18
edge_view_required_data:  25.5
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 3.9912544248975155
second_moment_service_time_of_types: 
 0.00011525671774849044


----------------------------------------------------------------
data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 21
edge_view_required_data: 7
edge_view_required_data: 9
edge_view_required_data: 12
edge_view_required_data: 12
edge_view_required_data: 20
edge_view_required_data: 12
edge_view_required_data: 12
edge_view_required_data: 15
edge_view_required_data: 14
edge_view_required_data:  13.4
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 6.652092629515526
second_moment_service_time_of_types: 
 0.00032013733955035025


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 25
edge_view_required_data: 12
edge_view_required_data: 17
edge_view_required_data: 16
edge_view_required_data: 18
edge_view_required_data: 24
edge_view_required_data: 20
edge_view_required_data: 20
edge_view_required_data: 22
edge_view_required_data: 20
edge_view_required_data:  19.4
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 6.652092629515526
second_moment_service_time_of_types: 
 0.00032013733955035025


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 26
edge_view_required_data: 19
edge_view_required_data: 26
edge_view_required_data: 24
edge_view_required_data: 27
edge_view_required_data: 33
edge_view_required_data: 28
edge_view_required_data: 29
edge_view_required_data: 25
edge_view_required_data: 18
edge_view_required_data:  25.5
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 6.652092629515526
second_moment_service_time_of_types: 
 0.00032013733955035025


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 37
edge_view_required_data: 24
edge_view_required_data: 31
edge_view_required_data: 29
edge_view_required_data: 35
edge_view_required_data: 35
edge_view_required_data: 31
edge_view_required_data: 31
edge_view_required_data: 29
edge_view_required_data: 35
edge_view_required_data:  31.7
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 6.652092629515526
second_moment_service_time_of_types: 
 0.00032013733955035025


data_size_of_types:  4.528162913162762
data_types_in_vehicle:  8
data_types_in_vehicle:  6
data_types_in_vehicle:  4
data_types_in_vehicle:  5
data_types_in_vehicle:  6
data_types_in_vehicle:  3
data_types_in_vehicle:  4
data_types_in_vehicle:  3
data_types_in_vehicle:  6
data_types_in_vehicle:  4
sum_of_data_types_in_vehicles:  4.9
edge_view_required_data: 40
edge_view_required_data: 34
edge_view_required_data: 36
edge_view_required_data: 32
edge_view_required_data: 43
edge_view_required_data: 42
edge_view_required_data: 38
edge_view_required_data: 35
edge_view_required_data: 38
edge_view_required_data: 38
edge_view_required_data:  37.6
sum_edge_view_in_edge_node:  1.43
mean_service_time_of_types: 
 6.652092629515526
second_moment_service_time_of_types: 
 0.00032013733955035025