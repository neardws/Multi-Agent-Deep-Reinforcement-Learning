# Hierarchical-Reinforcement-Learning
Hierarchical Reinforcement Learning to solve Minimizing the Average Age of View (MAAV) Problem

![](https://img.shields.io/github/issues/neardws/Hierarchical-Reinforcement-Learning)
![](https://img.shields.io/github/forks/neardws/Hierarchical-Reinforcement-Learning)
![](https://img.shields.io/github/stars/neardws/Hierarchical-Reinforcement-Learning)
![](https://img.shields.io/github/license/neardws/Hierarchical-Reinforcement-Learning)

### TODO

- [ ] Unit testing
  - [ ] bug found on global trajectory process with zero values, and view required data maybe all zero in one view which means it may require no data

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