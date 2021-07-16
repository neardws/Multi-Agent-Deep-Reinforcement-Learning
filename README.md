# Hierarchical-Reinforcement-Learning
Hierarchical Reinforcement Learning to solve Minimizing the Average Age of View (MAAV) Problem

### TODO

- [ ] Work on Agents

- [ ] Work on Configuration
    - [ ] Reorganize parameters in config according to HMAIMD_Agent
- [ ] Work on VehicularNetworkEnv
    - [ ] Service time add into sensor node state
    - [ ] Trajectories related, such as define global trajectory within the experiment, and add trajectory predicted time in experiment parameters   
- [ ] Work on Replay Buffer
    - [ ] Compass experiences to save GPU RAM
- [ ] Code review  

### Done ✓

- [x] Update the readme.md 2021-07-03
- [x] Renew Configuration 2021-07-04
- [x] NN Input and Output Dimensions 2021-07-05
- [x] Initialization of Actor Network and Critic Network for sensor nodes, edge node, and reward function, respectively 2021-07-06
- [x] Workflow of HMAIMD_Agent 2021-07-06
- [x] Read DDPG in detail 2021-07-06
- [x] Transplant implementation code from DDPG 2021-07-09
- [x] Create NN functions for actor and critic networks 2021-07-11
- [x] Renew buffer input and output 2021-07-12 
- [x] Save NN input and output tensor into self.parameters 2021-07-12
- [x] Init and update NN input and output 2021-07-13
- [x] Normalization the input of NN 2021-07-13
- [x] combined_action function realization 2021-07-13
- [x] Transfer NN output to action 2021-07-14
- [x] step in VehicularNetworkEnv 2021-07-15