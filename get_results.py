from Utilities.FileOperator import load_obj

def agents_run_results(agent_file_name, num_episodes, result_name):
    agent = load_obj(name=agent_file_name)
    agent.run_n_episodes_as_results(num_episodes, result_name)


if __name__ == '__main__':
    agent_file_name = "/home/neardws/Hierarchical-Reinforcement-Learning/Data/Data1018/2021-10-18-16-57-25/temple_agent_7dc9a740efc44bf88c3a0b8fce9393cb.pkl"
    num_episodes = 10
    result_name = "results.csv"
    agents_run_results(agent_file_name, num_episodes, result_name)