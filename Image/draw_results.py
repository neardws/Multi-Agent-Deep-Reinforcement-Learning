# -*- coding: utf-8 -*-
"""
# @Project: Hierarchical-Reinforcement-Learning
# @File : draw_results.py
# @Author : Neardws
# @Time : 2021/8/11 2:20 下午
"""
from file_name import project_dir
from Utilities.FileSaver import load_obj
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)


def draw_results(given_list_file_name):
    correct_list_file_name = project_dir + '/Data/' + given_list_file_name
    list_file = load_obj(name=correct_list_file_name)
    csv_file_name = list_file[6]
    df = pd.read_csv(csv_file_name, names=["Epoch index","Total reward","Time taken"], header=0)
    epoch_index = df["Epoch index"].values.tolist()[0:200]
    rewards = df["Total reward"].values.tolist()[0:200]
    plt.plot(epoch_index, rewards, 's-', color='r', label="ATT-RLSTM")  # s-:方形
    # plt.plot(x, k2, 'o-', color='g', label="CNN-RLSTM")  # o-:圆形
    # plt.xlabel("region length")  # 横坐标名字
    # plt.ylabel("accuracy")  # 纵坐标名字
    # plt.legend(loc="best")  # 图例
    plt.show()

if __name__ == '__main__':
    draw_results(given_list_file_name='2021-08-10-10-42-50-list_file_name.pkl')
