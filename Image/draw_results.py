# -*- coding: utf-8 -*-
"""
# @Project: Hierarchical-Reinforcement-Learning
# @File : draw_results.py
# @Author : Neardws
# @Time : 2021/8/11 2:20 下午
"""
from file_name import project_dir
from Utilities.FileOperator import load_obj, load_name
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)


def draw_results(given_list_file_name):
    correct_list_file_name = project_dir + '/Data/' + given_list_file_name
    list_file = load_obj(name=correct_list_file_name)
    csv_file_name = load_name(list_file, 'temple_result_name')
    df = pd.read_csv(csv_file_name, names=["Epoch index", "Total reward", "Time taken"], header=0)
    epoch_index = df["Epoch index"].values.tolist()
    rewards = df["Total reward"].values.tolist()
    plt.plot(epoch_index, rewards, 'o-', color='b')  # s-:方形
    # plt.plot(x, k2, 'o-', color='g', label="CNN-RLSTM")  # o-:圆形
    # plt.xlabel("region length")  # 横坐标名字
    # plt.ylabel("accuracy")  # 纵坐标名字
    # plt.legend(loc="best")  # 图例
    plt.savefig('results.png')
    plt.show()


if __name__ == '__main__':
    draw_results(given_list_file_name='2021-08-21-05-56-04-list_file_name.pkl')
    draw_results(given_list_file_name='2021-08-28-09-11-22-list_file_name.pkl')
    draw_results(given_list_file_name='2021-08-29-23-03-17-list_file_name.pkl')
    # draw_results(given_list_file_name='2021-08-30-02-18-29-list_file_name.pkl')

