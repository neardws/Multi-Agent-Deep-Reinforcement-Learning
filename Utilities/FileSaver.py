# -*- coding: utf-8 -*-
"""
# @Project: Hierarchical-Reinforcement-Learning
# @File : FileSaver.py
# @Author : Neardws
# @Time : 2021/8/10 1:14 上午
"""
import pickle


def save_obj(obj, name):
    """
    Saves given object as a pickle file
    :param obj:
    :param name:
    :return:
    """
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads a pickle file object
    :param name:
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)
