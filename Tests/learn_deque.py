# -*- coding: utf-8 -*-
"""
# @Project: Hierarchical-Reinforcement-Learning
# @File : learn_deque.py
# @Author : Neardws
# @Time : 2021/8/11 3:00 下午
"""
from collections import deque


if __name__ == '__main__':
    memory = deque(maxlen=3)
    memory.append(1)
    print(memory)
    memory.append(2)
    print(memory)
    memory.append(3)
    print(memory)
    memory.append(4)
    print(memory)
