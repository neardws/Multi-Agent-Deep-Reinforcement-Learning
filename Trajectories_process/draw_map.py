# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：draw_map.py
@Author  ：Neardws
@Date    ：7/19/21 2:38 下午 
"""
import csv
import matplotlib.pyplot as plt

csvFile = open("/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle.csv")
f_read = csv.reader(csvFile)
row = list(f_read)
x = []
y = []
for elem in row[1:]:
    x_location = int(float(elem[3]))
    y_location = int(float(elem[4]))
    x.append(x_location)
    y.append(y_location)
csvFile.close()

plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(x, y)
plt.show()
