import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()  # 定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

start_value = 0.01
stop_value = 0.99
step = 0.01

#      定义三维数据
xx = np.arange(start_value, stop_value, step)
arr_xx = []
for x in xx:
    arr_xx.append(xx)
X = np.array(arr_xx)

arr_yy = []
for x in xx:
    y = start_value
    arr_y = []
    while (x + y) < 0.999:
        arr_y.append(y)
        y += step
    # last_y = arr_y[len(arr_y) - 1]
    #
    while len(arr_y) < len(xx):
        arr_y.append(0.01)
    arr_yy.append(np.array(arr_y))
Y = np.array(arr_yy)

print(X)
print(Y)
print(X.shape)
print(Y.shape)
# print(xx.shape)
# X, Y = np.meshgrid(xx, yy)
# print(Y.shape)
# print(X)
# print(X.shape)
# print("*"*32)
# print(Y)
#
# Z = 1 / X + X / (2 * (1 - X)) + 1 / Y + (1 / (1 - X)) * (1 + (X + Y) / (2 * (1 - X - Y))) - 1
# Z = 1 / X + X / (2 * (1 - X))
# Z = 1 / Y + (1 / (1 - X)) * (1 + (X + Y) / (2 * (1 - X - Y))) - 1

Z = X + Y
print(Z)
#
#
# 作图
ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# ax3.contour(X,Y,Z,offset=-2, cmap = 'rainbow')#绘制等高线
plt.show()
