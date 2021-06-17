import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(0.1, 0.9, 0.001)
yy = np.arange(0.01, 0.99, 0.001)
X, Y = np.meshgrid(xx, yy)
Z = X + Y

print(X)
print(Y)
print(Z)

for x, xz in enumerate(Z):
    for y, z in enumerate(xz):
        if z > 0.99:
            diff = z - 0.99
            Y[x][y] = Y[x][y] - diff


# Z = X + Y
# # print(X)
# # print(Y)
# print(Z)

M = 1 / X + X / (2 * (1 - X))
N = 1 / Y + (1 / (1 - X)) * (1 + (X + Y) / (2 * (1 - X - Y))) - 1

M_max = np.array(M).max()
M_min = np.array(M).min()

N_max = np.array(N).max()
N_min = np.array(N).min()

Z = (M - M_min) / (M_max - M_min) + (N - N_min) / (N_max - N_min)

# Z = 1 / X + X / (2 * (1 - X)) + 1 / Y + (1 / (1 - X)) * (1 + (X + Y) / (2 * (1 - X - Y))) - 1

# Z = 1 / Y + Y / (2 * (1 - Y)) + 1 / X + (1 / (1 - Y)) * (1 + (X + Y) / (2 * (1 - X - Y))) - 1
# Z = 1 / X + X / (2 * (1 - X))
# Z = 1 / Y + (1 / (1 - X)) * (1 + (X + Y) / (2 * (1 - X - Y))) - 1
print(Z)

num = [0, 0, 0, 0, 0, 0, 0]  # <0, 0-10, 10-50, 50-100, 100-150, 150-200, 200
Z_min = np.array(Z).min()
for x, zx in enumerate(Z):
    for y, z in enumerate(zx):
        # if z <= 0:
        #     num[0] += 1
        # elif 0 < z <= 10:
        #     num[1] += 1
        # elif 10 < z <= 50:
        #     num[2] += 1
        # elif 50 < z <= 100:
        #     num[3] += 1
        # elif 100 < z <= 150:
        #     num[4] += 1
        # elif 150 < z <= 200:
        #     num[5] += 1
        # elif z > 200:
        #     num[6] += 1

        if z == Z_min:
            num[0] += 1
            print("*"*83)
            xxxx = X[x][y]
            yyyy = Y[x][y]
            print(xxxx, yyyy)
            m = 1 / xxxx + xxxx / (2 * (1 - xxxx))
            n = 1 / yyyy + (1 / (1 - xxxx)) * (1 + (xxxx + yyyy) / (2 * (1 - xxxx - yyyy))) - 1
            print(m+n)
            print("*" * 83)
            xxxx = Y[x][y]
            yyyy = X[x][y]
            print(xxxx, yyyy)
            m = 1 / xxxx + xxxx / (2 * (1 - xxxx))
            n = 1 / yyyy + (1 / (1 - xxxx)) * (1 + (xxxx + yyyy) / (2 * (1 - xxxx - yyyy))) - 1
            print(m + n)
        elif 8.1 < z <= 8.2:
            num[1] += 1
        elif 8.3 < z <= 8.4:
            num[2] += 1
        elif 8.4 < z <= 8.5:
            num[3] += 1
        elif 9 < z <= 10:
            num[4] += 1
        elif 10 < z <= 11:
            num[5] += 1
        elif z > 6:
            num[6] += 1

print(np.array(Z).min())

print("*"* 83)
print(num)
#作图
ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax3.set_xlabel(r"$\lambda_1$")
ax3.set_ylabel(r"$\lambda_2$")
ax3.set_zlabel(r"$\Delta_{i 1}^{*} + \Delta_{i 2}^{*}$")
# ax3.contour(X, Y, Z, offset=-2, cmap='rainbow')  # 绘制等高线
# plt.show()
plt.savefig('example.jpg', dpi=300)