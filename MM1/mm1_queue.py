import numpy as np
from matplotlib import pyplot as plt

# Poisson分布
# Arrival rate 8
# Service rate 10

arrival_rate = 8
service_rate = 10

size = 1000

x = np.random.poisson(lam=arrival_rate, size=size)  # lam为λ size为k
y = np.random.poisson(lam=service_rate, size=size)  # lam为λ size为k

print(x)
print(y)

x = 1 / x
y = 1 / y

print(x)
print(y)


start = np.zeros(size)

value_sum = 0
for index, value in enumerate(x):
    value_sum += value
    start[index] = value_sum

waiting = np.zeros(size)

spend = start[0] + y[0]
for index, value in enumerate(start):
    if index == 0:
        waiting[0] = 0
        pass
    else:
        w = spend - value
        if w > 0:
            waiting[index] = w
            spend += y[index]
        else:
            waiting[index] = 0
            spend = y[index] + value

stop = start + waiting + y
#
# noinspection PyArgumentList
queue_length = np.zeros(int(np.array(stop).max() * 100 + 1))

for index, value in enumerate(start):
    start_value = int(value * 100)
    stop_value = int(stop[index] * 100)
    for m in range(start_value, stop_value + 1):
        queue_length[m] += 1
#
print("*"*32)
print(x)
print(start)
print(waiting)
print(y)
print(stop)
print(queue_length)
print("*"*32)
num = 0
for i in range(len(queue_length)):
    num += queue_length[i]
average = num / len(queue_length)
print(average)
colors = '#00CED1'  # 点的颜色
area = np.pi * 1**1  # 点面积
# noinspection PyArgumentList
y_index = range(int(np.array(stop).max() * 100 + 1))
plt.scatter(y_index, queue_length, s=area, c=colors, alpha=0.4)
plt.show()