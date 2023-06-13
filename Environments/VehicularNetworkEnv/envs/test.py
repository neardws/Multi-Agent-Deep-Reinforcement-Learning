a = 0.3
b = 0.3
c = 0.3

max_reward = 0
max_i = 0
max_j = 0
max_k = 0

for i in range(100):
    for j in range(100 - i):
        for k in range(100 - i - j):
            reward = a * i * 0.01 + b * j * 0.01 + c * k * 0.01
            if reward > max_reward:
                max_reward = reward
                max_i = i
                max_j = j
                max_k = k

print("max_i: ", max_i, "max_j:", max_j, "max_k:", max_k) 

