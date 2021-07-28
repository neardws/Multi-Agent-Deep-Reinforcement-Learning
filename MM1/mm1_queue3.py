import numpy as np

arrival_rate = 2
service_rate = 3


def get_average():

    size = 1000

    x = np.random.exponential(scale=arrival_rate, size=size)  # lam为λ size为k
    y = np.random.exponential(scale=service_rate, size=size)  # lam为λ size为k

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

    queue_length = np.zeros(int(np.array(stop).max() * 10 + 1))

    for index, value in enumerate(start):
        start_value = int(value * 10)
        stop_value = int(stop[index] * 10)
        for m in range(start_value, stop_value + 1):
            queue_length[m] += 1

    num = 0
    for i in range(len(queue_length)):
        num += queue_length[i]
    average = num / len(queue_length)
    return average


if __name__ == '__main__':
    sum = 0
    range_number = 100
    for i in range(range_number):
        average = get_average()
        sum += average
        print(average)

    new_average = sum / range_number
    print("*" * 32)
    l = (1 / arrival_rate) / (1 / service_rate - 1 / arrival_rate)
    print(l)
    print(new_average)
