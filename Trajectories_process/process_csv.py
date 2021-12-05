import pandas as pd
import numpy as np

csv_file = '/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_22.csv'

df = pd.read_csv(csv_file)
x_max = df['longitude'].max()
y_max = df['latitude'].max()

vehicle_id = df['vehicle_id'].drop_duplicates()
vehicle_info_list = []
for i in vehicle_id:
    df_i = df[df['vehicle_id'] == i]
    longitude_diff = df_i['longitude'].max() - df_i['longitude'].min()
    latitude_diff = df_i['latitude'].max() - df_i['latitude'].min()
    distance = np.sqrt(longitude_diff ** 2 + latitude_diff ** 2)
    vehicle_info_list.append([i, distance])

frame = []
vehicle_info_list.sort(key=lambda x: x[1], reverse=True)
for index, vehicle_info in enumerate(vehicle_info_list):
    print('vehicle_id:', vehicle_info[0], 'distance:', vehicle_info[1])
    df_i = df[df['vehicle_id'] == vehicle_info[0]]
    frame.append(df_i)
    if index == 9:
        break

new_df = pd.concat(frame)
new_df.to_csv('/home/neardws/Hierarchical-Reinforcement-Learning/CSV/vehicle_1116_22_processed.csv', index=False)


