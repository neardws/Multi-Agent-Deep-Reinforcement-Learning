from cmath import nan
import pandas as pd
import numpy as np
from File_Name import project_dir

def output_characteristics(trajectories_file_name, trajectories_file_name_with_no_fill):
    adt = 0.0  # average dwell time (s) of vehicles
    adt_std = 0.0  # standard deviation of dwell time (s) of vehicles
    anv = 0.0  # average number of vehicles in each second
    anv_std = 0.0  # standard deviation of number of vehicles in each second
    asv = 0.0  # average speed (m/s) of vehicles
    asv_std = 0.0  # standard deviation of speed (m/s) of vehicles


    df = pd.read_csv(trajectories_file_name, names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)

    vehicle_ids = df['vehicle_id'].unique()
    
    number_of_vehicles_in_seconds = np.zeros(300)
    vehicle_dwell_times = []
    for vehicle_id in vehicle_ids:
        new_df = df[df['vehicle_id'] == vehicle_id]
        vehicle_dwell_time = 0.0
        for row in new_df.itertuples():
            time = getattr(row, 'time')
            x = getattr(row, 'longitude')
            y = getattr(row, 'latitude')
            distance = np.sqrt((x - 1500) ** 2 + (y - 1500) ** 2)
            if distance <= 1500:
                vehicle_dwell_time += 1.0
                number_of_vehicles_in_seconds[int(time)] += 1.0
        vehicle_dwell_times.append(vehicle_dwell_time)

    assert len(vehicle_dwell_times) == len(vehicle_ids)
    print("vehicle_number: ", len(vehicle_ids))
    adt = np.mean(vehicle_dwell_times)
    adt_std = np.std(vehicle_dwell_times)

    anv = np.mean(number_of_vehicles_in_seconds)
    anv_std = np.std(number_of_vehicles_in_seconds)

    # print("Dwell time: ", vehicle_dwell_times)
    print("Average dwell time (s):", adt)
    print("Standard deviation of dwell time (s):", adt_std)
    # print("Number of vehicles in each second:", number_of_vehicles_in_seconds)
    print("Average number of vehicles in each second:", anv)
    print("Standard deviation of number of vehicles in each second:", anv_std)

    vehicle_speeds = []
    df = pd.read_csv(trajectories_file_name_with_no_fill, names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)
    vehicle_ids = df['vehicle_id'].unique()
    for vehicle_id in vehicle_ids:
        vehicle_speed = []
        new_df = df[df['vehicle_id'] == vehicle_id]
        vehicle_dwell_time = 0.0
        last_time = -1.0
        last_x = 0.0
        last_y = 0.0
        for row in new_df.itertuples():
            if int(last_time) == -1:
                last_time = getattr(row, 'time')
                last_x = getattr(row, 'longitude')
                last_y = getattr(row, 'latitude')
                continue
            time = getattr(row, 'time')
            x = getattr(row, 'longitude')
            y = getattr(row, 'latitude')
            distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
            speed = distance / (time - last_time)
            if not np.isnan(speed):
                vehicle_speed.append(speed)
            last_time = time
            last_x = x
            last_y = y
        if vehicle_speed != []:
            average_vehicle_speed = np.mean(vehicle_speed)
            vehicle_speeds.append(average_vehicle_speed)
    
    # print(vehicle_speeds)
    # print(type(vehicle_speeds))
    asv = np.mean(vehicle_speeds)
    asv_std = np.std(vehicle_speeds)
    # print("Vehicle speed:", vehicle_speeds)
    print("Average speed (m/s):", asv)
    print("Standard deviation of speed (m/s):", asv_std)


if __name__ == "__main__":
    print("scenario_1")
    trajectories_file_name = project_dir + "/CSV/scenario/vehicle_1116_08.csv"
    trajectories_file_name_with_no_fill = project_dir + "/CSV/scenario/vehicle_1116_08_nofill.csv"
    output_characteristics(trajectories_file_name, trajectories_file_name_with_no_fill)

    print("scenario_2")
    trajectories_file_name = project_dir + "/CSV/scenario/vehicle_1116_13.csv"
    trajectories_file_name_with_no_fill = project_dir + "/CSV/scenario/vehicle_1116_13_nofill.csv"
    output_characteristics(trajectories_file_name, trajectories_file_name_with_no_fill)

    print("scenario_3")
    trajectories_file_name = project_dir + "/CSV/scenario/vehicle_1116_18.csv"
    trajectories_file_name_with_no_fill = project_dir + "/CSV/scenario/vehicle_1116_18_nofill.csv"
    output_characteristics(trajectories_file_name, trajectories_file_name_with_no_fill)

