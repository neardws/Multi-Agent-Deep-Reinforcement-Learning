# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：gps_to_xy.py
@Author  ：Neardws
@Date    ：7/19/21 10:26 上午 
"""
import time
import datetime
from math import asin
from math import cos
from math import fabs
from math import pi
from math import radians
from math import sin
from math import sqrt
import pandas as pd
import csv
from File_Name import project_dir


def out_put_long_lat(file_name, longitude_min, longitude_max, latitude_min, latitude_max, time_start, time_end, outfile):
    df = pd.read_csv(file_name, names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], header=0)
    # 经纬度定位
    df.drop(df.columns[[1]], axis=1, inplace=True)
    df.dropna(axis=0)

    df = df[
        (df['longitude'] > longitude_min) & (df['longitude'] < longitude_max) & (df['latitude'] > latitude_min) & (
                df['latitude'] < latitude_max) & (df['time'] > time_start) & (df['time'] < time_end)]  # location
    # 排序
    df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)

    print("sorted")

    list_longitude = []
    list_latitude = []
    for index, row in df.iterrows():

        row = dict(df.iloc[index])
        longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
        list_longitude.append(longitude)
        list_latitude.append(latitude)

    df = pd.DataFrame({'longitude': list_longitude, 'latitude': list_latitude})

    print("transformed")
    df.to_csv(outfile, index = False)


def main(file_name, longitude_min, longitude_max, latitude_min, latitude_max, time_start, time_end, outfile, output_no_fill_file_name):
    df = pd.read_csv(file_name, names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], header=0)
    # 经纬度定位
    df.drop(df.columns[[1]], axis=1, inplace=True)
    df.dropna(axis=0)

    df = df[
        (df['longitude'] > longitude_min) & (df['longitude'] < longitude_max) & (df['latitude'] > latitude_min) & (
                df['latitude'] < latitude_max) & (df['time'] > time_start) & (df['time'] < time_end)]  # location
    # 排序
    df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)

    print("sorted")

    new_longitude_min, new_latitude_min = gcj02_to_wgs84(longitude_min, latitude_min)
    vehicle_number = 0
    old_vehicle_id = None
    for index, row in df.iterrows():

        row = dict(df.iloc[index])
        vehicle_id = row['vehicle_id']

        if old_vehicle_id:
            if vehicle_id == old_vehicle_id:
                row['vehicle_id'] = vehicle_number
                longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = get_distance(new_longitude_min, new_latitude_min, longitude, new_latitude_min)
                y = get_distance(new_longitude_min, new_latitude_min, new_longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)
            else:
                vehicle_number += 1
                row['vehicle_id'] = vehicle_number
                longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = get_distance(new_longitude_min, new_latitude_min, longitude, new_latitude_min)
                y = get_distance(new_longitude_min, new_latitude_min, new_longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)
        else:
            row['vehicle_id'] = vehicle_number
            longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
            row['time'] = row['time'] - time_start
            x = get_distance(new_longitude_min, new_latitude_min, longitude, new_latitude_min)
            y = get_distance(new_longitude_min, new_latitude_min, new_longitude_min, latitude)
            row['longitude'] = x
            row['latitude'] = y
            df.iloc[index] = pd.Series(row)

        old_vehicle_id = vehicle_id
    
    print("transformed")
    df.to_csv(output_no_fill_file_name)

    old_row = None
    for index, row in df.iterrows():
        new_row = dict(df.iloc[index])
        if old_row:
            if old_row['vehicle_id'] == new_row['vehicle_id']:
                add_number = int(new_row['time']) - int(old_row['time']) - 1
                if add_number > 0:
                    add_longitude = (float(new_row['longitude']) - float(old_row['longitude'])) / float(add_number)
                    add_latitude = (float(new_row['latitude']) - float(old_row['latitude'])) / float(add_number)
                    for time_index in range(add_number):
                        df = df.append(pd.DataFrame({'vehicle_id': [old_row['vehicle_id']],
                                                     'time': [old_row['time'] + time_index + 1],
                                                     'longitude': [old_row['longitude'] + (time_index + 1) * add_longitude],
                                                     'latitude': [old_row['latitude'] + (time_index + 1) * add_latitude]}),
                                       ignore_index=True)
            else:
                if old_row['time'] < time_end - time_start:
                    for time_index in range(time_end - time_start - int(old_row['time']) - 1):
                        df = df.append(pd.DataFrame({'vehicle_id': [old_row['vehicle_id']],
                                                     'time': [old_row['time'] + time_index + 1],
                                                     'longitude': [old_row['longitude']],
                                                     'latitude': [old_row['latitude']]}),
                                       ignore_index=True)
                if new_row['time'] > 0:
                    for time_index in range(int(new_row['time'])):
                        df = df.append(pd.DataFrame({'vehicle_id': [new_row['vehicle_id']],
                                                     'time': [time_index],
                                                     'longitude': [new_row['longitude']],
                                                     'latitude': [new_row['latitude']]}),
                                       ignore_index=True)
            old_row = new_row
        else:
            if new_row['time'] > 0:
                for time_index in range(int(new_row['time'])):
                    df = df.append(pd.DataFrame({'vehicle_id': [new_row['vehicle_id']],
                                                 'time': [time_index],
                                                 'longitude': [new_row['longitude']],
                                                 'latitude': [new_row['latitude']]}),
                                   ignore_index=True)
            old_row = new_row
    df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
    print("added")
    df.to_csv(outfile)


def extract(file_name, longitude_min, longitude_max, latitude_min, latitude_max, time_start, time_end, outfile):
    df = pd.read_csv(file_name, names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], header=0)
    # 经纬度定位
    df.drop(df.columns[[1]], axis=1, inplace=True)
    df.dropna(axis=0)

    data = df[
        (df['longitude'] > longitude_min) & (df['longitude'] < longitude_max) & (df['latitude'] > latitude_min) & (
                df['latitude'] < latitude_max) & (df['time'] > time_start) & (df['time'] < time_end)]  # location
    # 排序
    data.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
    data.to_csv(outfile)
    print(data)
    return outfile


def tran(file_name, outfile, time_start, longitude_min, latitude_min):
    df = pd.read_csv(file_name, names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)
    new_longitude_min, new_latitude_min = gcj02_to_wgs84(longitude_min, latitude_min)
    vehicle_number = 0
    old_vehicle_id = None
    for index, row in df.iterrows():

        row = dict(df.iloc[index])
        vehicle_id = row['vehicle_id']

        if old_vehicle_id:
            if vehicle_id == old_vehicle_id:
                row['vehicle_id'] = vehicle_number
                longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = get_distance(new_longitude_min, new_latitude_min, longitude, new_latitude_min)
                y = get_distance(new_longitude_min, new_latitude_min, new_longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)
            else:
                vehicle_number += 1
                row['vehicle_id'] = vehicle_number
                longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
                row['time'] = row['time'] - time_start
                x = get_distance(new_longitude_min, new_latitude_min, longitude, new_latitude_min)
                y = get_distance(new_longitude_min, new_latitude_min, new_longitude_min, latitude)
                row['longitude'] = x
                row['latitude'] = y
                df.iloc[index] = pd.Series(row)
        else:
            row['vehicle_id'] = vehicle_number
            longitude, latitude = gcj02_to_wgs84(float(row['longitude']), float(row['latitude']))
            row['time'] = row['time'] - time_start
            x = get_distance(new_longitude_min, new_latitude_min, longitude, new_latitude_min)
            y = get_distance(new_longitude_min, new_latitude_min, new_longitude_min, latitude)
            row['longitude'] = x
            row['latitude'] = y
            df.iloc[index] = pd.Series(row)

        old_vehicle_id = vehicle_id

    df.to_csv(outfile)
    return outfile


def fill(file_name, outfile, time_length):
    df = pd.read_csv(file_name, names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)
    old_row = None
    for index, row in df.iterrows():
        new_row = dict(df.iloc[index])
        if old_row:
            if old_row['vehicle_id'] == new_row['vehicle_id']:
                add_number = int(new_row['time']) - int(old_row['time']) - 1
                if add_number > 0:
                    add_longitude = (float(new_row['longitude']) - float(old_row['longitude'])) / float(add_number)
                    add_latitude = (float(new_row['latitude']) - float(old_row['latitude'])) / float(add_number)
                    for time_index in range(add_number):
                        df = df.append(pd.DataFrame({'vehicle_id': [old_row['vehicle_id']],
                                                     'time': [old_row['time'] + time_index + 1],
                                                     'longitude': [old_row['longitude'] + (time_index + 1) * add_longitude],
                                                     'latitude': [old_row['latitude'] + (time_index + 1) * add_latitude]}),
                                       ignore_index=True)
            else:
                if old_row['time'] < time_end - time_start:
                    for time_index in range(time_length - int(old_row['time']) - 1):
                        df = df.append(pd.DataFrame({'vehicle_id': [old_row['vehicle_id']],
                                                     'time': [old_row['time'] + time_index + 1],
                                                     'longitude': [old_row['longitude']],
                                                     'latitude': [old_row['latitude']]}),
                                       ignore_index=True)
                if new_row['time'] > 0:
                    for time_index in range(int(new_row['time'])):
                        df = df.append(pd.DataFrame({'vehicle_id': [new_row['vehicle_id']],
                                                     'time': [time_index],
                                                     'longitude': [new_row['longitude']],
                                                     'latitude': [new_row['latitude']]}),
                                       ignore_index=True)
            old_row = new_row
        else:
            if new_row['time'] > 0:
                for time_index in range(int(new_row['time'])):
                    df = df.append(pd.DataFrame({'vehicle_id': [new_row['vehicle_id']],
                                                 'time': [time_index],
                                                 'longitude': [new_row['longitude']],
                                                 'latitude': [new_row['latitude']]}),
                                   ignore_index=True)
            old_row = new_row
    df.sort_values(by=['vehicle_id', 'time'], inplace=True, ignore_index=True)
    df.to_csv(outfile)
    return outfile


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323

    d_lat = trans_form_of_lat(lng - 105.0, lat - 35.0)
    d_lng = trans_form_of_lon(lng - 105.0, lat - 35.0)

    rad_lat = lat / 180.0 * pi
    magic = sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrt_magic = sqrt(magic)

    d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * pi)
    d_lng = (d_lng * 180.0) / (a / sqrt_magic * cos(rad_lat) * pi)
    mg_lat = lat + d_lat
    mg_lng = lng + d_lng
    return [lng * 2 - mg_lng, lat * 2 - mg_lat]


def trans_form_of_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 *
            sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lat * pi) + 40.0 *
            sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(lat / 12.0 * pi) + 320 *
            sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def trans_form_of_lon(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 *
            sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lng * pi) + 40.0 *
            sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(lng / 12.0 * pi) + 300.0 *
            sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def get_distance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    d_lon = lng2 - lng1
    d_lat = lat2 - lat1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance * 1000


# Scenario 1
# Longitude_min = 104.04565967220308  
# Longitude_max = 104.07650822204591  
# Latitude_min = 30.654605745741608  
# Latitude_max = 30.68394513007631  

# Scenario 2 (Xian)
# Longitude_min = 108.93445967220308
# Longitude_max = 108.96780822204591
# Latitude_min = 34.22454605745741
# Latitude_max = 34.25274513007631

if __name__ == "__main__":

    # Scenario 1
    Longitude_min = 104.04565967220308  
    Longitude_max = 104.07650822204591  
    Latitude_min = 30.654605745741608  
    Latitude_max = 30.68394513007631  

    # Scenario 2 (Xian)
    # Longitude_min = 108.93445967220308
    # Longitude_max = 108.96780822204591
    # Latitude_min = 34.22454605745741
    # Latitude_max = 34.25274513007631

    time_style = "%Y-%m-%d %H:%M:%S"
    # time_start_array = time.strptime("2016-11-16 18:00:00", time_style)
    # time_end_array = time.strptime("2016-11-16 18:05:00", time_style)
    # time_start = int(time.mktime(time_start_array))
    # time_end = int(time.mktime(time_end_array))

    # input_file_name = project_dir + "/CSV/gps_20161116"
    # output_file_name = project_dir + "/CSV/scenario/vehicle_1116_18.csv"
    # output_no_fill_file_name = project_dir + "/CSV/scenario/vehicle_1116_18_nofill.csv"
    
    # main(input_file_name, Longitude_min, Longitude_max, Latitude_min, Latitude_max, time_start, time_end,
    #     output_file_name, output_no_fill_file_name)

    # out_put_long_lat(input_file_name, Longitude_min, Longitude_max, Latitude_min, Latitude_max, time_start, time_end,
    #     output_file_name)



    time_start_array = time.strptime("2016-11-16 18:00:00", time_style)
    time_end_array = time.strptime("2016-11-16 18:05:00", time_style)
    time_start = int(time.mktime(time_start_array))
    time_end = int(time.mktime(time_end_array))
    # print(int(time.mktime(time_start_array)))
    # print(time.strftime(time_style, time.localtime(time_start)))
    input_file_name = project_dir + "/CSV/gps_20161116"
    step1_file_name = project_dir + "/CSV/step1_gps_20161116.csv"
    step2_file_name = project_dir + "/CSV/step2_gps_20161116.csv"
    step3_file_name = project_dir + "/CSV/step3_gps_20161116.csv"
    output_file_name = project_dir + "/CSV/scenario/vehicle_1116_18_lon_lat.csv"
    # output_no_fill_file_name = project_dir + "/CSV/scenario/vehicle_1127_08_lon_lat.csv"

    # main(input_file_name, Longitude_min, Longitude_max, Latitude_min, Latitude_max, time_start, time_end,
    #     output_file_name, output_no_fill_file_name)

    out_put_long_lat(input_file_name, Longitude_min, Longitude_max, Latitude_min, Latitude_max, time_start, time_end,
        output_file_name)

    # time_start_array = time.strptime("2016-11-16 23:00:00", time_style)
    # time_end_array = time.strptime("2016-11-16 23:05:00", time_style)
    # time_start = int(time.mktime(time_start_array))
    # time_end = int(time.mktime(time_end_array))

    # output_file_name = project_dir + "/CSV/scenario/vehicle_1116_23_lon_lat.csv"

    # out_put_long_lat(input_file_name, Longitude_min, Longitude_max, Latitude_min, Latitude_max, time_start, time_end,
    #     output_file_name)    
    
    # # print("Extract finished")
    # # file2 = tran(file, step2_file_name, time_start, Longitude_min, Latitude_min)
    # # print("Tran finished")
    # # file3 = fill(file2, step3_file_name, time_end - time_start)
    # # print("Fill finished")
    # # x=geodistance(108.9345752,34.214946,108.9345752,34.241936)
    # # coordinate_transformation(step3_file_name, Longitude_min, Latitude_min, output_file_name, time_start)
    # print("Finial finished")


    # df = pd.read_csv(input_file_name, names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], header=0)
    # # 经纬度定位
    # print(df.head(10))
    # max_longtiude = df['longitude'].max()
    # min_longtiude = df['longitude'].min()
    # max_latitude = df['latitude'].max()
    # min_latitude = df['latitude'].min()
    # print(max_longtiude)
    # print(min_longtiude)
    # print(max_latitude)
    # print(min_latitude)