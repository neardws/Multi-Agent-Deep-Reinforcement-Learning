# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：gps_to_xy.py
@Author  ：Neardws
@Date    ：7/19/21 10:26 上午 
"""
import csv
from math import asin
from math import cos
from math import fabs
from math import pi
from math import radians
from math import sin
from math import sqrt

import pandas as pd


# Extract，传入文件名，输出结果
# noinspection PyPep8Naming
def Extract(file_name, longitude_min, longitude_max, latitude_min, latitude_max, time_start, time_end, outfile):
    df = pd.read_csv(file_name, names=['vehicle_id', 'order_number', 'time', 'longitude', 'latitude'], index_col=0)
    # 经纬度定位
    df.drop(df.columns[[0]], axis=1, inplace=True)
    print(df)
    data = df[
        (df['longitude'] > longitude_min) & (df['longitude'] < longitude_max) & (df['latitude'] > latitude_min) & (
                df['latitude'] < latitude_max)]         # location
    # 时间戳定位
    data2 = data[((data['time'] > time_start) & (data['time'] < time_end))]
    # 排序
    data2.sort_values(by=['vehicle_id', 'time'], inplace=True)
    data2.to_csv(outfile)
    return outfile


# 坐标转化
# noinspection PyPep8Naming
def Tran(file_name, outfile):
    with open(file_name) as csv_file:
        f_read = csv.reader(csv_file)
        row = list(f_read)
        # 写入
        with open(outfile, 'a', newline='') as out_file:
            f_write = csv.writer(out_file)
            f_write.writerow(row[0])
            for item in row[1:]:
                item[2], item[3] = gcj02_to_wgs84(float(item[2]), float(item[3]))
                f_write.writerow(item)
    return outfile


# 填充
# noinspection PyPep8Naming
def Fill(file_name, outfile, time_start, time_end):
    with open(file_name) as csv_file:
        f_read = csv.reader(csv_file)
        row = list(f_read)
        # 写入
        with open(outfile, 'a', newline='') as out_file:
            f_write = csv.writer(out_file)
            old_row = row[0]
            f_write.writerow(old_row)
            for item in row[1:]:
                new_row = item
                # 车辆不同
                if new_row[0] != old_row[0]:
                    if old_row[1] == 'time':
                        if int(float(new_row[1])) > time_start:
                            old_row[0] = new_row[0]
                            old_row[2] = new_row[2]
                            old_row[3] = new_row[3]
                            time = time_start
                            while time < int(float(new_row[1])):
                                old_row[1] = time
                                f_write.writerow(old_row)
                                time += 1
                    else:
                        if int(float(old_row[1])) < time_end:
                            time = int(float(old_row[1])) + 1
                            while time < time_end:
                                old_row[1] = time
                                f_write.writerow(old_row)
                                time += 1
                        if int(float(new_row[1])) > time_start:
                            old_row[0] = new_row[0]
                            old_row[2] = new_row[2]
                            old_row[3] = new_row[3]
                            time = time_start
                            while time < int(float(new_row[1])):
                                old_row[1] = time
                                f_write.writerow(old_row)
                                time += 1
                    f_write.writerow(new_row)
                else:
                    line_number = int(float(new_row[1])) - int(float(old_row[1]))  # 新加的行数
                    for i in range(line_number):
                        old_row[1] = int(float(old_row[1])) + 1  # 时间+1s
                        old_row[2] = float(old_row[2]) + (float(old_row[2]) - float(new_row[2])) / float(line_number)
                        old_row[3] = float(old_row[3]) + (float(old_row[3]) - float(new_row[3])) / float(line_number)
                        f_write.writerow(old_row)
                old_row = new_row
    return outfile


# 经纬度转x,y坐标,米为单位
# noinspection PyPep8Naming
def coordinate_transformation(file_name, longitude_min, latitude_min, outfile, time_start):
    with open(file_name) as csv_file:
        f_read = csv.reader(csv_file)
        row = list(f_read)
        # 写入
        with open(outfile, 'a', newline='') as out_file:
            f_write = csv.writer(out_file)
            f_write.writerow(row[0])
            vehicle_index = -1
            old_vehicleID = "temp"
            for item in row[1:]:
                # 存储当前的经纬度信息
                lng = item[2]
                lat = item[3]
                item[2] = geodistance(longitude_min, latitude_min, lng, latitude_min)
                item[3] = geodistance(longitude_min, latitude_min, longitude_min, lat)
                if item[2] > 3000 or item[3] > 3000:
                    continue
                if item[0] != old_vehicleID:
                    vehicle_index += 1
                    old_vehicleID = item[0]
                item[0] = vehicle_index
                item[1] = int(float(item[1])) - time_start
                f_write.writerow(item)
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

    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)

    radlat = lat / 180.0 * pi
    magic = sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = sqrt(magic)

    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 *
            sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lat * pi) + 40.0 *
            sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(lat / 12.0 * pi) + 320 *
            sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * sqrt(fabs(lng))
    ret += (20.0 * sin(6.0 * lng * pi) + 20.0 *
            sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(lng * pi) + 40.0 *
            sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(lng / 12.0 * pi) + 300.0 *
            sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance * 1000


# if __name__ == "__main__":
#     Longitude_min = 104.04565967220308  # 104.05089219802858
#     Longitude_max = 104.07650822204591  # 104.082306230011
#     Latitude_min = 30.654605745741608  # 30.64253859556557
#     Latitude_max = 30.68394513007631  # 30.6684641634594
#     time_start = 1479031200
#     time_end = 1479031500
#     file = Extract("../CSV/gps_20161113", Longitude_min, Longitude_max,
#     Latitude_min, Latitude_max, time_start, time_end,
#                    "../CSV/step1_gps_20161113.csv")
#     print("Extract finished")
#     file2 = Tran(file, "../CSV/step2_gps_20161113.csv")
#     print("Tran finished")
#     file3 = Fill(file2, "../CSV/step3_gps_20161113.csv", time_start, time_end)
#     print("Fill finished")
#     # x=geodistance(108.9345752,34.214946,108.9345752,34.241936)
#     coordinate_transformation("../CSV/step3_gps_20161113.csv", Longitude_min, Latitude_min, "../CSV/vehicle.csv", time_start)
#     print("Finial finished")

# gps_20161110 训练集 10min
# gps_20161113 仿真集 5min
