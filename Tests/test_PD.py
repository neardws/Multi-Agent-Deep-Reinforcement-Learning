# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_PD.py
@Author  ：Neardws
@Date    ：7/29/21 3:30 下午 
"""
import pandas as pd


if __name__ == '__main__':
    df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
    df = df.append(pd.DataFrame([[5, 6], [7, 8]], columns=list('AB')), ignore_index=True)
    print(df)
    Longitude_min = 104.04565967220308  # 104.05089219802858
    Longitude_max = 104.07650822204591  # 104.082306230011
    Latitude_min = 30.654605745741608  # 30.64253859556557
    Latitude_max = 30.68394513007631  # 30.6684641634594
    time_start = 1479031200
    time_end = 1479031500
    # file = Extract(file_name="../CSV/gps_20161113",
    #                longitude_min=Longitude_min,
    #                longitude_max=Longitude_max,
    #                latitude_min=Latitude_min,
    #                latitude_max=Latitude_max,
    #                time_start=time_start,
    #                time_end=time_end,
    #                outfile="../CSV/step1_gps_20161113.csv")
    # print("Extract finished")
    # file2 = Tran("../CSV/step1_gps_20161113.csv", "../CSV/step2_gps_20161113.csv",
    #              time_start=time_start,
    #              longitude_min=Longitude_min,
    #              latitude_min=Latitude_min)
    # print("Tran finished")
    # file3 = Fill("../CSV/step2_gps_20161113.csv", "../CSV/vehicle.csv", time_length=300)
    # print("Fill finished")
    # # x=geodistance(108.9345752,34.214946,108.9345752,34.241936)
    # coordinate_transformation("../CSV/step3_gps_20161113.csv", Longitude_min, Latitude_min, "../CSV/vehicle.csv", time_start)
    # print("Finial finished")