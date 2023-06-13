# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：save_file.py
@Author  ：Neardws
@Date    ：8/9/21 7:55 下午 
"""
import uuid
import os
import datetime
from file_name import project_dir

if __name__ == '__main__':

    # 年-月-日 时:分:秒
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 年-月-日
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    # 时:分:秒
    hourTime = datetime.datetime.now().strftime('%H:%M:%S')
    print(nowTime + '\n' + dayTime + hourTime + '\n' + hourTime)

    pwd = project_dir + '/Data/' + dayTime + '-' + hourTime
    print(pwd)
    # print(pwd)
    # 判断文件夹是否已存在
    if not os.path.exists(pwd):
        os.makedirs(pwd)

    uuid_str = uuid.uuid4().hex
    init_experiment_config_name = 'init_experiment_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    init_agent_config_name = 'init_agent_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    init_environment_name = 'init_environment_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_agent_config_name = 'temple_agent_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_agent_name = 'temple_agent_%s.pkl' % uuid_str

    print(init_experiment_config_name)
    print(init_agent_config_name)
    print(init_environment_name)
    print(temple_agent_config_name)
    print(temple_agent_name)
