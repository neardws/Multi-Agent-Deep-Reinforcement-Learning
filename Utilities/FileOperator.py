# -*- coding: utf-8 -*-
"""
# @Project: Hierarchical-Reinforcement-Learning
# @File : FileSaver.py
# @Author : Neardws
# @Time : 2021/8/10 1:14 上午
"""
import pickle
import uuid
import os
import datetime
from File_Name import project_dir, data


def save_obj(obj, name):
    """
    Saves given object as a pickle file
    :param obj:
    :param name:
    :return:
    """
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads a pickle file object
    :param name:
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def init_compare_file_name():
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    hourTime = datetime.datetime.now().strftime('%H-%M-%S')
    pwd = project_dir + data + dayTime + '-' + hourTime

    if not os.path.exists(pwd):
        os.makedirs(pwd)

    return pwd + '/'


def init_file_name():
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    hourTime = datetime.datetime.now().strftime('%H-%M-%S')
    pwd = project_dir + data + dayTime + '-' + hourTime

    if not os.path.exists(pwd):
        os.makedirs(pwd)

    list_file_name = project_dir + data + dayTime + '-' + hourTime + '-' + 'list_file_name.pkl'

    uuid_str = uuid.uuid4().hex
    init_experiment_config_name = pwd + '/' + 'init_experiment_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    init_agent_config_name = pwd + '/' + 'init_agent_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    init_environment_name = pwd + '/' + 'init_environment_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_agent_config_name = pwd + '/' + 'temple_agent_config_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_agent_name = pwd + '/' + 'temple_agent_%s.pkl' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_result_name = pwd + '/' + 'temple_result_%s.csv' % uuid_str

    uuid_str = uuid.uuid4().hex
    temple_loss_name = pwd + '/' + 'temple_loss_%s.csv' % uuid_str

    return {
        "list_file_name": list_file_name,
        "init_experiment_config_name": init_experiment_config_name,
        "init_agent_config_name": init_agent_config_name,
        "init_environment_name": init_environment_name,
        "temple_agent_config_name": temple_agent_config_name,
        "temple_agent_name": temple_agent_name,
        "temple_result_name": temple_result_name,
        "temple_loss_name": temple_loss_name
    }


def save_init_files(list_file_name, experiment_config, agent_config, vehicularNetworkEnv):
    if isinstance(list_file_name, list):
        save_obj(obj=list_file_name, name=list_file_name[0])
        save_obj(obj=experiment_config, name=list_file_name[1])
        save_obj(obj=agent_config, name=list_file_name[2])
        save_obj(obj=vehicularNetworkEnv, name=list_file_name[3])
        print("save init files successful")
    if isinstance(list_file_name, dict):
        save_obj(obj=list_file_name, name=list_file_name['list_file_name'])
        save_obj(obj=experiment_config, name=list_file_name['init_experiment_config_name'])
        save_obj(obj=agent_config, name=list_file_name['init_agent_config_name'])
        save_obj(obj=vehicularNetworkEnv, name=list_file_name['init_environment_name'])
        print("save init files successful")


def load_name(list_file_name_obj, name):

    if name == 'list_file_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[0]
    if name == 'init_experiment_config_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[1]
    if name == 'init_agent_config_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[2]
    if name == 'init_environment_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[3]
    if name == 'temple_agent_config_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[4]
    if name == 'temple_agent_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[5]
    if name == 'temple_result_name' and isinstance(list_file_name_obj, list):
        return list_file_name_obj[6]

    if name == 'list_file_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['list_file_name']
    if name == 'init_experiment_config_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['init_experiment_config_name']
    if name == 'init_agent_config_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['init_agent_config_name']
    if name == 'init_environment_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['init_environment_name']
    if name == 'temple_agent_config_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['temple_agent_config_name']
    if name == 'temple_agent_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['temple_agent_name']
    if name == 'temple_result_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['temple_result_name']
    if name == 'temple_loss_name' and isinstance(list_file_name_obj, dict):
        return list_file_name_obj['temple_loss_name']
