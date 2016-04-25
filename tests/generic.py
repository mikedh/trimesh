import unittest
import inspect
import logging
import os
import json
import time

import numpy as np

import trimesh

dir_current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_models  = os.path.join(dir_current, '../models')
dir_data    = os.path.join(dir_current, 'data')

def _load_data():
    data = {}
    for file_name in os.listdir(dir_data):
        name, extension = os.path.splitext(file_name)
        if extension != '.json': 
            continue
        file_path = os.path.join(dir_data, file_name)
        with open(file_path, 'r') as file_obj:
            data[name] = json.load(file_obj)
            
    data['model_paths'] = [os.path.join(dir_models, f) for f in os.listdir(dir_models)]

    return data

data = _load_data()

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

