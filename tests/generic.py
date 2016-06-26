'''
Module which contains all the imports and data available to unit tests
'''

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
dir_2D      = os.path.join(dir_current, '../models/2D')
dir_data    = os.path.join(dir_current, 'data')

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

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
    data['2D_files']    = [os.path.join(dir_2D, f) for f in os.listdir(dir_2D)]
    return data

def get_mesh(file_name):
    mesh = trimesh.load(os.path.join(dir_models,
                                     file_name))
    return mesh
    
def get_meshes(count):
    meshes = [get_mesh(fn) for fn in [i for i in os.listdir(dir_models) if '.' in i][:count]]
    return meshes
    
data = _load_data()
