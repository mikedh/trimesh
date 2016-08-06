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

from collections import deque
from copy import deepcopy
from trimesh.constants import tol

#python 3
try:
    from cStringIO import StringIO
    _PY3 = False
except ImportError:
    from io import StringIO
    from io import BytesIO
    _PY3 = True

dir_current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_models  = os.path.join(dir_current, '../models')
dir_2D      = os.path.join(dir_current, '../models/2D')
dir_data    = os.path.join(dir_current, 'data')

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

def io_wrap(item):
    if isinstance(item, str):
        return StringIO(item)
    if _PY3 and isinstance(item, bytes):
        return BytesIO(item)
    return item

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
    if trimesh.util.is_sequence(mesh):
        for i in mesh:
            i.metadata['file_name'] = str(file_name)
    else:
        mesh.metadata['file_name'] = str(file_name)
    return mesh
    
def get_meshes(count=None):
    ls = os.listdir(dir_models)
    if count is None:
        count = len(ls)
    meshes = deque()
    for file_name in ls:
        extension = os.path.splitext(file_name)[-1][1:].lower()
        if extension in trimesh.available_formats():
            loaded = get_mesh(file_name)
            if trimesh.util.is_instance_named(loaded, 'Trimesh'):
                meshes.append(loaded)
            else:
                log.error('file %s loaded garbage!', file_name)
        else:
            log.warning('%s has no loader, not running test on!', file_name)
            
        if len(meshes) >= count:
            break
    return list(meshes)
    
data = _load_data()
