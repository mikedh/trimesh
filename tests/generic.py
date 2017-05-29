'''
Module which contains all the imports and data available to unit tests
'''

import unittest
import itertools
import inspect
import logging
import os
import sys
import json
import time

import numpy as np
import trimesh
import collections

from collections import deque
from copy import deepcopy
from trimesh.constants import tol, tol_path
from trimesh.base import Trimesh

from shapely.geometry import Point, Polygon

# python 3
try:
    from cStringIO import StringIO
    _PY3 = False
except ImportError:
    from io import StringIO
    from io import BytesIO
    _PY3 = True

dir_current = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
dir_models = os.path.abspath(os.path.join(dir_current, '..', 'models'))
dir_2D = os.path.abspath(os.path.join(dir_current, '..', 'models', '2D'))
dir_data = os.path.abspath(os.path.join(dir_current, 'data'))

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

'''
# block will print who is importing us
for i in inspect.stack():
    if i.code_context is None:
        continue
    if any('import generic' in j for j in i.code_context if j is not None):
        file_name = os.path.split(i.filename)[-1]
        print('\n\nRunning tests contained in: {}'.format(file_name))
        break
'''

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

    data['model_paths'] = [os.path.join(dir_models, f)
                           for f in os.listdir(dir_models)]
    data['2D_files'] = [os.path.join(dir_2D, f) for f in os.listdir(dir_2D)]
    return data


def get_mesh(file_name, *args, **kwargs):
    meshes = collections.deque()
    for name in np.append(file_name, args):
        location = os.path.join(dir_models, name)
        log.info('loading mesh from: %s', location)
        meshes.append(trimesh.load(location, **kwargs))
    if len(meshes) == 1:
        return meshes[0]
    return list(meshes)


def get_meshes(count=np.inf,
               raise_error=False,
               only_watertight=True):
    '''
    Get a list of meshes to test with.

    Arguments
    ----------
    count: int, approximate number of meshes you want
    raise_error: bool, if True raise a ValueError if a mesh
                 that should be loadable returns a non- Trimesh object.

    Returns
    ----------
    meshes: list, of Trimesh objects
    '''

    # randomly order the directory so if tests are running on a small
    # number of meshes for test speed we sample the whole test set eventually.
    file_names = np.random.permutation(os.listdir(dir_models))

    meshes = deque()
    for file_name in file_names:
        extension = trimesh.util.split_extension(file_name).lower()
        if extension in trimesh.available_formats():
            loaded = trimesh.util.make_sequence(get_mesh(file_name))
            for i in loaded:
                is_mesh  = trimesh.util.is_instance_named(i, 'Trimesh')
                is_scene = trimesh.util.is_instance_named(i, 'Scene')
                if raise_error and not is_mesh and not is_scene:
                    raise ValueError('%s returned a non- Trimesh object!', 
                                     file_name)
                if not is_mesh or (only_watertight and not i.is_watertight):
                    continue
                meshes.append(i)
        else:
            log.warning('%s has no loader, not running test on!',
                        file_name)
        if len(meshes) >= count:
            break
    return list(meshes)


def get_2D(count=None):
    ls = os.listdir(dir_2D)
    if count is None:
        count = len(ls)
    paths = deque()
    for file_name in ls:
        location = os.path.join(dir_2D, file_name)
        print(location)
        try:
            paths.append(trimesh.load(location))
        except:
            log.warning('skipping path load', exc_info=True)
        if len(paths) >= count:
            break
    return list(paths)


data = _load_data()
