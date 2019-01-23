"""
Module which contains all the imports and data available to
unit tests to reduce the amount of boilerplate.
"""
import os
import sys
import json
import time
import shutil
import timeit
import inspect
import logging
import platform
import tempfile
import unittest
import itertools
import subprocess
import contextlib
import threading

try:
    # Python 3
    from http.server import SimpleHTTPRequestHandler
    import socketserver
except ImportError:
    # Python 2
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    import SocketServer as socketserver

import numpy as np

try:
    import sympy as sp
except ImportError:
    pass

import trimesh
import collections

from copy import deepcopy
from collections import deque
from trimesh.constants import tol, tol_path
from trimesh.base import Trimesh

# make sure everyone knows they should run additional
# validation checks and raise exceptions
trimesh.constants.tol.strict = True
trimesh.constants.tol_path.strict = True


try:
    from shapely.geometry import Point, Polygon, LineString
    has_path = True
except ImportError:
    has_path = False

# Python version as an array, i.e. [3, 6]
python_version = np.array([sys.version_info.major,
                           sys.version_info.minor])

# some repeatable homogenous transforms to use in tests
transforms = [trimesh.transformations.euler_matrix(np.pi / 4, i, 0)
              for i in np.linspace(0.0, np.pi * 2.0, 100)]
# should be a (100, 4, 4) float
transforms = np.array(transforms)

try:
    # do the imports for Python 2
    from cStringIO import StringIO
    _PY3 = False
except ImportError:
    # if that didn't work we're probably on Python 3
    from io import StringIO
    from io import BytesIO
    _PY3 = True

# are we on linux
is_linux = 'linux' in platform.system().lower()

# find the current absolute path using inspect
dir_current = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
# the absolute path for our reference models
dir_models = os.path.abspath(
    os.path.join(dir_current, '..', 'models'))
# the absolute path for our 2D reference models
dir_2D = os.path.abspath(
    os.path.join(dir_current, '..', 'models', '2D'))
# the absolute path for our test data and truth
dir_data = os.path.abspath(
    os.path.join(dir_current, 'data'))

# a logger for tests to call
log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

# turn strings / bytes into file- like objects
io_wrap = trimesh.util.wrap_as_stream


def random(*args, **kwargs):
    """
    A random function always seeded from the same value.

    Replaces: np.random.random(*args, **kwargs)
    """
    state = np.random.RandomState(seed=1)
    return state.random_sample(*args, **kwargs)


def _load_data():
    """
    Load the JSON files from our truth directory.
    """
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
    """
    Get a mesh from the models directory by name.

    Parameters
    -------------
    file_name : str
      Name of model in /models/
    *args : [str]
      Additional files to load

    Returns
    -----------
    meshes : trimesh.Trimesh or list
      Single mesh or list of meshes from args
    """
    meshes = collections.deque()
    for name in np.append(file_name, args):
        location = os.path.join(dir_models, name)
        log.info('loading mesh from: %s', location)
        meshes.append(trimesh.load(location, **kwargs))
    if len(meshes) == 1:
        return meshes[0]
    return list(meshes)


@contextlib.contextmanager
def serve_meshes():
    """
    This context manager serves meshes over HTTP at some
    available port.
    """
    class _ServerThread(threading.Thread):
        def run(self):
            os.chdir(dir_models)
            Handler = SimpleHTTPRequestHandler
            self.httpd = socketserver.TCPServer(('', 0), Handler)
            _, self.port = self.httpd.server_address
            self.httpd.serve_forever()

    t = _ServerThread()
    t.daemon = False
    t.start()
    time.sleep(0.2)
    yield 'http://localhost:{}'.format(t.port)
    t.httpd.shutdown()
    t.join()


def get_meshes(count=np.inf,
               raise_error=False,
               only_watertight=True):
    """
    Get meshes to test with.

    Parameters
    ----------
    count : int
      Approximate number of meshes you want
    raise_error : bool
      If True raise a ValueError if a mesh
      that should be loadable returns a non- Trimesh object.

    Returns
    ----------
    meshes : list
      Trimesh objects from models folder
    """
    # use deterministic file name order
    file_names = sorted(os.listdir(dir_models))

    meshes = []
    for file_name in file_names:
        extension = trimesh.util.split_extension(file_name).lower()
        if extension in trimesh.available_formats():
            loaded = trimesh.util.make_sequence(get_mesh(file_name))
            for m in loaded:
                is_mesh = trimesh.util.is_instance_named(m, 'Trimesh')
                is_scene = trimesh.util.is_instance_named(m, 'Scene')
                if raise_error and not is_mesh and not is_scene:
                    raise ValueError('%s returned a non- Trimesh object!',
                                     file_name)
                if not is_mesh or (only_watertight and not m.is_watertight):
                    continue
                meshes.append(m)
                yield m
        else:
            log.warning('%s has no loader, not running test on!',
                        file_name)

        if len(meshes) >= count:
            break


def get_2D(count=None):
    """
    Get Path2D objects to test with.

    Parameters
    --------------
    count : int
      Number of 2D drawings to return

    Yields
    --------------
    path : trimesh.path.Path2D
      Drawing from models folder
    """
    # if no path loading return empty list
    if not has_path:
        raise StopIteration

    # all files in the 2D models directory
    listdir = sorted(os.listdir(dir_2D))
    # if count isn't passed return all files
    if count is None:
        count = len(listdir)
    # save resulting loaded paths
    paths = []
    for file_name in listdir:
        # check to see if the file is loadable
        ext = trimesh.util.split_extension(file_name)
        if ext not in trimesh.available_formats():
            continue
        # full path
        location = os.path.join(dir_2D, file_name)
        try:
            paths.append(trimesh.load(location))
        except BaseException as E:
            log.error('failed on: {}'.format(file_name),
                      exc_info=True)
            raise E

        yield paths[-1]

        # if we don't need every path break
        if len(paths) >= count:
            break


# all the JSON files with truth data
data = _load_data()

# find executables to run with subprocess
from distutils.spawn import find_executable
# formats supported by meshlab for export tests
if any(find_executable(i) is None
       for i in ['xfvb-run', 'meshlabserver']):
    meshlab_formats = []
else:
    meshlab_formats = ['3ds', 'ply', 'stl', 'obj', 'qobj', 'off', 'ptx', 'vmi',
                       'bre', 'dae', 'ctm', 'pts', 'apts', 'xyz', 'gts', 'pdb',
                       'tri', 'asc', 'x3d', 'x3dv', 'wrl']
