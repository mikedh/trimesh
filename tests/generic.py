# flake8: noqa
"""
Module which contains most imports and data unit tests
might need, to reduce the amount of boilerplate.
"""
import os
import sys
import json
import copy
import time
import shutil
import timeit
import base64
import inspect
import logging
import platform
import warnings
import tempfile
import unittest
import threading
import itertools
import contextlib
import subprocess
import collections
import numpy as np

import trimesh

from trimesh.base import Trimesh
from trimesh.constants import tol, tol_path
from collections import deque
from copy import deepcopy


if sys.version_info >= (3, 1):
    # Python 3
    from http.server import SimpleHTTPRequestHandler
    import socketserver

    from pyinstrument import Profiler

else:
    # Python 2
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    import SocketServer as socketserver

    # make a dummy profiler
    class Profiler(object):
        def __enter__(*args, **kwargs):
            pass

        def __exit__(*args, **kwargs):
            pass


# should we require all soft dependencies
# this is set in the docker images to catch missing packages
all_dep = 'alldep' in ''.join(sys.argv)

if all_dep:
    # make sure pyembree is importable
    from pyembree import rtcore_scene

try:
    import sympy as sp
except ImportError as E:
    if all_dep:
        raise E
    sp = None

try:
    import jsonschema
except BaseException as E:
    jsonschema = trimesh.exceptions.ExceptionModule(E)

# make sure functions know they should run additional
# potentially slow validation checks and raise exceptions
trimesh.util._STRICT = True
trimesh.constants.tol.strict = True
trimesh.constants.tol_path.strict = True


try:
    from mapbox_earcut import triangulate_float64
    has_earcut = True
except BaseException as E:
    if all_dep:
        raise E
    else:
        has_earcut = False

try:
    from shapely.geometry import Point, Polygon, LineString
    has_path = True
except ImportError as E:
    if all_dep:
        raise E
    has_path = False

try:
    from scipy import spatial, sparse
except BaseException as E:
    if all_dep:
        raise E

try:
    import networkx as nx
except BaseException as E:
    pass

# find_executable for binvox
has_binvox = trimesh.exchange.binvox.binvox_encoder is not None

# Python version as a tuple, i.e. [3, 6]
PY_VER = (sys.version_info.major,
          sys.version_info.minor)

# some repeatable homogeneous transforms to use in tests
transforms = [trimesh.transformations.euler_matrix(np.pi / 4, i, 0)
              for i in np.linspace(0.0, np.pi * 2.0, 100)]
# should be a (100, 4, 4) float
transforms = np.array(transforms)

try:
    # do the imports for Python 2
    from cStringIO import StringIO
    PY3 = False
except ImportError:
    # if that didn't work we're probably on Python 3
    from io import StringIO
    from io import BytesIO
    PY3 = True

# are we on linux
is_linux = 'linux' in platform.system().lower()

# find the current absolute path using inspect
dir_current = os.path.dirname(
    os.path.abspath(os.path.expanduser(__file__)))
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
        location = get_path(name)
        log.info('loading mesh from: %s', location)
        meshes.append(trimesh.load(location, **kwargs))
    if len(meshes) == 1:
        return meshes[0]
    return list(meshes)


def get_path(file_name):
    """
    Get the absolute location of a referenced model file.

    Parameters
    ------------
    file_name : str
      Name of model relative to `repo/models`

    Returns
    ------------
    full : str
      Full absolute path to model.
    """
    return os.path.abspath(
        os.path.join(dir_models, file_name))


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
                # is the loaded mesh a Geometry object or a subclass:
                # Trimesh, PointCloud, Scene
                type_ok = isinstance(m, trimesh.parent.Geometry)
                if raise_error and not type_ok:
                    raise ValueError('%s returned a non- Trimesh object!',
                                     file_name)
                if not isinstance(m, trimesh.Trimesh) or (
                        only_watertight and not m.is_watertight):
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


def check_path2D(path):
    """
    Make basic assertions on Path2D objects
    """
    # root count should be the same as the closed polygons
    assert len(path.root) == len(path.polygons_full)

    # make sure polygons are really polygons
    assert all(type(i).__name__ == 'Polygon'
               for i in path.polygons_full)
    assert all(type(i).__name__ == 'Polygon'
               for i in path.polygons_closed)

    # these should all correspond to each other
    assert len(path.discrete) == len(path.polygons_closed)
    assert len(path.discrete) == len(path.paths)

    # make sure None polygons are not referenced in graph
    assert all(path.polygons_closed[i] is not None
               for i in path.enclosure_directed.nodes())

    if any(e.color is not None for e in path.entities):
        assert path.colors.shape == (len(path.entities), 4)


def scene_equal(a, b):
    """
    Do a simple check on two scenes and assert
    that they have the same geometry.

    Parameters
    ------------
    a : trimesh.Scene
      Object to be compared
    b : trimesh.Scene
      Object to be compared
    """
    # should have the same number of geometries
    assert len(a.geometry) == len(b.geometry)
    for k, m in a.geometry.items():
        # each mesh should correspond by name
        # and have the same volume
        assert np.isclose(
            m.volume, b.geometry[k].volume, rtol=0.001)


def texture_equal(a, b):
    """
    Make sure the texture in two meshes have the same
    face-uv-position results.

    Parameters
    ------------
    a : trimesh.Trimesh
      Mesh to check
    b : trimesh.Trimesh
      Should be identical
    """
    try:
        from scipy.spatial import cKDTree
    except BaseException:
        log.error('no scipy for check!', exc_info=True)
        return

    # an ordered position-face-UV blob to check
    pa = np.hstack((a.vertices, a.visual.uv))[
        a.faces].reshape((-1, 15))
    pb = np.hstack((b.vertices, b.visual.uv))[
        b.faces].reshape((-1, 15))
    # query their actual ordered values against each other
    q = cKDTree(pa).query_ball_tree(cKDTree(pb), r=1e-4)
    assert all(i in match for i, match in enumerate(q))


def check_fuze(fuze):
    """
    Check the classic textured mesh: a fuze bottle
    """
    # these loaded fuze bottles should have textures
    assert isinstance(
        fuze.visual, trimesh.visual.TextureVisuals)
    # image should be loaded with correct resolution
    assert fuze.visual.material.image.size == (1024, 1024)
    # UV coordinates should be unmerged correctly
    assert len(fuze.visual.uv) == 664
    # UV coordinates shouldn't be all zero- ish
    assert fuze.visual.uv[:, :2].ptp(axis=0).min() > 0.1
    # UV coordinates should be between zero and 1
    assert fuze.visual.uv.min() > -tol.merge
    assert fuze.visual.uv.max() < 1 + tol.merge
    # check color factors
    factors = [fuze.visual.material.ambient,
               fuze.visual.material.diffuse,
               fuze.visual.material.specular]
    for f in factors:
        # should be RGBA
        assert len(f) == 4
        assert f.dtype == np.uint8
        # should be between 0 and 255
        assert f.min() >= 0
        assert f.max() <= 255
        # should have some nonzero values
        assert (f > 0).any()

    # vertices should correspond with UV
    assert fuze.vertices.shape == (664, 3)
    # convert TextureVisuals to ColorVisuals
    viz = fuze.visual.to_color()
    assert viz.kind == 'vertex'
    # should be actual colors defined
    assert viz.vertex_colors.ptp(axis=0).ptp() != 0
    # shouldn't crash
    fuze.visual.copy()
    fuze.visual.material.copy()


def wrapload(exported, file_type, **kwargs):
    """
    Reload an exported byte blob into a mesh.

    Parameters
    -----------
    exported : bytes or str
      Exported mesh
    file_type : str
      Type of file to load
    kwargs : dict
      Passed to trimesh.load

    Returns
    ---------
    loaded : trimesh.Trimesh
      Loaded result
    """
    return trimesh.load(file_obj=trimesh.util.wrap_as_stream(exported),
                        file_type=file_type,
                        **kwargs)


TemporaryDirectory = trimesh.util.TemporaryDirectory

# all the JSON files with truth data
data = _load_data()

# find executables to run with subprocess
# formats supported by meshlab for export tests
if any(trimesh.util.which(i) is None
       for i in ['xfvb-run', 'meshlabserver']):
    meshlab_formats = []
else:
    meshlab_formats = ['3ds', 'ply', 'stl', 'obj', 'qobj', 'off', 'ptx', 'vmi',
                       'bre', 'dae', 'ctm', 'pts', 'apts', 'xyz', 'gts', 'pdb',
                       'tri', 'asc', 'x3d', 'x3dv', 'wrl']
