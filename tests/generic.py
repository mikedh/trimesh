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

from uuid import uuid4

from trimesh.base import Trimesh
from trimesh.constants import tol, tol_path
from collections import deque
from copy import deepcopy

tf = trimesh.transformations

if sys.version_info >= (3, 1):
    # Python 3
    from http.server import SimpleHTTPRequestHandler
    import socketserver

else:
    # Python 2
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    import SocketServer as socketserver

# make a dummy profiler which does nothing


class DummyProfiler(object):
    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(*args, **kwargs):
        pass

    def output_text(*args, **kwargs):
        return 'no `pyinstrument`'


# make sure dummy profiler works
with DummyProfiler() as _P:
    pass
assert len(_P.output_text()) > 0

try:
    from pyinstrument import Profiler
except BaseException:
    Profiler = DummyProfiler


# should we require all soft dependencies
# this is set in the docker images to catch missing packages
argv = ''.join(sys.argv)
# if we're supposed to have everything
all_dependencies = 'ALL_DEPENDENCIES' in argv
# if we're testing rendering scenes
include_rendering = 'INCLUDE_RENDERING' in argv

if all_dependencies and not trimesh.ray.has_embree:
    raise ValueError('missing embree!')

try:
    import sympy as sp
except ImportError as E:
    if all_dependencies:
        raise E
    sp = None

try:
    import jsonschema
except BaseException as E:
    jsonschema = trimesh.exceptions.ExceptionWrapper(E)

# make sure functions know they should run additional
# potentially slow validation checks and raise exceptions
trimesh.util._STRICT = True
trimesh.constants.tol.strict = True
trimesh.constants.tol_path.strict = True


try:
    from mapbox_earcut import triangulate_float64
    has_earcut = True
except BaseException as E:
    if all_dependencies:
        raise E
    else:
        has_earcut = False

try:
    from shapely.geometry import Point, Polygon, LineString
    has_path = True
except ImportError as E:
    if all_dependencies:
        raise E
    has_path = False

try:
    from scipy import spatial, sparse
except BaseException as E:
    if all_dependencies:
        raise E

try:
    import networkx as nx
except BaseException as E:
    pass

# find_executable for binvox
has_binvox = trimesh.exchange.binvox.binvox_encoder is not None
if all_dependencies and not has_binvox:
    raise ValueError('missing binvox')

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
    A random function always seeded from the same value so tests
    can use random data but they execute mostly deterministically
    in a large test matrix.

    Drop-in replacement for `np.random.random(*args, **kwargs)`
    """
    state = np.random.RandomState(seed=1)
    return state.random_sample(*args, **kwargs)


def random_transforms(count, translate=1000):
    """
    Deterministic generation of random transforms so unit
    tests have limited levels of flake.

    Parameters
    -------------
    count : int
      Number of repeatable but random-ish transforms to generate
    translate : float
      The scale of translation to apply.

    Yields
    ------------
    transform : (4, 4) float
      Homogenous transformation matrix
    """
    quaternion = random((count, 3))
    translate = (random((count, 3)) - 0.5) * float(translate)

    for quat, trans in zip(quaternion, translate):
        matrix = tf.random_rotation_matrix(rand=quat)
        matrix[:3, 3] = trans
        yield matrix


# random should be deterministic
assert np.allclose(random(10), random(10))
assert np.allclose(list(random_transforms(10)),
                   list(random_transforms(10)))


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
               split=False,
               min_volume=None,
               only_watertight=True):
    """
    Get meshes to test with.

    Parameters
    ----------
    count : int
      Approximate number of meshes you want
    raise_error : bool
      Raise exceptions in `trimesh.load` or just continue.
    split : bool
      Split multibody meshes into single body meshes.
    min_volume : None or float
      Only return meshes above a volume threshold
    only_watertight : bool
      Only return watertight meshes

    Yields
    ----------
    mesh : trimesh.Trimesh
      Trimesh objects from models folder
    """
    # use deterministic file name order
    file_names = sorted(os.listdir(dir_models))

    # count items we've returned as an array so
    # we don't have to make it a global
    returned = [0]

    def check(item):
        # only return our limit
        if returned[0] >= count:
            return None

        # run our return argument checks on an item
        if only_watertight and not item.is_watertight:
            return None

        if min_volume is not None and item.volume < min_volume:
            return None

        returned[0] += 1
        return item

    for file_name in file_names:
        extension = trimesh.util.split_extension(file_name).lower()
        if extension in trimesh.available_formats():
            try:
                loaded = trimesh.load(
                    os.path.join(dir_models, file_name))
            except BaseException as E:
                if raise_error:
                    raise E
                continue

            batched = []
            if isinstance(loaded, trimesh.Scene):
                batched.extend(m for m in loaded.geometry.values()
                               if isinstance(m, trimesh.Trimesh))
            elif isinstance(loaded, trimesh.Trimesh):
                batched.append(loaded)

            for mesh in batched:
                mesh.metadata['file_name'] = file_name
                # only return our limit
                if returned[0] >= count:
                    return
                # previous checks should ensure only trimesh
                assert isinstance(mesh, trimesh.Trimesh)
                if split:
                    for submesh in mesh.split(
                            only_watertight=only_watertight):
                        checked = check(submesh)
                        if checked is not None:
                            yield checked
                else:
                    checked = check(mesh)
                    if checked is not None:
                        yield checked
        else:
            log.warning('%s has no loader, not running test on!',
                        file_name)


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
    # the axis aligned bounding box should be the same
    assert np.allclose(a.bounds, b.bounds)

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
try:
    import pymeshlab
    meshlab_formats = ['3ds', 'ply', 'stl', 'obj', 'qobj', 'off', 'ptx', 'vmi',
                       'bre', 'dae', 'ctm', 'pts', 'apts', 'gts', 'pdb',
                       'tri', 'asc', 'x3d', 'x3dv', 'wrl']
except BaseException:
    meshlab_formats = []
