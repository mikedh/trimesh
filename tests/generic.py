# flake8: noqa
"""
Module which contains most imports and data unit tests
might need, to reduce the amount of boilerplate.
"""

import os
import sys
import io
import json
import types
import copy
import time
import pickle
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
import pytest

import trimesh

from uuid import uuid4

from trimesh.base import Trimesh
from trimesh.constants import tol, tol_path
from collections import deque
from copy import deepcopy
from http.server import SimpleHTTPRequestHandler
import socketserver


tf = trimesh.transformations
TemporaryDirectory = tempfile.TemporaryDirectory


# make a dummy profiler which does nothing
class DummyProfiler(object):
    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(*args, **kwargs):
        pass

    def output_text(*args, **kwargs):
        return "no `pyinstrument`"


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
argv = "".join(sys.argv)
# if we're supposed to have everything
all_dependencies = "ALL_DEPENDENCIES" in argv
# if we're testing rendering scenes
include_rendering = "INCLUDE_RENDERING" in argv

if all_dependencies and not trimesh.ray.has_embree:
    import embreex

try:
    import sympy as sp
except ImportError as E:
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
    raise ValueError("missing binvox")

# `trimesh-setup --matrix` sets i.e. `pyglet==2.1.19` or `blender==4.5.15`
# so a matrix entry can't quietly test a different version than it installed
expect = os.environ.get("TRIMESH_EXPECT")
if expect is not None:
    expect_name, expect_version = expect.split("==")
    if expect_name == "pyglet":
        import pyglet

        installed = pyglet.version
    elif expect_name == "blender":
        # the version of the Blender trimesh will run, i.e. `(4, 5, 15)`
        installed = ".".join(str(i) for i in trimesh.interfaces.blender.version())
    else:
        raise ValueError(f"unknown `TRIMESH_EXPECT={expect}`")
    assert installed == expect_version, f"`TRIMESH_EXPECT={expect}` but `{installed}`"

# Python version as a tuple, i.e. [3, 6]
PY_VER = (sys.version_info.major, sys.version_info.minor)

# some repeatable homogeneous transforms to use in tests
transforms = [
    trimesh.transformations.euler_matrix(np.pi / 4, i, 0)
    for i in np.linspace(0.0, np.pi * 2.0, 100)
]
# should be a (100, 4, 4) float
transforms = np.array(transforms)

from io import StringIO
from io import BytesIO

PY3 = True

# are we on linux
is_linux = "linux" in platform.system().lower()

# find the current absolute path using inspect
dir_current = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# the absolute path for our reference models
dir_models = os.path.abspath(os.path.join(dir_current, "..", "models"))
# the absolute path for our 2D reference models
dir_2D = os.path.abspath(os.path.join(dir_current, "..", "models", "2D"))
# the absolute path for our test data and truth
dir_data = os.path.abspath(os.path.join(dir_current, "data"))

# a logger for tests to call
log = logging.getLogger("trimesh")
log.addHandler(logging.NullHandler())

# turn strings / bytes into file- like objects
io_wrap = trimesh.util.wrap_as_stream


def random(*args, **kwargs):
    """
    A FIXED array of pseudorandom values, not a random stream.

    This re-seeds on every call so it is a constant: `random(10)` returns
    the same array every time and `random(3)` is a prefix of `random(10)`.
    Two calls in one scope are the SAME array, which silently makes
    "independent" inputs identical to each other.

    Correct for one-shot bulk data. If you need more than one draw use
    `RandomSeed`, which yields a real stream.
    """
    state = np.random.RandomState(seed=1)
    return state.random_sample(*args, **kwargs)


def random_transforms(count, translate=1000, seed=0, rigid=True):
    """
    Deterministic generation of random transforms so unit
    tests have limited levels of flake.

    Parameters
    -------------
    count : int
      Number of repeatable but random-ish transforms to generate
    translate : float
      The scale of translation to apply.
    seed : int
      Seed the generator these are drawn from.
    rigid : bool
      If False every fifth transform is mirrored and all are
      scaled differently along each axis between 0.2 and 4.

    Yields
    ------------
    transform : (4, 4) float
      Homogeneous transformation matrix
    """
    # draw eagerly: a `with` block spanning a `yield` would stay open
    # across the consumer and never restore if they stopped early
    with RandomSeed(seed) as r:
        quaternion = r.random((count, 3))
        translate = (r.random((count, 3)) - 0.5) * float(translate)
        # drawn last so rigid transforms don't change with the option
        scale = r.uniform(0.2, 4.0, (count, 3))
    scale[::5, 0] *= -1.0

    for quat, trans, stretch in zip(quaternion, translate, scale):
        matrix = tf.random_rotation_matrix(rand=quat)
        if not rigid:
            matrix[:3, :3] *= stretch
        matrix[:3, 3] = trans
        yield matrix


class RandomSeed:
    """
    A scope with deterministic randomness which leaves no trace.

    Yields a seeded `numpy.random.Generator` to draw test data from, and
    seeds the global numpy RNG for code we don't control inside the block
    — the upstream `transformations` doctests are full of `np.random.*`.
    The original global state is put back on exit even if the body raised
    so a seed can never leak out and shift the values a later test sees.

    Unlike `random` the yielded generator is a *stream*: successive draws
    differ, so two "independent" inputs really are independent.

    Parameters
    -------------
    seed : int
      Value to seed both the generator and the global numpy RNG with.

    Examples
    -------------
    >>> with RandomSeed(0) as r:
    ...     a = r.random((100, 3))
    ...     b = r.random((100, 3))  # genuinely independent of `a`
    """

    def __init__(self, seed: int = 0):
        if not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer, not {type(seed)!s}")
        self.seed = int(seed)

    def __enter__(self):
        # stash the state from before we stomp on it
        self._state = np.random.get_state()
        np.random.seed(self.seed)
        return np.random.default_rng(self.seed)

    def __exit__(self, *args):
        # always restore even if the body raised
        np.random.set_state(self._state)
        return False


# `random` is a fixed source: pin the values rather than checking it
# against itself which would only assert that re-seeding re-seeds
assert np.allclose(random(3), [4.17022005e-01, 7.20324493e-01, 1.14374817e-04])
assert np.allclose(random(3), random(10)[:3])

# `RandomSeed` should reproduce across processes
with RandomSeed(0) as r:
    _check = r.random(10)
with RandomSeed(0) as r:
    assert np.allclose(r.random(10), _check)

# it should be a stream rather than a fixed value: this is the entire
# point of it and the property `random` doesn't have
with RandomSeed(0) as r:
    assert not np.allclose(r.random(10), r.random(10))

# it should leave no trace on the global RNG, including for the
# third-party code it exists to contain
_before = np.random.get_state()[2]
with RandomSeed(0) as r:
    r.random(10)
    np.random.random(5)
assert np.random.get_state()[2] == _before

# transforms should repeat and shouldn't alias rotation into translation
assert np.allclose(list(random_transforms(10)), list(random_transforms(10)))
_check = np.array(list(random_transforms(100, translate=10)))
assert not np.allclose(_check[:, :3, 3], _check[0, :3, 3])


def _load_data():
    """
    Load the JSON files from our truth directory.
    """
    data = {}
    for file_name in os.listdir(dir_data):
        name, extension = os.path.splitext(file_name)
        if extension != ".json":
            continue
        file_path = os.path.join(dir_data, file_name)
        with open(file_path, "r") as file_obj:
            data[name] = json.load(file_obj)

    data["model_paths"] = [os.path.join(dir_models, f) for f in os.listdir(dir_models)]
    data["2D_files"] = [os.path.join(dir_2D, f) for f in os.listdir(dir_2D)]
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
        log.info("loading mesh from: %s", location)
        meshes.append(trimesh.load(location, **kwargs))
    if len(meshes) == 1:
        return meshes[0]
    return list(meshes)


def get_scene(file_name: str, **kwargs) -> trimesh.Scene:
    """
    Return the results of trimesh.load_scene
    """
    return trimesh.load_scene(get_path(file_name), **kwargs)


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
    return os.path.abspath(os.path.join(dir_models, file_name))


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
            self.httpd = socketserver.TCPServer(("", 0), Handler)
            _, self.port = self.httpd.server_address
            self.httpd.serve_forever()

    t = _ServerThread()
    t.daemon = True
    t.start()
    time.sleep(0.2)
    try:
        yield "http://localhost:{}".format(t.port)
    finally:
        # tear down the server even if the `with` body raised
        t.httpd.shutdown()
        t.join()


def get_meshes(
    count=np.inf, raise_error=False, split=False, min_volume=None, only_watertight=True
):
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
                loaded = trimesh.load(os.path.join(dir_models, file_name))
            except BaseException as E:
                if raise_error:
                    log.error(f"failed to load {file_name}")
                    raise E
                continue

            batched = []
            if isinstance(loaded, trimesh.Scene):
                batched.extend(
                    m for m in loaded.geometry.values() if isinstance(m, trimesh.Trimesh)
                )
            elif isinstance(loaded, trimesh.Trimesh):
                batched.append(loaded)

            for mesh in batched:
                # only return our limit
                if returned[0] >= count:
                    return
                # previous checks should ensure only trimesh
                assert isinstance(mesh, trimesh.Trimesh)
                if split:
                    for submesh in mesh.split(only_watertight=only_watertight):
                        checked = check(submesh)
                        if checked is not None:
                            yield checked
                else:
                    checked = check(mesh)
                    if checked is not None:
                        yield checked
        else:
            log.warning("%s has no loader, not running test on!", file_name)


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
            log.error("failed on: {}".format(file_name), exc_info=True)
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
    assert all(type(i).__name__ == "Polygon" for i in path.polygons_full)
    assert all(type(i).__name__ == "Polygon" for i in path.polygons_closed)

    # these should all correspond to each other
    assert len(path.discrete) == len(path.polygons_closed)
    assert len(path.discrete) == len(path.paths)

    # make sure None polygons are not referenced in graph
    assert all(
        path.polygons_closed[i] is not None for i in path.enclosure_directed.nodes()
    )

    if any(e.color is not None for e in path.entities):
        assert path.colors.shape == (len(path.entities), 4)

    # the reported paths should be unique
    unique = set(map(lambda x: tuple(x), path.paths))
    assert len(unique) == len(path.paths)

    # the identifier should be a numpy array
    assert isinstance(path.identifier, np.ndarray)
    assert len(path.identifier) >= 5, path.identifier
    assert isinstance(path.identifier_hash, str)
    assert len(path.identifier_hash) == 64, len(path.identifier_hash)

    assert path.convex_hull is not None


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
        assert np.isclose(m.volume, b.geometry[k].volume, rtol=0.001)
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
        log.error("no scipy for check!", exc_info=True)
        return

    # an ordered position-face-UV blob to check
    pa = np.hstack((a.vertices, a.visual.uv))[a.faces].reshape((-1, 15))
    pb = np.hstack((b.vertices, b.visual.uv))[b.faces].reshape((-1, 15))
    # query their actual ordered values against each other
    q = cKDTree(pa).query_ball_tree(cKDTree(pb), r=1e-4)
    assert all(i in match for i, match in enumerate(q))


def check_fuze(fuze):
    """
    Check the classic textured mesh: a fuze bottle
    """
    # these loaded fuze bottles should have textures
    assert isinstance(fuze.visual, trimesh.visual.TextureVisuals)
    # image should be loaded with correct resolution
    assert fuze.visual.material.image.size == (1024, 1024)
    # UV coordinates should be unmerged correctly
    assert len(fuze.visual.uv) == 664
    # UV coordinates shouldn't be all zero- ish
    assert np.ptp(fuze.visual.uv[:, :2], axis=0).min() > 0.1
    # UV coordinates should be between zero and 1
    assert fuze.visual.uv.min() > -tol.merge
    assert fuze.visual.uv.max() < 1 + tol.merge
    # check color factors
    factors = [
        fuze.visual.material.ambient,
        fuze.visual.material.diffuse,
        fuze.visual.material.specular,
    ]
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
    assert viz.kind == "vertex"
    # should be actual colors defined
    assert np.ptp(np.ptp(viz.vertex_colors, axis=0)) != 0
    # shouldn't crash
    fuze.visual.copy()
    fuze.visual.material.copy()


def roundtrip(exported, file_type, **kwargs):
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
    return trimesh.load(
        file_obj=trimesh.util.wrap_as_stream(exported), file_type=file_type, **kwargs
    )


def require_rendering():
    """Skip without OpenGL unless `INCLUDE_RENDERING`, where failing is loud."""
    try:
        from trimesh.viewer.opengl.viewer import hidden_context

        hidden_context()
    except ValueError:
        # our own shaders failing to compile is a bug, never a skip
        raise
    except Exception as E:
        if include_rendering:
            raise
        raise unittest.SkipTest(f"unable to create an OpenGL context: {E}")


def set_environment(scene, environment, pose=None):
    """
    Light `scene` with an `EnvironmentLight` on a node at `pose` and the
    scene's own lights or the default key and rim lights, returning it.
    """
    from trimesh.scene.lighting import EnvironmentLight

    lights = [light for light in scene.lights if not isinstance(light, EnvironmentLight)]
    if pose is None:
        pose = np.eye(4)
    scene.graph[environment.name] = pose
    scene.lights = [*lights, environment]
    return scene


def offscreen_viewer(scene, resolution=(320, 240), studio=False, **kwargs):
    """A hidden viewer, without floor or occlusion unless `studio`."""
    require_rendering()
    from trimesh.viewer.opengl.viewer import SceneViewer

    if not studio:
        kwargs = {"ground_shadow": False, **kwargs}
        kwargs["flags"] = {"ambient_occlusion": False, **kwargs.get("flags", {})}
    # tests compare against rays from the scene's camera at this resolution
    scene.camera.fit_resolution(resolution)
    return SceneViewer(
        scene, visible=False, start_loop=False, resolution=resolution, **kwargs
    )


def render(scene, include_depth=False, include_segmentation=False, **kwargs):
    """`to_array` of a scene from a hidden viewer which is then closed."""
    viewer = offscreen_viewer(scene, **kwargs)
    try:
        return viewer.to_array(
            include_depth=include_depth, include_segmentation=include_segmentation
        )
    finally:
        viewer.close()


def png(data):
    """Decode PNG bytes to an array."""
    from PIL import Image

    return np.array(Image.open(trimesh.util.wrap_as_stream(data)))


def iou(a, b):
    """Intersection over union of two boolean masks."""
    return (a & b).sum() / max((a | b).sum(), 1)


def camera_hits(scene, meshes=None, corner=(0.0, 0.0)):
    """
    Where every pixel's camera ray first hits `meshes`, which default to
    each mesh of `scene.dump()`, i.e. as posed and skinned. `corner` moves
    every ray across its pixel, in pixels: (0.5, 0.5) is a pixel's corner.
    """
    if meshes is None:
        # bare triangles are all a ray needs, and `node` is `nodes_geometry` order
        nodes = scene.graph.nodes_geometry
        dumped = [m for m in scene.dump() if isinstance(m, trimesh.Trimesh)]
        meshes = [trimesh.Trimesh(m.vertices, m.faces, process=False) for m in dumped]
        node = np.array([nodes.index(m.metadata["node"]) for m in dumped])
    else:
        node = np.arange(len(meshes))
    combined = trimesh.util.concatenate(meshes)
    origins, vectors, pixels = scene.camera_rays()
    if any(corner):
        # the same rays through another point of each pixel
        camera = scene.camera
        coordinates = trimesh.scene.cameras.ray_pixel_coords(camera)[0]
        step = np.ptp(coordinates, axis=0) / (np.array(camera.resolution) - 1)
        coordinates = coordinates + np.array(corner) * step
        to_world = scene.camera_transform
        if camera.magnification is None:
            directions = np.column_stack((coordinates, -np.ones(len(coordinates))))
        else:
            directions = np.tile([0.0, 0.0, -1.0], (len(coordinates), 1))
            origins = trimesh.transformations.transform_points(
                np.column_stack((coordinates, np.zeros(len(coordinates)))), to_world
            )
        vectors = trimesh.util.unitize(
            trimesh.transformations.transform_points(
                directions, to_world, translate=False
            )
        )
    points, ray, face = combined.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )
    mesh = np.repeat(np.arange(len(meshes)), [len(m.faces) for m in meshes])[face]
    return types.SimpleNamespace(
        combined=combined,
        points=points,
        ray=ray,
        face=face,
        # which of `meshes` each hit is on, and which node that is
        mesh=mesh,
        node=node[mesh],
        origins=origins,
        vectors=vectors,
        # camera ray pixels have the first row at the bottom
        rows=scene.camera.resolution[1] - 1 - pixels[ray, 1],
        columns=pixels[ray, 0],
    )


def ray_reference(scene):
    """
    What the camera sees by casting rays: the `node` index (-1 for nothing)
    and view `depth` (0 for nothing) of each pixel's center ray, and `agreed`
    where the center ray and the 4 rays through the corners of the pixel
    all hit the same node, i.e. no silhouette or node edge crosses the pixel.
    """
    width, height = scene.camera.resolution
    forward = trimesh.util.unitize(-scene.camera_transform[:3, 2])
    nodes, depths = [], []
    for corner in [(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]:
        hits = camera_hits(scene, corner=corner)
        node = np.full((height, width), -1)
        node[hits.rows, hits.columns] = hits.node
        nodes.append(node)
        # perspective depth is from the camera and orthographic from its plane
        depth = np.zeros((height, width))
        depth[hits.rows, hits.columns] = (hits.points - hits.origins[hits.ray]) @ forward
        depths.append(depth)
    nodes, depths = np.array(nodes), np.array(depths)
    agreed = (nodes == nodes[0]).all(axis=0)
    return types.SimpleNamespace(
        node=nodes[0],
        depth=depths[0],
        # how far the depth changes across each pixel
        span=np.ptp(depths, axis=0),
        agreed=agreed,
        covered=agreed & (nodes[0] >= 0),
    )


def ray_checks(scene, arrays, prefix: str = ""):
    """
    A render of `scene` (`color`, `depth` and `segmentation` as from
    `to_array`) is exactly what its camera's rays hit: where the center
    and corner rays of a pixel agree the segmentation is that node and the
    depth is the ray's distance along the view axis, and pixels where they
    don't are only along the edges between nodes and the background.
    """
    from scipy import ndimage

    reference = ray_reference(scene)
    node, agreed = reference.node, reference.agreed
    covered = reference.covered
    # depth is float32 so it is rounded to half an epsilon of its own size,
    # which is at most the farthest hit: allow 4 times that. The viewport
    # transform is float32 too so a sample can sit a sliver of a pixel from
    # its center, which moves depth by that fraction of the depth across the
    # pixel: allow 1% where the worst case measured on these scenes is 0.25%
    tolerance = 4.0 * np.finfo(np.float32).eps * reference.depth.max()
    tolerance = tolerance + 0.01 * reference.span
    # an edge is any pixel whose neighbors' center rays hit something else
    edge = ndimage.maximum_filter(node, 3) != ndimage.minimum_filter(node, 3)
    edge = ndimage.binary_dilation(edge, structure=np.ones((3, 3)))
    # nothing is known past the border so a boundary can hide just outside it
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    # and one agreeing pixel next to a disagreeing one can be sampled on the
    # wrong side of a steep surface, so only the deep interior is exact
    deep = ndimage.binary_erosion(agreed, structure=np.ones((3, 3)))
    checks = {
        "rays hit the scene": covered.any(),
        "ambiguity only on edges": not (~agreed & ~edge).any(),
        "segmentation is rays": (arrays.segmentation == node)[agreed].all(),
        "depth is rays": (np.abs(arrays.depth - reference.depth) <= tolerance)[
            deep
        ].all(),
    }
    return {f"{prefix}{name}": bool(ok) for name, ok in checks.items()}


def glb_chunks(export):
    """Split a GLB into its parsed JSON header and its binary chunk."""
    chunks = {}
    start = 12
    while start < len(export):
        length, kind = np.frombuffer(export[start : start + 8], dtype="<u4")
        chunks[int(kind)] = export[start + 8 : start + 8 + int(length)]
        start += 8 + int(length)
    # 0x4E4F534A is "JSON" and 0x004E4942 is "BIN"
    return json.loads(chunks[0x4E4F534A]), chunks.get(0x004E4942, b"")


def accessor_values(header, blob):
    """
    Every accessor of a GLB as (count, width) float64 with normalized
    integers scaled to one, from the spec's own tables rather than
    trimesh's so a shared bug can't hide.
    """
    dtypes = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    values = []
    for accessor in header["accessors"]:
        dtype = np.dtype(dtypes[accessor["componentType"]])
        view = header["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        count = accessor["count"] * widths[accessor["type"]]
        data = np.frombuffer(blob, dtype, count, start).astype(np.float64)
        if accessor.get("normalized", False):
            data /= np.iinfo(dtype).max
        values.append(data.reshape((accessor["count"], -1)))
    return values


def dolly_scene(node, duration=1.0):
    """
    A red box seen by a camera on a dolly, with `node`, which is "box"
    or "dolly", sliding along X a unit a second for `duration` seconds.
    """
    from trimesh.scene.animation import Animation

    box = trimesh.creation.box()
    box.visual.face_colors = [255, 0, 0, 255]
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="box")
    scene.graph.update(frame_to="dolly", matrix=np.eye(4))
    scene.camera = trimesh.scene.Camera(name="lens", fov=(60, 45))
    lens = trimesh.transformations.translation_matrix([0, 0, 6])
    scene.graph.update(frame_from="dolly", frame_to="lens", matrix=lens)
    times = np.linspace(0.0, duration, 5)
    slide = np.tile(np.eye(4), (len(times), 1, 1))
    slide[:, 0, 3] = times
    scene.animations.append(Animation(edges=[(None, node)], times=times, matrices=slide))
    return scene


def to_glb_bytes(geom: trimesh.Geometry) -> bytes:
    """
    Returns the content of the binary glTF file for the given geometry.

    Parameters
    ----------
    geom : trimesh.Geometry
      Geometry to export

    Returns
    -------
    data : bytes
      Content of the binary glTF file
    """
    data = trimesh.exchange.gltf.export_glb(geom)

    return data


def from_glb_bytes(data: bytes) -> trimesh.Scene:
    """
    Returns the scene from the given binary glTF file content.

    Parameters
    ----------
    data : bytes
      Content of the binary glTF file

    Returns
    -------
    scene : trimesh.Scene
      Scene
    """
    bytes_stream = io.BytesIO(data)
    kwargs = trimesh.exchange.gltf.load_glb(bytes_stream)

    scene = trimesh.exchange.load._load_kwargs(kwargs)

    return scene


# Khronos' official file validator
# can be installed with the helper script:
# `helpers/trimesh-setup --install=gltf_validator`
_gltf_validator = shutil.which("gltf_validator")


def validate_glb(data, name=None):
    """Raise if the Khronos validator reports any error or warning on a GLB."""
    if _gltf_validator is None:
        if all_dependencies:
            raise ValueError("gltf_validator is required!")
        log.warning("no gltf_validator!")
        return
    with tempfile.NamedTemporaryFile(suffix=".glb") as f:
        f.write(data)
        f.flush()
        # gltf_validator has bugs when run outside of the file's directory
        temp_dir, file_name = os.path.split(f.name)
        report = subprocess.run(
            [_gltf_validator, file_name, "-o"], cwd=temp_dir, capture_output=True
        )
        issues = json.loads(report.stdout)["issues"]
    # exports are clean: a warning is a viewer somewhere doing the wrong thing,
    # except tangents being absent, which the glTF spec has clients generate
    # with MikkTSpace and which source files like AlphaBlendModeTest.glb note
    # counted from the listed codes, so a truncated list only counts more
    generated = sum(
        message["severity"] == 1
        and message["code"] == "MESH_PRIMITIVE_GENERATED_TANGENT_SPACE"
        for message in issues["messages"]
    )
    counted = issues["numWarnings"] - generated
    if report.returncode != 0 or issues["numErrors"] > 0 or counted > 0:
        log.error(f"`{name}` failed validation: {issues['messages']} {report.stderr}")
        raise ValueError("gltf_validator failed")


def decode_video(path, resolution):
    """Decode a video file to (frames, height, width, 3) uint8 with ffmpeg."""
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    width, height = resolution
    return np.frombuffer(decoded, dtype=np.uint8).reshape((-1, height, width, 3))


# all the JSON files with truth data
data = _load_data()

# find executables to run with subprocess
# formats supported by meshlab for export tests
try:
    import pymeshlab

    meshlab_formats = [
        "3ds",
        "ply",
        "stl",
        "obj",
        "qobj",
        "off",
        "ptx",
        "vmi",
        "bre",
        "dae",
        "ctm",
        "pts",
        "apts",
        "gts",
        "pdb",
        "tri",
        "asc",
        "x3d",
        "x3dv",
        "wrl",
    ]
except BaseException:
    meshlab_formats = []
