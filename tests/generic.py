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


def random_transforms(count, translate=1000, seed=0):
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

    for quat, trans in zip(quaternion, translate):
        matrix = tf.random_rotation_matrix(rand=quat)
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
        from trimesh.viewer.windowed import hidden_context

        hidden_context()
    except ValueError:
        # our own shaders failing to compile is a bug, never a skip
        raise
    except BaseException as E:
        if include_rendering:
            raise
        raise unittest.SkipTest(f"unable to create an OpenGL context: {E}")


def set_render(scene, **settings):
    """Change `RenderSettings` fields stored in a scene, returning it."""
    from dataclasses import replace

    from trimesh.scene.metadata import RenderSettings

    stored = RenderSettings.from_metadata(scene.metadata)
    scene.metadata.update(replace(stored, **settings).to_metadata())
    return scene


def offscreen_viewer(scene, resolution=(320, 240), studio=False, **kwargs):
    """A hidden viewer, without floor or occlusion unless `studio`."""
    require_rendering()
    from trimesh.viewer.windowed import SceneViewer

    if not studio:
        set_render(scene, ground_shadow=False)
        kwargs["flags"] = {"ambient_occlusion": False, **kwargs.get("flags", {})}
    # tests compare against rays from the scene's camera at this resolution
    scene.camera.resolution = resolution
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


def rgb(scene, **kwargs):
    """A render's color as (h, w, 3) float64."""
    return render(scene, **kwargs).color[..., :3].astype(np.float64)


def png(data):
    """Decode PNG bytes to an array."""
    from PIL import Image

    return np.array(Image.open(trimesh.util.wrap_as_stream(data)))


def iou(a, b):
    """Intersection over union of two boolean masks."""
    return (a & b).sum() / max((a | b).sum(), 1)


def camera_hits(scene, meshes=None):
    """
    Where every pixel's camera ray first hits `meshes`, which default
    to each geometry node's mesh in `nodes_geometry` order.
    """
    if meshes is None:
        meshes = [
            scene.geometry[name].copy(include_visual=False).apply_transform(transform)
            for transform, name in (scene.graph[n] for n in scene.graph.nodes_geometry)
        ]
    combined = trimesh.util.concatenate(meshes)
    origins, vectors, pixels = scene.camera_rays()
    points, ray, face = combined.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )
    return types.SimpleNamespace(
        combined=combined,
        points=points,
        ray=ray,
        face=face,
        # which of `meshes` each hit is on
        mesh=np.repeat(np.arange(len(meshes)), [len(m.faces) for m in meshes])[face],
        origins=origins,
        vectors=vectors,
        # camera ray pixels have the first row at the bottom
        rows=scene.camera.resolution[1] - 1 - pixels[ray, 1],
        columns=pixels[ray, 0],
    )


def silhouette_checks(scene, image, prefix: str = "") -> dict:
    """A render covers exactly the pixels whose camera ray hits the scene."""
    from scipy import ndimage

    rendered = (image[..., :3] < 250).any(axis=-1)
    # the whole scene as drawn, including skinning
    hits = camera_hits(scene, [scene.to_mesh()])
    reference = np.zeros(rendered.shape, dtype=bool)
    reference[hits.rows, hits.columns] = True
    checks = {
        "reference is not empty": reference.any(),
        # antialiased edges may touch any of the 8 neighbors
        "rendered inside reference": not (
            rendered & ~ndimage.binary_dilation(reference, structure=np.ones((3, 3)))
        ).any(),
        "reference inside rendered": not (
            ndimage.binary_erosion(reference) & ~rendered
        ).any(),
        "silhouette IoU": iou(reference, rendered) > 0.9,
    }
    return {f"{prefix}{name}": ok for name, ok in checks.items()}


def camera_ray_reference(scene):
    """The view depth and node index each pixel's camera ray hits, or 0 and -1."""
    width, height = scene.camera.resolution
    hits = camera_hits(scene)
    forward = -scene.camera_transform[:3, 2]
    # perspective depth is from the camera and orthographic from its plane
    eye = scene.camera_transform[:3, 3]
    if scene.camera.magnification is not None:
        eye = hits.origins[hits.ray]
    depth = np.zeros((height, width))
    index = np.full((height, width), -1)
    depth[hits.rows, hits.columns] = (hits.points - eye) @ forward
    index[hits.rows, hits.columns] = hits.mesh
    return depth, index


def environment_reference(environment, directions, sharpness):
    """
    An environment's linear light averaged by a normalized spherical
    Gaussian around each direction, from textbook float64 integrals.
    """
    directions = np.asarray(directions, dtype=np.float64)
    sharpness = np.broadcast_to(np.asarray(sharpness, dtype=np.float64), len(directions))
    # the mean cosine of a spherical Gaussian is coth(s) - 1/s
    mean = 1.0 / np.tanh(sharpness) - 1.0 / sharpness
    height = (directions @ trimesh.util.unitize(environment.up)) * mean
    sky, ground = np.array(environment.sky), np.array(environment.ground)
    light = ground + (sky - ground) * (0.5 + 0.5 * height)[:, None]
    # a kernel of sharpness s integrates to 2 pi (1 - exp(-2 s)) / s
    total = 2.0 * np.pi * -np.expm1(-2.0 * sharpness) / sharpness
    for box in environment.softboxes:
        axis = trimesh.util.unitize(box.direction)
        combined = np.linalg.norm(
            box.sharpness * axis + sharpness[:, None] * directions, axis=1
        )
        # 4 pi exp(-s1 - s2) sinh(r) / r without overflowing sinh
        product = (
            2.0
            * np.pi
            * np.exp(combined - box.sharpness - sharpness)
            * -np.expm1(-2.0 * combined)
            / combined
        )
        light = light + np.array(box.color) * (product / total)[:, None]
    return light


def shade_reference(scene, normal=None, studio=None):
    """
    The sRGB 0-255 image the viewer should draw for opaque `PBRMaterial`
    meshes, NaN where no ray hits, evaluated independently from camera
    rays and the glTF and `KHR_lights_punctual` specs, optionally with a
    constant tangent space `normal` map and lights from `studio`.
    """
    from trimesh.scene.lighting import DirectionalLight, SpotLight
    from trimesh.scene.metadata import (
        DIFFUSE_SHARPNESS,
        SPECULAR_SHARPNESS,
        RenderSettings,
    )

    if studio is None:
        studio = RenderSettings.from_metadata(scene.metadata).resolve(
            scene, scene.camera_transform
        )
    width, height = scene.camera.resolution
    meshes = [mesh.smooth_shaded for mesh in scene.dump()]
    hits = camera_hits(scene, meshes)
    points, corners = hits.points, hits.combined.faces[hits.face]
    barycentric = trimesh.triangles.points_to_barycentric(
        hits.combined.triangles[hits.face], points
    )
    surface = trimesh.util.unitize(
        (hits.combined.vertex_normals[corners] * barycentric[:, :, None]).sum(axis=1)
    )
    view = -trimesh.util.unitize(hits.vectors[hits.ray])
    # light the back of a triangle as if it were the front
    surface[(surface * view).sum(axis=1) < 0.0] *= -1.0
    if normal is not None:
        # the tangent and bitangent are how position changes with uv
        uv = np.vstack([m.visual.uv for m in meshes])[corners]
        position = hits.combined.vertices[corners]
        inverse = np.linalg.inv(uv[:, 1:] - uv[:, :1])
        edges = position[:, 1:] - position[:, :1]
        tangent = trimesh.util.unitize(np.matmul(inverse[:, :1], edges)[:, 0])
        bitangent = trimesh.util.unitize(np.matmul(inverse[:, 1:2], edges)[:, 0])
        surface = trimesh.util.unitize(
            tangent * normal[0] + bitangent * normal[1] + surface * normal[2]
        )

    materials = [m.visual.material for m in meshes]
    base = (np.array([m.baseColorFactor for m in materials]) / 255.0)[hits.mesh, :3]
    metallic = np.array([[m.metallicFactor] for m in materials])[hits.mesh]
    roughness = np.clip([[m.roughnessFactor] for m in materials], 0.04, 1.0)[hits.mesh]
    reflectance = 0.04 * (1.0 - metallic) + base * metallic
    diffuse_color = base * (1.0 - metallic)
    # `KHR_materials_diffuse_transmission` mixes the dielectric Lambert
    # reflection with Lambert transmission of light from the other side
    transmission = np.array([[m.diffuseTransmissionFactor or 0.0] for m in materials])
    transmission_color = np.ones((len(materials), 3))
    for index, material in enumerate(materials):
        if material.diffuseTransmissionColorFactor is not None:
            transmission_color[index] = material.diffuseTransmissionColorFactor
    transmission_color = (transmission_color * transmission)[hits.mesh] * (1.0 - metallic)
    diffuse_color = diffuse_color * (1.0 - transmission[hits.mesh])
    alpha_squared = roughness**4
    geometry_k = (roughness + 1.0) ** 2 / 8.0
    normal_dot_view = np.maximum((surface * view).sum(axis=1, keepdims=True), 1e-4)

    color = np.zeros_like(points)
    for light, pose in zip(studio.lights, studio.transforms):
        # lights shine down their -Z, and only directional lights have
        # no position, falloff or range
        backwards = trimesh.util.unitize(pose[:3, 2])
        to_light = np.tile(backwards, (len(points), 1))
        distance_squared = np.ones((len(points), 1))
        window = np.ones_like(distance_squared)
        if not isinstance(light, DirectionalLight):
            to_light = pose[:3, 3] - points
            distance_squared = (to_light**2).sum(axis=1, keepdims=True)
            if light.radius is not None:
                window = np.clip(1.0 - (distance_squared / light.radius**2) ** 2, 0, 1)
        direction = trimesh.util.unitize(to_light)
        cone = np.ones_like(distance_squared)
        if isinstance(light, SpotLight):
            inner, outer = np.cos(light.innerConeAngle), np.cos(light.outerConeAngle)
            cosine = (direction * backwards).sum(axis=1, keepdims=True)
            cone = np.clip((cosine - outer) / max(inner - outer, 1e-3), 0.0, 1.0)
        radiance = light.color[:3] / 255.0 * light.intensity * (window * cone) ** 2
        radiance = radiance / distance_squared

        half = trimesh.util.unitize(direction + view)
        cosine_light = (surface * direction).sum(1, keepdims=True)
        normal_dot_light = np.maximum(cosine_light, 0.0)
        # light from behind is transmitted with the fresnel term of the
        # light mirrored through the surface
        behind = np.maximum(-cosine_light, 0.0)
        mirrored = trimesh.util.unitize(direction + 2.0 * behind * surface + view)
        mirrored_dot_half = np.clip((view * mirrored).sum(1, keepdims=True), 0.0, 1.0)
        transmitted = (
            (1.0 - reflectance)
            * (1.0 - (1.0 - mirrored_dot_half) ** 5)
            * transmission_color
            / np.pi
        )
        normal_dot_half = np.maximum((surface * half).sum(1, keepdims=True), 0.0)
        view_dot_half = np.maximum((view * half).sum(1, keepdims=True), 0.0)
        fresnel = reflectance + (1.0 - reflectance) * (1.0 - view_dot_half) ** 5
        distribution = alpha_squared / (
            np.pi * (normal_dot_half**2 * (alpha_squared - 1.0) + 1.0) ** 2
        )
        geometry = (
            normal_dot_view / (normal_dot_view * (1 - geometry_k) + geometry_k)
        ) * (normal_dot_light / (normal_dot_light * (1 - geometry_k) + geometry_k))
        specular = (
            fresnel
            * distribution
            * geometry
            / (4 * normal_dot_view * normal_dot_light + 1e-4)
        )
        diffuse = (1.0 - fresnel) * diffuse_color / np.pi
        color += (
            (diffuse + specular) * normal_dot_light + transmitted * behind
        ) * radiance

    # Karis's fit of the environment BRDF
    fit = roughness * [-1.0, -0.0275, -0.572, 0.022] + [1.0, 0.0425, 1.04, -0.04]
    bias = (
        np.minimum(fit[:, :1] ** 2, np.exp2(-9.28 * normal_dot_view)) * fit[:, :1]
        + fit[:, 1:2]
    )
    # diffuse sees the cosine lobe around the normal and specular the
    # reflection blurred by a spherical Gaussian as wide as the GGX lobe
    reflected = 2.0 * (surface * view).sum(axis=1, keepdims=True) * surface - view
    environment = studio.environment
    color += diffuse_color * environment_reference(
        environment, surface, DIFFUSE_SHARPNESS
    ) + transmission_color * environment_reference(
        environment, -surface, DIFFUSE_SHARPNESS
    )
    blur = np.exp(np.polyval(SPECULAR_SHARPNESS, np.log(roughness[:, 0])))
    color += environment_reference(environment, reflected, blur) * (
        reflectance * (-1.04 * bias + fit[:, 2:3]) + 1.04 * bias + fit[:, 3:4]
    )

    # the Khronos PBR Neutral tone mapper
    darkest = color.min(axis=1, keepdims=True)
    color -= np.where(darkest < 0.08, darkest - 6.25 * darkest**2, 0.04)
    peak = color.max(axis=1, keepdims=True)
    start, headroom = 0.76, 0.24
    new_peak = 1.0 - headroom**2 / (peak + headroom - start)
    fade = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0)
    compressed = color * new_peak / np.maximum(peak, 1e-12)
    compressed = compressed * (1 - fade) + new_peak * fade
    color = np.clip(np.where(peak < start, color, compressed), 0.0, 1.0)
    srgb = np.where(color < 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)

    image = np.full((height, width, 3), np.nan)
    image[hits.rows, hits.columns] = srgb * 255.0
    return image


def pbr_mesh(mesh, color, metallic, roughness, uv=None):
    """Give a mesh a plain `PBRMaterial`, with zero uv unless passed."""
    if uv is None:
        uv = np.zeros((len(mesh.vertices), 2))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=metallic, roughnessFactor=roughness
        ),
    )
    return mesh


def affine(count, seed=0, rigid=False):
    """Random transforms, mirrored and non-uniformly scaled unless `rigid`."""
    random = np.random.default_rng(seed)
    matrices = trimesh.transformations.random_rotation_matrix(num=count, seed=seed)
    if not rigid:
        scale = random.uniform(0.2, 4.0, (count, 3))
        scale[::5, 0] *= -1.0
        matrices[:, :3, :3] *= scale.reshape((-1, 1, 3))
    matrices[:, :3, 3] = random.uniform(-10.0, 10.0, (count, 3))
    return matrices


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
    """Every accessor of a GLB as a (count, width) array of its stored dtype."""
    from trimesh.exchange.gltf import _dtypes, _shapes

    values = []
    for accessor in header["accessors"]:
        view = header["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        dtype = np.dtype(_dtypes[accessor["componentType"]])
        width = int(np.prod(_shapes[accessor["type"]]))
        data = np.frombuffer(
            blob[start : start + accessor["count"] * width * dtype.itemsize], dtype
        )
        values.append(data.reshape((accessor["count"], width)))
    return values


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
    # exports are clean: a warning is a viewer somewhere doing the wrong thing
    if report.returncode != 0 or issues["numErrors"] > 0 or issues["numWarnings"] > 0:
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
