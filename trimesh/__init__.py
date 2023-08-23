"""
https://github.com/mikedh/trimesh
------------------------------------

Trimesh is a pure Python (2.7- 3.3+) library for loading and using triangular
meshes with an emphasis on watertight meshes. The goal of the library is to
provide a fully featured Trimesh object which allows for easy manipulation
and analysis, in the style of the Polygon object in the Shapely library.
"""

# current version
# avoid a circular import in trimesh.base
from . import bounds, collision, nsphere, primitives, smoothing, voxel

# geometry objects
from .base import Trimesh

# general numeric tolerances
from .constants import tol

# loader functions
from .exchange.load import available_formats, load, load_mesh, load_path, load_remote
from .points import PointCloud
from .scene.scene import Scene
from .transformations import transform_points

# utility functions
from .util import unitize
from .version import __version__

try:
    # handle vector paths
    from . import path
except BaseException as E:
    # raise a useful error if path hasn't loaded
    from .exceptions import ExceptionWrapper
    path = ExceptionWrapper(E)

# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [__version__,
           'Trimesh',
           'PointCloud',
           'Scene',
           'voxel',
           'unitize',
           'bounds',
           'nsphere',
           'collision',
           'smoothing',
           'tol',
           'path',
           'load',
           'load_mesh',
           'load_path',
           'load_remote',
           'primitives',
           'transform_points',
           'available_formats']
