'''
trimesh.py
========
Python library for loading triangular meshes and doing simple operations on them. Included loaders are binary/ASCII STL and Wavefront (OBJ), included exporters are binary STL or COLLADA. If Assimp/pyassimp are available, meshes can be loaded using the assimp loaders.

Using
-----
    >>> import trimesh
    >>> m = trimesh.load_mesh('models/ballA.off')
    >>> m.show()

'''

from sys import version_info as _version_info
if _version_info[:2] < (2, 7):
    message = "Python 2.7 or later is required for trimesh.py (%d.%d detected)."
    raise ImportError(message % sys.version_info[:2])

from .base import Trimesh
from .geometry import unitize, transform_points
from .mesh_io import load_mesh, available_formats
from . import transformations

