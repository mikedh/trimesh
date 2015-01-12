"""
trimesh.py
========
Python library for loading triangular meshes and doing simple operations on them. Included loaders are binary/ASCII STL and Wavefront (OBJ), included exporters are binary STL or COLLADA. If Assimp/pyassimp are available, meshes can be loaded using the assimp loaders.

Using
-----
    >>> import trimesh
    >>> m = trimesh.load_mesh('models/ballA.off')
    >>> m.show()
"""

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    message = "Python 2.7 or later is required for trimesh.py (%d.%d detected)."
    raise ImportError(message % sys.version_info[:2])

from trimesh.base import Trimesh
from trimesh.mesh_io import load_mesh, available_formats
from trimesh.geometry import unitize, transform_points
