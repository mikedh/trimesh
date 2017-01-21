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
from .version import __version__
from .base import Trimesh
from .scene.scene import Scene

from .util import unitize
from .transformations import transform_points

from .io.load import load_mesh, load_path, load, available_formats

from . import transformations
from . import primitives
