"""
https://github.com/mikedh/trimesh
------------------------------------

Trimesh is a pure Python (2.7- 3.3+) library for loading and using triangular meshes with an emphasis on watertight meshes. The goal of the library is to provide a fully featured Trimesh object which allows for easy manipulation and analysis, in the style of the Polygon object in the Shapely library.
"""

from .version import __version__
from .base import Trimesh
from .scene.scene import Scene

from .util import unitize
from .transformations import transform_points

from .constants import tol

from .io.load import load_mesh, load_path, load, available_formats

from . import transformations
from . import primitives
