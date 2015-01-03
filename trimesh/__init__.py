"""
NetworkX
========
    NetworkX (NX) is a Python package for the creation, manipulation, and
    study of the structure, dynamics, and functions of complex networks.
    https://networkx.lanl.gov/
Using
-----
    Just write in Python
    >>> import networkx as nx
    >>> G=nx.Graph()
    >>> G.add_edge(1,2)
    >>> G.add_node(42)
    >>> print(sorted(G.nodes()))
    [1, 2, 42]
    >>> print(sorted(G.edges()))
    [(1, 2)]
"""

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    message = "Python 2.7 or later is required for trimesh.py (%d.%d detected)."
    raise ImportError(message % sys.version_info[:2])

from trimesh.mesh_base import *
from trimesh.mesh_io import load_mesh, available_formats
