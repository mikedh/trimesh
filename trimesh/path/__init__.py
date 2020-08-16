"""
trimesh.path
-------------

Handle 2D and 3D vector paths such as those contained in an
SVG or DXF file.
"""

from .path import Path2D, Path3D

# explicitly add objects to all as per pep8
__all__ = ['Path2D', 'Path3D']
