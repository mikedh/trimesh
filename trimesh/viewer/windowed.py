"""
windowed.py
---------------

The OpenGL 3.3 viewer used by `Scene.show` and `render_scene`.
The deprecated `pyglet<2` viewer is in `trimesh.viewer.pyglet1`.
"""

from .opengl.viewer import SceneViewer
from .render import render_scene

__all__ = ["SceneViewer", "render_scene"]
