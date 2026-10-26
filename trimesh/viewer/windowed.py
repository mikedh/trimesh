"""
windowed.py
---------------

The viewer `Scene.show` opens and `trimesh.viewer.render_scene`: the
OpenGL 3.3 viewer in `trimesh.viewer.opengl.viewer`, which works with
every pyglet. The fixed function viewer for `pyglet<2` is deprecated
and only used if imported explicitly from `trimesh.viewer.pyglet1`.
"""

from .opengl.viewer import SceneViewer
from .render import render_scene

__all__ = ["SceneViewer", "render_scene"]
