"""
windowed.py
---------------

The viewer `Scene.show` opens and `Scene.save_image` renders with: the
OpenGL 3.3 viewer in `trimesh.viewer.opengl.viewer`, or the fixed
function viewer in `trimesh.viewer.pyglet1.viewer` on `pyglet<2`.
"""

import warnings

import pyglet

if int(pyglet.version.split(".")[0]) < 2:
    with warnings.catch_warnings():
        # the one warning is this module's, not the package it imports
        warnings.simplefilter("ignore", DeprecationWarning)
        from .pyglet1.viewer import SceneViewer, render_scene

    warnings.warn(
        "`pyglet<2` support is deprecated and will be removed after April 2028: "
        + "upgrade with `pip install -U pyglet` for the new viewer",
        category=DeprecationWarning,
        stacklevel=2,
    )
else:
    from .opengl.viewer import SceneViewer
    from .render import render_scene

__all__ = ["SceneViewer", "render_scene"]
