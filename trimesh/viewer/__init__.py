"""
viewer
-------------

View and render meshes and scenes via pyglet or inline HTML.
"""

from .. import exceptions
from .notebook import (
    in_notebook,
    scene_to_html,
    scene_to_mo_notebook,
    scene_to_notebook,
)


def __getattr__(name: str):
    """
    Lazily import the viewer, as importing pyglet needs a display.

    Parameters
    ------------
    name : str
      Attribute of this module.

    Returns
    ------------
    value : SceneViewer, render_scene or ExceptionWrapper
      From `trimesh.viewer.windowed` or the import error.
    """
    if name not in ("SceneViewer", "render_scene"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from . import windowed
    except Exception as E:
        # only raise the exception if someone tries to use it
        return exceptions.ExceptionWrapper(E)
    return getattr(windowed, name)


__all__ = [
    "SceneViewer",
    "in_notebook",
    "render_scene",
    "scene_to_html",
    "scene_to_mo_notebook",
    "scene_to_notebook",
]
