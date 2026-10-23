"""
viewer
-------------

View meshes and scenes via pyglet or inline HTML, and render
them to images or video.
"""

from .. import exceptions
from .notebook import (
    in_notebook,
    scene_to_html,
    scene_to_mo_notebook,
    scene_to_notebook,
)
from .render import render_scene


def __getattr__(name: str):
    """
    Import the OpenGL viewer only when it is used, as importing
    pyglet connects to a display or creates an OpenGL context.

    Parameters
    ------------
    name : str
      Attribute of this module.

    Returns
    ------------
    value : SceneViewer or ExceptionWrapper
      The viewer, or why it couldn't be imported.
    """
    if name != "SceneViewer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .windowed import SceneViewer
    except Exception as E:
        # only raise the exception if someone tries to use it
        return exceptions.ExceptionWrapper(E)
    return SceneViewer


# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [
    "SceneViewer",
    "in_notebook",
    "render_scene",
    "scene_to_html",
    "scene_to_mo_notebook",
    "scene_to_notebook",
]
