"""
viewer
-------------

View meshes and scenes via pyglet or inline HTML.
"""

from .. import exceptions
from .notebook import (
    in_notebook,
    scene_to_html,
    scene_to_mo_notebook,
    scene_to_notebook,
)

try:
    # the modern shader-based viewer for `pyglet>=2`
    from .pyglet2 import SceneViewer, render_scene
except ImportError as E:
    try:
        # fall back to the legacy fixed-function viewer for `pyglet<2`
        from .pyglet1.viewer import SceneViewer, render_scene
    except ImportError:
        # neither pyglet was importable: defer the error until use
        SceneViewer = exceptions.ExceptionWrapper(E)
        render_scene = exceptions.ExceptionWrapper(E)


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
