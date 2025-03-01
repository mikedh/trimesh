"""
viewer
-------------

View meshes and scenes via pyglet or inline HTML.
"""

from .. import exceptions
from .notebook import in_notebook, scene_to_html, scene_to_notebook

try:
    # try importing windowed which will fail
    # if we can't create an openGL context
    from .pyglet2 import SceneViewer, render_scene
except ImportError as E:
    try:
        from .pyglet1 import SceneViewer, render_scene
        # if windowed failed to import only raise
        # the exception if someone tries to use them
    except ImportError as J:
        print(J)
        SceneViewer = exceptions.ExceptionWrapper(E)
        render_scene = exceptions.ExceptionWrapper(E)


# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [
    "SceneViewer",
    "in_notebook",
    "render_scene",
    "scene_to_html",
    "scene_to_notebook",
]
