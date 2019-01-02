"""
viewer
-------------

View meshes and scenes via pyglet or inline HTML.
"""

from .windowed import (SceneViewer,
                       render_scene)

from .notebook import (in_notebook,
                       scene_to_notebook,
                       scene_to_html)

# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [SceneViewer,
           render_scene,
           in_notebook,
           scene_to_notebook,
           scene_to_html]
