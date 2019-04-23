"""
viewer
-------------

View meshes and scenes via pyglet or inline HTML.
"""


from .notebook import (in_notebook,
                       scene_to_notebook,
                       scene_to_html)


def _cloture(exc):
    """
    Return a function which will accept any arguments
    but raise the exception when called.

    Parameters
    ------------
    exc : Exception
      Will be raised later

    Returns
    -------------
    failed : function
      When called will raise `exc`
    """
    # scoping will save exception
    def failed(*args, **kwargs):
        raise exc
    return failed


try:
    # try importing windowed which will fail
    # if we can't create an openGL context
    from .windowed import (SceneViewer,
                           render_scene)
except BaseException as E:
    # if windowed failed to import only raise
    # the exception if someone tries to use them
    SceneViewer = _cloture(E)
    render_scene = _cloture(E)


try:
    from .widget import SceneWidget
except BaseException as E:
    SceneWidget = _cloture(E)


# this is only standard library imports

# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [SceneWidget,
           SceneViewer,
           render_scene,
           in_notebook,
           scene_to_notebook,
           scene_to_html]
