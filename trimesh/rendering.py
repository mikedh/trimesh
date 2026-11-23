"""
rendering.py
--------------

Deprecated: moved to `trimesh.viewer.pyglet1.conversion`.
"""

import warnings


def __getattr__(name: str):
    """
    Forward to `trimesh.viewer.pyglet1.conversion` with a warning.

    Parameters
    ------------
    name : str
      Attribute of the old module.

    Returns
    ------------
    value : any
      The attribute from its new location.
    """
    if name.startswith("__"):
        # dunders probed by `inspect`, `pickle` and `pytest`
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with warnings.catch_warnings():
        # silence the pyglet1 package warning: this module warns below
        warnings.simplefilter("ignore", DeprecationWarning)
        from .viewer.pyglet1 import conversion

    warnings.warn(
        "`trimesh.rendering` is deprecated and will be removed after "
        + "April 2028: use `trimesh.viewer.pyglet1.conversion`",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return getattr(conversion, name)
