from logging import getLogger

from .viewer.pyglet1.conversion import *

getLogger(__name__).warning(
    "`trimesh.rendering` is Pyglet1-specific, and is deprecated. This compatibility shim may be removed any time after June 2027"
)
