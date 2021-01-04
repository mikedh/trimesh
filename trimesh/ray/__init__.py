from .import basic
from ..util import log

from collections import deque

# engines is an ordered list of available ray-triangle
# engines, faster ones will always be index 0
engines = deque([basic.RayMeshIntersector])

# optionally load an interface to the embree raytracer
try:
    from . import embree2
    engines.appendleft(embree2.RayMeshIntersector)
except BaseException as E:
    log.debug('no embree2', exc_info=True)

try:
    from . import embree3
    engines.appendleft(embree3.RayMeshIntersector)
except BaseException as E:
    log.debug('no embree3', exc_info=True)

# add to __all__ as per pep8
__all__ = ['engines']
