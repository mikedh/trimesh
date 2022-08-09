from . import embree3
from . import embree2
from .import basic
from ..constants import tol
from ..util import log

from collections import deque

# engines is an ordered list of available ray-triangle
# engines, faster ones will always be index 0
engines = deque([basic.RayMeshIntersector])

# optionally load an interface to the embree raytracer
# try:
engines.appendleft(embree2.RayMeshIntersector)
# except BaseException as E:
#    log.debug('no embree2', exc_info=True)

# try:
# for development

# if tol.strict:
engines.appendleft(embree3.RayMeshIntersector)
# except BaseException as E:
# 3    log.debug('no embree3', exc_info=True)

# add to __all__ as per pep8
__all__ = [engines]
