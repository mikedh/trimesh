from .import basic
from ..constants import tol
from ..util import log

from collections import deque

# engines is an ordered list of available ray-triangle
# engines, faster ones will always be index 0
engines = deque([basic.RayMeshIntersector])

from . import embree3
engines.appendleft(embree3.RayMeshIntersector)
# except BaseException as E:
# 3    log.debug('no embree3', exc_info=True)

# add to __all__ as per pep8
__all__ = [engines]
