#to avoid confusion over function vs module name
from time import time as time_function

# consider floating point numbers less than this zero
TOL_ZERO      = 1e-12
# when merging vertices, consider vertices closer than this
# to be the same vertex. this is the same value (1e-8)
# as the one solidworks uses (according to their documentation)
TOL_MERGE     = 1e-8
# if a point is within this distance to a plane, consider it on
# the plane
TOL_PLANAR    = 1e-5
# the maximum squared radius two faces can form and still be
# considered part of a facet (also known as coplanar)
TOL_FACET_RSQ = 1e8

import logging as _logging
log = _logging.getLogger('trimesh')
log.addHandler(_logging.NullHandler())

def log_time(method):
    def timed(*args, **kwargs):
        tic    = time_function()
        result = method(*args, **kwargs)
        log.debug('%s executed in %.4f seconds.',
                   method.__name__,
                   time_function()-tic)
        return result
    timed.__name__ = method.__name__
    timed.__doc__  = method.__doc__
    return timed

class MeshError(Exception): pass
class TransformError(Exception): pass
