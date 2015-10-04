from time import time as time_function
from collections import namedtuple as _namedtuple
from logging import getLogger   as _getLogger
from logging import NullHandler as _NullHandler

### numerical tolerances
class NumericalTolerance(_namedtuple('NumericalTolerance', 
                                     ['zero', 
                                      'merge',
                                      'planar',
                                      'facet_rsq'])):
    '''
    tol.zero: consider floating point numbers less than this zero
    tol.merge: when merging vertices, consider vertices closer than this
               to be the same vertex. Here we use the same value (1e-8)
               as SolidWorks uses, according to their documentation.
    tol.planar: the maximum distance from a plane a point can be and
                still be considered to be on the plane
    tol.facet_rsq: the minimum radius squared that an arc drawn from the 
                   center of a face to the center of an adjacent face can
                   be to consider the two faces coplanar. This method is more
                   robust than considering just normal angles as it is tolerant
                   of numerical error on very small faces. 
    '''
class NumericalResolution(_namedtuple('NumericalResolution', 
                                      ['mesh'])):
    '''
    res.mesh: when meshing parts, what resolution to use
    '''
tol = NumericalTolerance(zero      = 1e-12,
                         merge     = 1e-8,
                         planar    = 1e-5,
                         facet_rsq = 1e8)
res = NumericalResolution(mesh = 5e-3)


### logging
log = _getLogger('trimesh')
log.addHandler(_NullHandler())
def _log_time(method):
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


### exceptions
class MeshError(Exception):
    pass
class TransformError(Exception): 
    pass
