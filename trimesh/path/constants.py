from time import time as time_function
from collections import namedtuple as _namedtuple
from logging import getLogger   as _getLogger
from logging import NullHandler as _NullHandler

### numerical tolerances
class NumericalTolerance(_namedtuple('NumericalTolerance', 
                                     ['zero', 
                                      'merge',
                                      'planar',
                                      'seg_length',
                                      'facet_ang',
                                      'radius',
                                      'aspect'])):
    '''
    tol.zero: consider floating point numbers less than this zero
    tol.merge: when merging vertices, consider vertices closer than this
               to be the same vertex
    tol.planar: the maximum distance from a plane a point can be and
                still be considered to be on the plane
    '''
class NumericalResolution(_namedtuple('NumericalResolution', 
                                      ['length',
                                       'angle',
                                       'max_sections',
                                       'min_sections'])):
    '''
    res.length: when discretizing curves, how long should a section be
    res.angle:  when discretizing curves, what angle should a section span
    '''
tol = NumericalTolerance(zero      = 1e-12,
                         merge     = 1e-5,
                         planar    = 1e-5,
                         seg_length = .19,
                         facet_ang = .2,
                         radius    = 1e-5,
                         aspect    = .1)
res = NumericalResolution(length = .005,
                          angle  = .18,
                          max_sections = 10,
                          min_sections = 5)
_EXPORT_PRECISION = '.5f'

### logging
log = _getLogger('path')
log.addHandler(_NullHandler())
