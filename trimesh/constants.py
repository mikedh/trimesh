from time import time as time_function
from collections import namedtuple as _namedtuple
from logging import getLogger   as _getLogger
from logging import NullHandler as _NullHandler

### numerical tolerances for meshes
class _NumericalToleranceMesh(_namedtuple('NumericalToleranceMesh', 
                                         ['zero', 
                                          'merge',
                                          'planar',
                                          'facet_rsq',
                                          'fit'])):
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
class _NumericalResolutionMesh(_namedtuple('NumericalResolutionMesh', 
                                          ['mesh'])):
    '''
    res.mesh: when meshing solids, what distance to use
    '''
tol = _NumericalToleranceMesh(zero      = 1e-12,
                              merge     = 1e-8,
                              planar    = 1e-5,
                              facet_rsq = 1e8,
                              fit       = 1e-2)
res = _NumericalResolutionMesh(mesh = 5e-3)


### numerical tolerances for paths
class _NumericalTolerancePath(_namedtuple('NumericalTolerancePath', 
                                         ['zero', 
                                          'merge',
                                          'planar',
                                          'buffer',
                                          'seg_frac',
                                          'seg_angle',
                                          'aspect_frac',
                                          'radius_frac',
                                          'radius_min',
                                          'radius_max',
                                          'tangent'])):
    '''
    tol.zero: consider floating point numbers less than this zero
    tol.merge: when merging vertices, consider vertices closer than this
               to be the same vertex
    tol.planar: the maximum distance from a plane a point can be and
                still be considered to be on the plane
    tol.seg_frac: when simplifying line segments what percentage of the drawing
                  scale can a segment be and have a curve fitted
    tol.seg_angle: when simplifying line segments to arcs, what angle
                   can a segment span to be acceptable.
    tol.aspect_frac: when simplifying line segments to closed arcs (circles)
                     what percentage can the aspect ratio differfrom 1:1
                     before escaping the fit early
    tol.radius_frac: when simplifying line segments to arcs, what percentage 
                     of the fit radius can vertices deviate to be acceptable
    tol.radius_min: when simplifying line segments to arcs, what is the minimum 
                    radius multiplied by document scale for an acceptable fit
    tol.radius_max: when simplifying line segments to arcs, what is the maximum
                    radius multiplied by document scale for an acceptable fit 
    tol.tangent: when simplifying line segments to curves, what is the maximum
                 angle the end sections can deviate from tangent that is acceptable.   
    '''
class _NumericalResolutionPath(_namedtuple('NumericalResolutionPath', 
                                           ['seg_frac',
                                            'seg_angle',
                                            'max_sections',
                                            'min_sections',
                                            'export'])):
    '''
    res.seg_frac: when discretizing curves, what percentage of the drawing
                  scale should we aim to make a single segment
    res.seg_angle: when discretizing curves, what angle should a section span
    res.max_sections: when discretizing splines, what is the maximum number 
                      of segments per control point
    res.min_sections: when discretizing splines, what is the minimum number
                      of segments per control point
    res.export: format string to use when exporting floating point vertices
    '''
tol_path = _NumericalTolerancePath(zero        = 1e-12,
                                   merge       = 1e-5,
                                   planar      = 1e-5,
                                   buffer      = .05,
                                   seg_frac    = .125,
                                   seg_angle   = .25,
                                   aspect_frac = .1,
                                   radius_frac = 1e-2,
                                   radius_min  = 1e-3,
                                   radius_max  = 50,
                                   tangent     = .0175)
res_path = _NumericalResolutionPath(seg_frac     = .05,
                                    seg_angle    = .08,
                                    max_sections = 10,
                                    min_sections = 5,
                                    export       = '.5f')

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
