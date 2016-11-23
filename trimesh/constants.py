from time import time as time_function
from logging import getLogger as _getLogger
from logging import NullHandler as _NullHandler

# numerical tolerances for meshes


class NumericalToleranceMesh(object):
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

    def __init__(self, **kwargs):
        self.zero = 1e-12
        self.merge = 1e-8
        self.planar = 1e-5
        self.facet_rsq = 1e8
        self.fit = 1e-2
        self.id_len = 6
        self.__dict__.update(kwargs)


class NumericalResolutionMesh(object):
    '''
    res.mesh: when meshing solids, what distance to use
    '''

    def __init__(self, **kwargs):
        self.mesh = 5e-3

        self.__dict__.update(kwargs)

tol = NumericalToleranceMesh()
res = NumericalResolutionMesh()

# numerical tolerances for paths


class NumericalTolerancePath(object):
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

    def __init__(self, **kwargs):
        # default values
        self.zero = 1e-12
        self.merge = 1e-5
        self.planar = 1e-5
        self.buffer = .05
        self.seg_frac = .125
        self.seg_angle = .8
        self.aspect_frac = .1
        self.radius_frac = 1e-2
        self.radius_min = 1e-4
        self.radius_max = 50
        self.tangent = .017

        self.__dict__.update(kwargs)


class NumericalResolutionPath(object):
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

    def __init__(self, **kwargs):
        self.seg_frac = .05
        self.seg_angle = .08
        self.max_sections = 10
        self.min_sections = 5
        self.export = '.5f'

tol_path = NumericalTolerancePath()
res_path = NumericalResolutionPath()

# logging
log = _getLogger('trimesh')
log.addHandler(_NullHandler())


def _log_time(method):
    def timed(*args, **kwargs):
        tic = time_function()
        result = method(*args, **kwargs)
        log.debug('%s executed in %.4f seconds.',
                  method.__name__,
                  time_function() - tic)
        return result
    timed.__name__ = method.__name__
    timed.__doc__ = method.__doc__
    return timed

# exceptions


class MeshError(Exception):
    pass


class TransformError(Exception):
    pass
