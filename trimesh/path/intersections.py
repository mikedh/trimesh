import numpy as np

from ..geometry import unitize 
from .util      import three_dimensionalize, euclidean
from .constants import TOL_ZERO
    
def line_line(origins, directions):
    '''
    Find the intersection between two lines. 
    Uses terminology from:
    http://geomalgorithms.com/a05-_intersect-1.html

    P(s) = p_0 + sU
    Q(t) = q_0 + tV


    Arguments
    ---------
    origins:    (2,d) list of points on lines (d in [2,3])
    directions: (2,d) list of direction vectors

    Returns
    ---------
    intersects:   boolean, whether the lines intersect. 
                  In 2D, false if the lines are parallel
                  In 3D, false if lines are not coplanar
    intersection: if intersects: (d) length point of intersection
                  else:          None
    '''

    is_2D, origins    = three_dimensionalize(origins)
    is_2D, directions = three_dimensionalize(directions)
    directions        = unitize(directions)

    if np.sum(np.abs(np.diff(directions, axis=0))) < TOL_ZERO:
        return False, None

    # the normal of the plane given by the two direction vectors
    plane_normal  = unitize(np.cross(*directions))
    # vectors perpendicular to the two lines
    perpendicular = unitize(np.cross(directions, plane_normal))
    # the vector from one origin point to the other
    origins_vector = np.diff(origins, axis=0)[0]

    # if the vector from origin to origin is on the plane given by
    # the direction vector, the dot product with the plane normal
    # should be within floating point error of zero
    coplanar = abs(np.dot(plane_normal, origins_vector)) < TOL_ZERO
    if not coplanar:
        return False, None

    s_intersection = (np.dot(-perpendicular[0], origins_vector) / 
                      np.dot( perpendicular[0], directions[1]))

    intersection = origins[0] + s_intersection*directions[0]

    return True, intersection[:(3-is_2D)]

