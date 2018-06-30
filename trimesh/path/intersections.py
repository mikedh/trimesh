import numpy as np

from ..util import three_dimensionalize
from ..constants import tol_path as tol


def line_line(origins,
              directions,
              plane_normal=None):
    """
    Find the intersection between two lines.
    Uses terminology from:
    http://geomalgorithms.com/a05-_intersect-1.html

    line 1:    P(s) = p_0 + sU
    line 2:    Q(t) = q_0 + tV

    Parameters
    ---------
    origins:      (2, d) float, points on lines (d in [2,3])
    directions:   (2, d) float, direction vectors
    plane_normal: (3, ) float, if not passed computed from cross

    Returns
    ---------
    intersects:   boolean, whether the lines intersect.
                  In 2D, false if the lines are parallel
                  In 3D, false if lines are not coplanar
    intersection: if intersects: (d) length point of intersection
                  else:          None
    """
    # check so we can accept 2D or 3D points
    is_2D, origins = three_dimensionalize(origins)
    is_2D, directions = three_dimensionalize(directions)

    # unitize direction vectors
    directions /= np.linalg.norm(directions,
                                 axis=1).reshape((-1, 1))

    # exit if values are parallel
    if np.sum(np.abs(np.diff(directions,
                             axis=0))) < tol.zero:
        return False, None

    # using notation from docstring
    q_0, p_0 = origins
    v, u = directions
    w = p_0 - q_0

    # recompute plane normal if not passed
    if plane_normal is None:
        # the normal of the plane given by the two direction vectors
        plane_normal = np.cross(u, v)
        plane_normal /= np.linalg.norm(plane_normal)

    # vectors perpendicular to the two lines
    v_perp = np.cross(v, plane_normal)
    v_perp /= np.linalg.norm(v_perp)

    # if the vector from origin to origin is on the plane given by
    # the direction vector, the dot product with the plane normal
    # should be within floating point error of zero
    coplanar = abs(np.dot(plane_normal, w)) < tol.zero
    if not coplanar:
        return False, None

    # value of parameter s where intersection occurs
    s_I = (np.dot(-v_perp, w) /
           np.dot(v_perp, u))
    # plug back into the equation of the line to find the point
    intersection = p_0 + s_I * u

    return True, intersection[:(3 - is_2D)]
