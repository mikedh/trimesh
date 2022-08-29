from numba import njit
import numpy as np

from .. import util

from ..constants import log
from ..constants import tol_path as tol
from ..constants import res_path as res

# floating point zero
_EPS_ZERO = 1e-10
_EPS_ONE = 1 - _EPS_ZERO
_EPS_NEG = _EPS_ZERO - 1


def arc_center_ori(points, return_normal=True, return_angle=True):
    """
    Given three points on a 2D or 3D arc find the center,
    radius, normal, and angular span.

    Parameters
    ---------
    points : (3, dimension) float
      Points in space, where dimension is either 2 or 3
    return_normal : bool
      If True calculate the 3D normal unit vector
    return_angle : bool
      If True calculate the start and stop angle and span

    Returns
    ---------
    result : dict
      Contains arc center and other keys:
       'center' : (d,) float, cartesian center of the arc
       'radius' : float, radius of the arc
       'normal' : (3,) float, the plane normal.
       'angles' : (2,) float, angle of start and end in radians
       'span'   : float, angle swept by the arc in radians
    """
    points = np.asanyarray(points, dtype=np.float64)

    # get the non-unit vectors of the three points
    vectors = points[[2, 0, 1]] - points[[1, 2, 0]]
    # we need both the squared row sum and the non-squared
    abc2 = np.dot(vectors ** 2, [1] * points.shape[1])
    # same as np.linalg.norm(vectors, axis=1)
    abc = np.sqrt(abc2)

    # perform radius calculation scaled to shortest edge
    # to avoid precision issues with small or large arcs
    scale = abc.min()
    # get the edge lengths scaled to the smallest
    edges = abc / scale
    # half the total length of the edges
    half = edges.sum() / 2.0
    # check the denominator for the radius calculation
    denom = half * np.product(half - edges)
    if denom < 1e-8:
        raise ValueError('arc is colinear!')
    # find the radius and scale back after the operation
    radius = scale * ((np.product(edges) / 4.0) / np.sqrt(denom))

    # use a barycentric approach to get the center
    ba2 = (abc2[[1, 2, 0, 0, 2, 1, 0, 1, 2]] *
           [1, 1, -1, 1, 1, -1, 1, 1, -1]).reshape(
               (3, 3)).sum(axis=1) * abc2
    center = points.T.dot(ba2) / ba2.sum()

    if tol.strict:
        # all points should be at the calculated radius from center
        assert util.allclose(
            np.linalg.norm(points - center, axis=1),
            radius)

    # start with initial results
    result = {'center': center,
              'radius': radius}

    if return_normal:
        if points.shape == (3, 2):
            # for 2D arcs still use the cross product so that
            # the sign of the normal vector is consistent
            result['normal'] = util.unitize(
                np.cross(np.append(-vectors[1], 0),
                         np.append(vectors[2], 0)))
        else:
            # otherwise just take the cross product
            result['normal'] = util.unitize(
                np.cross(-vectors[1], vectors[2]))

    if return_angle:
        # vectors from points on arc to center point
        vector = util.unitize(points - center)
        # find the angle between the first and last vector
        dot = np.dot(*vector[[0, 2]])
        if dot < _EPS_NEG:
            angle = np.pi
        elif dot > _EPS_ONE:
            angle = 0.0
        else:
            angle = np.arccos(dot)
        # if the angle is nonzero and vectors are opposite direction
        # it means we have a long arc rather than the short path
        edge_direction = np.diff(points, axis=0)

        edot = np.dot(points[1] - points[0], points[2] - points[1])

        assert np.isclose(np.dot(*edge_direction), edot)

        if abs(angle) > _EPS_ZERO and np.dot(*edge_direction) < 0.0:
            angle = (np.pi * 2) - angle
        # convoluted angle logic
        angles = np.arctan2(*vector[:, :2].T[::-1]) + np.pi * 2
        angles_sorted = np.sort(angles[[0, 2]])
        reverse = angles_sorted[0] < angles[1] < angles_sorted[1]
        angles_sorted = angles_sorted[::(1 - int(not reverse) * 2)]
        #result['angles'] = angles_sorted
        result['span'] = angle

    return result


def arc_center(points, return_normal=True, return_angle=True):
    a = arc_center_njit(points, return_normal, return_angle)
    b = arc_center_ori(points, return_normal, return_angle)

    if set(a.keys()) != set(b.keys()):
        from IPython import embed
        embed()

    if not all(np.allclose(a[k], b[k])
               for k in a.keys()):
        #from IPython import embed
        # embed()
        pass
    return a


def arc_center_njit(points, return_normal=True, return_angle=True):
    points = np.array(points, dtype=np.float64)
    c, r, a = _arc_center(points)
    result = {'center': c, 'radius': r}
    if return_angle:
        result['span'] = a

    if return_normal:
        # get the non-unit vectors of the three points
        vectors = points[[2, 0, 1]] - points[[1, 2, 0]]
        if points.shape == (3, 2):
            # for 2D arcs still use the cross product so that
            # the sign of the normal vector is consistent
            result['normal'] = util.unitize(
                np.cross(np.append(-vectors[1], 0),
                         np.append(vectors[2], 0)))
        else:
            # otherwise just take the cross product
            result['normal'] = util.unitize(
                np.cross(-vectors[1], vectors[2]))
    return result


@njit
def _arc_center(points):
    """
    Given three points on a 2D or 3D arc find the center,
    radius, normal, and angular span.

    Parameters
    ---------
    points : (3, dimension) float
      Points in space, where dimension is either 2 or 3
    return_normal : bool
      If True calculate the 3D normal unit vector
    return_angle : bool
      If True calculate the start and stop angle and span

    Returns
    ---------
    result : dict
      Contains arc center and other keys:
       'center' : (d,) float, cartesian center of the arc
       'radius' : float, radius of the arc
       'normal' : (3,) float, the plane normal.
       'angles' : (2,) float, angle of start and end in radians
       'span'   : float, angle swept by the arc in radians
    """

    idx_diff = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int32)

    # get the non-unit vectors of the three points
    vectors = points[idx_diff[0]] - points[idx_diff[1]]

    # get the non-unit vectors of the three points

    # we need both the squared row sum and the non-squared
    abc2 = (vectors ** 2).sum(axis=1)
    # same as np.linalg.norm(vectors, axis=1)
    abc = np.sqrt(abc2)

    # perform radius calculation scaled to shortest edge
    # to avoid precision issues with small or large arcs
    scale = abc.min()
    # get the edge lengths scaled to the smallest
    edges = abc / scale
    # half the total length of the edges
    half = edges.sum() / 2.0
    # check the denominator for the radius calculation
    #denom = half * np.product(half - edges)

    he = half - edges
    denom = half * he[0] * he[1] * he[2]

    edges_prod = edges[0] * edges[1] * edges[2]

    if denom < _EPS_ZERO:
        raise ValueError('arc is colinear!')
    # find the radius and scale back after the operation
    radius = scale * ((edges_prod / 4.0) / np.sqrt(denom))

    idx_ba2 = np.array([1, 2, 0, 0, 2, 1, 0, 1, 2], dtype=np.int32)
    signs = np.array([1, 1, -1, 1, 1, -1, 1, 1, -1], dtype=np.float64)
    # use a barycentric approach to get the center
    ba2 = (abc2[idx_ba2] * signs).reshape(
        (3, 3)).sum(axis=1) * abc2
    center = points.T.dot(ba2) / ba2.sum()

    # vectors from points on arc to center point
    vector = points - center
    for i, v in enumerate(vectors):
        vectors[i] = v / np.linalg.norm(v)

    # find the angle between the first and last vector
    dot = np.dot(vector[0], vector[2])

    if dot < _EPS_NEG:
        angle = np.pi
    elif dot > _EPS_ONE:
        angle = 0.0
    else:
        angle = np.arccos(dot)

    vec = np.array([[points[1] - points[0]],
                    [points[2] - points[1]]])
    edge_dot = np.dot(vec[0] / np.linalg.norm(vec[0]),
                      vec[1] / np.linalg.norm(vec[1]))
    # if the angle is nonzero and vectors are opposite direction
    # it means we have a long arc rather than the short path
    if abs(angle) > -_EPS_ZERO and edge_dot < _EPS_ZERO:
        angle = (np.pi * 2.0) - angle
        # pass

    return center, radius, angle


def discretize_arc(points,
                   close=False,
                   scale=1.0):
    """
    Returns a version of a three point arc consisting of
    line segments.

    Parameters
    ---------
    points : (3, d) float
      Points on the arc where d in [2,3]
    close :  boolean
      If True close the arc into a circle
    scale : float
      What is the approximate overall drawing scale
      Used to establish order of magnitude for precision

    Returns
    ---------
    discrete : (m, d) float
      Connected points in space
    """
    # make sure points are (n, 3)
    points, is_2D = util.stack_3D(points, return_2D=True)
    # find the center of the points
    try:
        # try to find the center from the arc points
        center_info = arc_center(points)
    except BaseException:
        # if we hit an exception return a very bad but
        # technically correct discretization of the arc
        if is_2D:
            return points[:, :2]
        return points

    center, R, N, angle = (center_info['center'],
                           center_info['radius'],
                           center_info['normal'],
                           center_info['span'])

    # if requested, close arc into a circle
    if close:
        angle = np.pi * 2

    # the number of facets, based on the angle criteria
    count_a = angle / res.seg_angle
    count_l = ((R * angle)) / (res.seg_frac * scale)

    # figure out the number of line segments
    count = np.max([count_a, count_l])
    # force at LEAST 4 points for the arc
    # otherwise the endpoints will diverge
    count = np.clip(count, 4, np.inf)
    count = int(np.ceil(count))

    V1 = util.unitize(points[0] - center)
    V2 = util.unitize(np.cross(-N, V1))
    t = np.linspace(0, angle, count)

    discrete = np.tile(center, (count, 1))
    discrete += R * np.cos(t).reshape((-1, 1)) * V1
    discrete += R * np.sin(t).reshape((-1, 1)) * V2

    # do an in-process check to make sure result endpoints
    # match the endpoints of the source arc
    if tol.strict and not close:
        arc_dist = util.row_norm(points[[0, -1]] - discrete[[0, -1]])
        arc_ok = (arc_dist < tol.merge).all()
        if not arc_ok:
            log.warning(
                'failed to discretize arc (endpoint_distance=%s R=%s)',
                str(arc_dist), R)
            log.warning('Failed arc points: %s', str(points))
            raise ValueError('Arc endpoints diverging!')
    discrete = discrete[:, :(3 - is_2D)]

    return discrete


def to_threepoint(center, radius, angles=None):
    """
    For 2D arcs, given a center and radius convert them to three
    points on the arc.

    Parameters
    -----------
    center : (2,) float
      Center point on the plane
    radius : float
      Radius of arc
    angles : (2,) float
      Angles in radians for start and end angle
      if not specified, will default to (0.0, pi)

    Returns
    ----------
    three : (3, 2) float
      Arc control points
    """
    # if no angles provided assume we want a half circle
    if angles is None:
        angles = [0.0, np.pi]
    # force angles to float64
    angles = np.asanyarray(angles, dtype=np.float64)
    if angles.shape != (2,):
        raise ValueError('angles must be (2,)!')
    # provide the wrap around
    if angles[1] < angles[0]:
        angles[1] += np.pi * 2

    center = np.asanyarray(center, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError('only valid on 2D arcs!')

    # turn the angles of [start, end]
    # into [start, middle, end]
    angles = np.array([angles[0],
                       angles.mean(),
                       angles[1]],
                      dtype=np.float64)
    # turn angles into (3, 2) points
    three = (np.column_stack(
        (np.cos(angles),
         np.sin(angles))) * radius) + center

    return three
