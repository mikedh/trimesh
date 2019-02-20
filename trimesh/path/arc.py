import numpy as np

from .. import util

from ..constants import log
from ..constants import tol_path as tol
from ..constants import res_path as res
from .intersections import line_line


def arc_center(points):
    """
    Given three points on an arc find:
    center, radius, normal, and angle.

    This uses the fact that the intersection of the perp
    bisectors of the segments between the control points
    is the center of the arc.

    Parameters
    ---------
    points : (3, dimension) float
      Points in space, where dimension is either 2 or 3

    Returns
    ---------
    result : dict
      Has keys:
        'center':   (d,) float, cartesian center of the arc
        'radius':   float, radius of the arc
        'normal':   (3,) float, the plane normal.
        'angle':    (2,) float, angle of start and end, in radians
        'span' :    float, angle swept by the arc, in radians
    """
    # it's a lot easier to treat 2D as 3D with a zero Z value
    points, is_2D = util.stack_3D(points, return_2D=True)

    # find the two edge vectors of the triangle
    edge_direction = np.diff(points, axis=0)
    edge_midpoints = (edge_direction * 0.5) + points[:2]

    # three points define a plane, so we find its normal vector
    plane_normal = np.cross(*edge_direction[::-1])
    plane_normal /= np.linalg.norm(plane_normal)

    # unit vector along edges
    vector_edge = (edge_direction /
                   np.linalg.norm(edge_direction, axis=1).reshape((-1, 1)))

    # perpendicular cector to each segment
    vector_perp = np.cross(vector_edge, plane_normal)
    vector_perp /= np.linalg.norm(vector_perp, axis=1).reshape((-1, 1))

    # run the line- line intersection to find the point
    intersects, center = line_line(origins=edge_midpoints,
                                   directions=vector_perp,
                                   plane_normal=plane_normal)

    if not intersects:
        raise ValueError('Segments do not intersect!')

    # radius is euclidean distance
    radius = ((points[0] - center) ** 2).sum() ** .5

    # vectors from points on arc to center point
    vector = points - center
    vector /= np.linalg.norm(vector, axis=1).reshape((-1, 1))

    angle = np.arccos(np.clip(np.dot(*vector[[0, 2]]), -1.0, 1.0))
    large_arc = (abs(angle) > tol.zero and
                 np.dot(*edge_direction) < 0.0)
    if large_arc:
        angle = (np.pi * 2) - angle

    angles = np.arctan2(*vector[:, :2].T[::-1]) + np.pi * 2
    angles_sorted = np.sort(angles[[0, 2]])
    reverse = angles_sorted[0] < angles[1] < angles_sorted[1]
    angles_sorted = angles_sorted[::(1 - int(not reverse) * 2)]

    result = {'center': center[:(3 - is_2D)],
              'radius': radius,
              'normal': plane_normal,
              'span': angle,
              'angles': angles_sorted}
    return result


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
    center_info = arc_center(points)
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
    if not close:
        arc_dist = np.linalg.norm(points[[0, -1]] -
                                  discrete[[0, -1]], axis=1)
        arc_ok = (arc_dist < tol.merge).all()
        if not arc_ok:
            log.warn(
                'failed to discretize arc (endpoint distance %s)',
                str(arc_dist))
            log.warn('Failed arc points: %s', str(points))
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
    # turn angles into (3,2) points
    three = np.column_stack((np.cos(angles),
                             np.sin(angles))) * radius
    three += center

    return three
