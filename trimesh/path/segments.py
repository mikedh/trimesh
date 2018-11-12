"""
segments.py
--------------

Deal with (n, 2, 3) line segments.
"""

import numpy as np

from .. import util
from .. import geometry


def segments_to_parameters(segments):
    """
    For 3D line segments defined by two points, turn
    them in to an origin defined as the closest point along
    the line to the zero origin as well as a direction vector
    and start and end parameter.

    Parameters
    ------------
    segments : (n, 2, 3) float
       Line segments defined by start and end points

    Returns
    --------------
    origins : (n, 3) float
       Point on line closest to [0, 0, 0]
    vectors : (n, 3) float
       Unit line directions
    parameters : (n, 2) float
       Start and end distance pairs for each line
    """
    segments = np.asanyarray(segments, dtype=np.float64)
    if not util.is_shape(segments, (-1, 2, (2, 3))):
        raise ValueError('incorrect segment shape!',
                         segments.shape)

    # make the initial origin one of the end points
    endpoint = segments[:, 0]
    vectors = segments[:, 1] - endpoint
    vectors_norm = np.linalg.norm(vectors, axis=1)
    vectors /= vectors_norm.reshape((-1, 1))

    # find the point along the line nearest the origin
    offset = util.diagonal_dot(endpoint, vectors)
    # points nearest [0, 0, 0] will be our new origin
    origins = endpoint + (offset.reshape((-1, 1)) * -vectors)

    # parametric start and end of line segment
    parameters = np.column_stack((offset, offset + vectors_norm))

    return origins, vectors, parameters


def parameters_to_segments(origins, vectors, parameters):
    """
    Convert a parametric line segment representation to
    a two point line segment representation

    Parameters
    ------------
    origins : (n, 3) float
       Line origin point
    vectors : (n, 3) float
       Unit line directions
    parameters : (n, 2) float
       Start and end distance pairs for each line

    Returns
    --------------
    segments : (n, 2, 3) float
       Line segments defined by start and end points
    """
    # don't copy input
    origins = np.asanyarray(origins, dtype=np.float64)
    vectors = np.asanyarray(vectors, dtype=np.float64)
    parameters = np.asanyarray(parameters, dtype=np.float64)

    # turn the segments into a reshapable 2D array
    segments = np.hstack((origins + vectors * parameters[:, :1],
                          origins + vectors * parameters[:, 1:]))

    return segments.reshape((-1, 2, origins.shape[1]))


def colinear_pairs(segments, radius=.01, angle=.01):
    """
    Find pairs of segments which are colinear.

    Parameters
    -------------
    segments : (n, 2, (2, 3)) float
      Two or three dimensional line segments
    radius : float
      Maximum radius line origins can differ
      and be considered colinear
    angle : float
      Maximum angle in radians segments can
      differ and still be considered colinear

    Returns
    ------------
    pairs : (m, 2) int
      Indexes of segments which are colinear
    """
    from scipy import spatial

    # convert segments to parameterized origins
    # which are the closest point on the line to
    # the actual zero- origin
    origins, vectors, parameters = segments_to_parameters(segments)

    # create a kdtree for origins
    tree = spatial.cKDTree(origins)

    # find origins closer than specified radius
    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    # calculate angles between pairs
    angles = geometry.vector_angle(vectors[pairs])

    # angles can be within tolerance of 180 degrees or 0.0 degrees
    angle_ok = np.logical_or(
        np.isclose(angles, np.pi, atol=angle),
        np.isclose(angles, 0.0, atol=angle))

    # check angle threshold
    colinear = pairs[angle_ok]

    return colinear
