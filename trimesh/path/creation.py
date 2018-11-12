from . import arc
from . import entities

from .. import util

import collections
import numpy as np


def circle_pattern(pattern_radius,
                   circle_radius,
                   count,
                   center=[0.0, 0.0],
                   angle=None,
                   **kwargs):
    """
    Create a Path2D representing a circle pattern.

    Parameters
    ------------
    pattern_radius: float, radius of circle centers
    circle_radius:  float, the radius of each circle
    count:          int, number of circles in the pattern
    center:         (2,) float, center of pattern
    angle:          float, if defined pattern will span this angle
                           if not pattern will be evenly spaced

    Returns
    -------------
    pattern: Path2D object
    """
    from .path import Path2D

    if angle is None:
        angles = np.linspace(0.0, np.pi * 2.0, count + 1)[:-1]
    elif isinstance(angle, float) or isinstance(angle, int):
        angles = np.linspace(0.0, angle, count)
    else:
        raise ValueError('angle must be float or int!')

    centers = np.column_stack((np.cos(angles),
                               np.sin(angles))) * pattern_radius

    verts = collections.deque()
    ents = collections.deque()
    for circle_center in centers:
        # (3,3) center points of arc
        three = arc.to_threepoint(angles=[0, np.pi],
                                  center=circle_center,
                                  radius=circle_radius)
        ents.append(entities.Arc(points=np.arange(3) + len(verts),
                                 closed=True))
        # keep flat array by extend instead of append
        verts.extend(three)

    # translate vertices to center
    verts = np.array(verts) + center
    pattern = Path2D(entities=ents,
                     vertices=verts,
                     **kwargs)
    return pattern


def rectangle(bounds, **kwargs):
    """
    Create a Path2D containing a single or multiple rectangles
    with the specified bounds.

    Parameters
    --------------
    bounds : (2, 2) float, or (m, 2, 2) float
      Minimum XY, Maximum XY

    Returns
    -------------
    rect : Path2D
      Path containing specified rectangles
    """
    from .path import Path2D

    # data should be float
    bounds = np.asanyarray(bounds, dtype=np.float64)

    # should have one bounds or multiple bounds
    if not (util.is_shape(bounds, (2, 2)) or
            util.is_shape(bounds, (-1, 2, 2))):
        raise ValueError('bounds must be (m, 2, 2) or (2, 2)')

    # hold entities.Line objects
    lines = []
    # hold (n, 2) cartesian points
    vertices = []

    # loop through each rectangle
    for lower, upper in bounds.reshape((-1, 2, 2)):
        lines.append(entities.Line((np.arange(5) % 4) + len(vertices)))
        vertices.extend([lower,
                         [upper[0], lower[1]],
                         upper,
                         [lower[0], upper[1]]])

    # create the Path2D with specified rectangles
    rect = Path2D(entities=lines,
                  vertices=vertices,
                  **kwargs)

    return rect
