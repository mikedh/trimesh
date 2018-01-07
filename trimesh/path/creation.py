from . import arc
from . import entities

import collections
import numpy as np


def circle_pattern(pattern_radius,
                   circle_radius,
                   count,
                   center=[0.0, 0.0],
                   angle=None):
    '''
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
    '''
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
                     vertices=verts)
    return pattern
