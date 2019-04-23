from . import arc
from . import entities

from .. import util
from .. import transformations

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
    pattern_radius : float
      Radius of circle centers
    circle_radius : float
      The radius of each circle
    count : int
      Number of circles in the pattern
    center : (2,) float
      Center of pattern
    angle :  float
      If defined pattern will span this angle
      If None, pattern will be evenly spaced

    Returns
    -------------
    pattern : trimesh.path.Path2D
      Path containing circular pattern
    """
    from .path import Path2D

    if angle is None:
        angles = np.linspace(0.0, np.pi * 2.0, count + 1)[:-1]
    elif isinstance(angle, float) or isinstance(angle, int):
        angles = np.linspace(0.0, angle, count)
    else:
        raise ValueError('angle must be float or int!')

    # centers of circles
    centers = np.column_stack((
        np.cos(angles), np.sin(angles))) * pattern_radius

    vert = []
    ents = []
    for circle_center in centers:
        # (3,3) center points of arc
        three = arc.to_threepoint(angles=[0, np.pi],
                                  center=circle_center,
                                  radius=circle_radius)
        # add a single circle entity
        ents.append(
            entities.Arc(
                points=np.arange(3) + len(vert),
                closed=True))
        # keep flat array by extend instead of append
        vert.extend(three)

    # translate vertices to pattern center
    vert = np.array(vert) + center
    pattern = Path2D(entities=ents,
                     vertices=vert,
                     **kwargs)
    return pattern


def circle(radius=None, center=None, **kwargs):
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

    if center is None:
        center = [0.0, 0.0]
    else:
        center = np.asanyarray(center, dtype=np.float64)
    if radius is None:
        radius = 1.0
    else:
        radius = float(radius)

    # (3, 2) float, points on arc
    three = arc.to_threepoint(angles=[0, np.pi],
                              center=center,
                              radius=radius) + center

    result = Path2D(entities=[entities.Arc(points=np.arange(3), closed=True)],
                    vertices=three,
                    **kwargs)
    return result


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

    # bounds are extents, re- shape to origin- centered rectangle
    if bounds.shape == (2,):
        half = np.abs(bounds) / 2.0
        bounds = np.array([-half, half])

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


def box_outline(extents=None, transform=None, **kwargs):
    """
    Return a cuboid.

    Parameters
    ------------
    extents : float, or (3,) float
      Edge lengths
    transform: (4, 4) float
      Transformation matrix
    **kwargs:
        passed to Trimesh to create box

    Returns
    ------------
    geometry : trimesh.Path3D
      Path outline of a cuboid geometry
    """
    from .exchange.load import load_path

    # create vertices for the box
    vertices = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    vertices = np.array(vertices,
                        order='C',
                        dtype=np.float64).reshape((-1, 3))
    vertices -= 0.5

    # resize the vertices based on passed size
    if extents is not None:
        extents = np.asanyarray(extents, dtype=np.float64)
        if extents.shape != (3,):
            raise ValueError('Extents must be (3,)!')
        vertices *= extents

    # apply transform if passed
    if transform is not None:
        vertices = transformations.transform_points(vertices, transform)

    # vertex indices
    indices = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 0, 2, 6, 7, 3, 1, 5]
    outline = load_path(vertices[indices])

    return outline
