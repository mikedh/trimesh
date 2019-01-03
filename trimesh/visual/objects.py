"""
objects.py
--------------

Deal with objects which hold visual properties, like
ColorVisuals and TextureVisuals.
"""
import numpy as np
import collections

from .color import ColorVisuals


def create_visual(**kwargs):
    """
    Create Visuals object from keyword arguments.

    Parameters
    ----------
    face_colors   :   (n,3|4) uint8, colors
    vertex_colors : (n,3|4) uint8, colors
    mesh:          Trimesh object

    Returns
    ----------
    visuals: ColorVisuals object.
    """
    return ColorVisuals(**kwargs)


def concatenate(visuals, *args):
    """
    Concatenate multiple visual objects.

    Parameters
    ----------
    visuals: ColorVisuals object, or list of same
    *args:  ColorVisuals object, or list of same

    Returns
    ----------
    concat: ColorVisuals object
    """
    # get a flat list of ColorVisuals objects
    if len(args) > 0:
        visuals = np.append(visuals, args)
    else:
        visuals = np.array(visuals)

    # get the type of visuals (vertex or face) removing undefined
    modes = {v.kind for v in visuals}.difference({None})
    if len(modes) == 0:
        # none of the visuals have anything defined
        return ColorVisuals()
    else:
        # if we have visuals with different modes defined
        # arbitrarily get one of them
        mode = modes.pop()

    # a linked list to store colors before stacking
    colors = collections.deque()
    # a string to evaluate which returns the colors we want
    append = 'v.{}_colors'.format(mode)
    for v in visuals:
        # use an eval so we can use the object property
        colors.append(eval(append))
    # use an eval so we can use the constructor
    concat = eval('ColorVisuals({}_colors=np.vstack(colors))'.format(mode))
    return concat
