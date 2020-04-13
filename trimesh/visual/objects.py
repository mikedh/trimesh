"""
objects.py
--------------

Deal with objects which hold visual properties, like
ColorVisuals and TextureVisuals.
"""
import numpy as np

from .color import ColorVisuals
from ..util import log


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

    try:
        # get the mode of the first visual
        mode = visuals[0].kind
        if mode == 'face':
            colors = np.vstack([
                v.face_colors for v in visuals])
            return ColorVisuals(face_colors=colors)
        elif mode == 'vertex':
            colors = np.vstack([
                v.vertex_colors for v in visuals])
            return ColorVisuals(vertex_colors=colors)
    except BaseException:
        log.warning('failed to concatenate visuals!',
                    exc_info=True)

    return ColorVisuals()
