"""
raster.py
------------

Turn 2D vector paths into raster images, using `pillow`
"""
import numpy as np

try:
    # keep pillow as a soft dependency
    from PIL import (Image,
                     ImageDraw,
                     ImageChops)
except BaseException as E:
    from .. import exceptions
    # re-raise the useful exception when called
    closure = exceptions.closure(E)
    Image = closure
    ImageDraw = closure
    ImageChops = closure


def rasterize(path,
              pitch,
              origin,
              resolution=None,
              fill=True,
              width=None):
    """
    Rasterize a Path2D object into a boolean image ("mode 1").

    Parameters
    ------------
    path:       Path2D object
    pitch:      float, length in model space of a pixel edge
    origin:     (2,) float, origin position in model space
    resolution: (2,) int, resolution in pixel space
    fill:       bool, if True will return closed regions as filled
    width:      int, if not None will draw outline this wide (pixels)

    Returns
    ------------
    raster: PIL.Image object, mode 1
    """

    # check inputs
    pitch = float(pitch)
    origin = np.asanyarray(origin, dtype=np.float64)

    # if resolution is None make it larger than path
    if resolution is None:
        span = np.vstack((path.bounds,
                          origin)).ptp(
                              axis=0)
        resolution = np.ceil(span / pitch) + 2
    resolution = np.asanyarray(resolution,
                               dtype=np.int64)
    resolution = tuple(resolution.tolist())

    # convert all discrete paths to pixel space
    discrete = [((i - origin) / pitch).astype(np.int)
                for i in path.discrete]

    # the path indexes that are exteriors
    # needed to know what to fill/empty but expensive
    roots = path.root

    # draw the exteriors
    exteriors = Image.new(mode='1', size=resolution)
    edraw = ImageDraw.Draw(exteriors)

    # if a width is specified draw the outline
    if width is not None:
        width = int(width)
        for coords in discrete:
            edraw.line(coords.flatten().tolist(),
                       fill=1,
                       width=width)
        # if we are not filling the polygon exit
        if not fill:
            del edraw
            return exteriors

    # draw the interiors
    interiors = Image.new(mode='1', size=resolution)
    idraw = ImageDraw.Draw(interiors)
    for i, points in enumerate(discrete):
        # draw the polygon on either the exterior or
        # interior image buffer
        if i in roots:
            edraw.polygon(points.flatten().tolist(),
                          fill=1)
        else:
            idraw.polygon(points.flatten().tolist(),
                          fill=1)
    # clean up the draw objects
    # this is in their examples, I have no idea if
    # it is actually necessary
    del edraw
    del idraw
    # the final result is the exteriors minus the interiors
    raster = ImageChops.subtract(exteriors, interiors)

    return raster
