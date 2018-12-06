"""
nsphere.py
--------------

Functions for fitting and minimizing nspheres:
circles, spheres, hyperspheres, etc.
"""
import numpy as np

from . import convex

from .constants import log, tol

try:
    from scipy import spatial
    from scipy.optimize import leastsq
except ImportError:
    log.warning('No scipy!')

# if we get a MemoryError change this to True
# so we don't keep trying tile operations
_LOWMEM = False


def minimum_nsphere(obj):
    """
    Compute the minimum n- sphere for a mesh or a set of points.

    Uses the fact that the minimum n- sphere will be centered at one of
    the vertices of the furthest site voronoi diagram, which is n*log(n)
    but should be pretty fast due to using the scipy/qhull implementations
    of convex hulls and voronoi diagrams.

    Parameters
    ----------
    obj: Trimesh object OR
         (n,d) float, set of points

    Returns
    ----------
    center: (d) float, center of n- sphere
    radius: float, radius of n-sphere
    """
    # reduce the input points or mesh to the vertices of the convex hull
    # since we are computing the furthest site voronoi diagram this reduces
    # the input complexity substantially and returns the same value
    points = convex.hull_points(obj)

    # we are scaling the mesh to a unit cube
    # this used to pass qhull_options 'QbB' to Voronoi however this had a bug somewhere
    # to avoid this we scale to a unit cube ourselves inside this function
    points_origin = points.min(axis=0)
    points_scale = points.ptp(axis=0).min()
    points = (points - points_origin) / points_scale

    # if all of the points are on an n-sphere already the voronoi
    # method will fail so we check a least squares fit before
    # bothering to compute the voronoi diagram
    fit_C, fit_R, fit_E = fit_nsphere(points)
    # return fit radius and center to global scale
    fit_R = (((points - fit_C)**2).sum(axis=1).max() ** .5) * points_scale
    fit_C = (fit_C * points_scale) + points_origin

    if fit_E < 1e-6:
        log.debug('Points were on an n-sphere, returning fit')
        return fit_C, fit_R

    # calculate a furthest site voronoi diagram
    # this will fail if the points are ALL on the surface of
    # the n-sphere but hopefully the least squares check caught those cases
    # , qhull_options='QbB Pp')
    voronoi = spatial.Voronoi(points, furthest_site=True)

    # if we've gotten a MemoryError before don't try tiling
    global _LOWMEM

    # find the maximum radius^2 point for each of the voronoi vertices
    # this is worst case quite expensive, but we have used quick convex
    # hull methods to reduce n for this operation
    # we are doing comparisons on the radius squared then rooting once
    if _LOWMEM or (len(points) * len(voronoi.vertices)) > 1e7:
        # if we have a bajillion points loop
        radii_2 = np.array([((points - v)**2).sum(axis=1).max()
                            for v in voronoi.vertices])
    else:
        # otherwise tiling is massively faster
        try:
            dim = points.shape[1]
            v_tile = np.tile(voronoi.vertices, len(points)).reshape((-1, dim))
            # tile points per voronoi vertex
            p_tile = np.tile(
                points, (len(
                    voronoi.vertices), 1)).reshape(
                (-1, dim))
            # find the maximum radius of points for each voronoi vertex
            radii_2 = ((p_tile - v_tile)**2).sum(axis=1).reshape(
                (len(voronoi.vertices), -1)).max(axis=1)
        except MemoryError:
            # don't try tiling again
            _LOWMEM = True
            # fall back to the list comprehension
            radii_2 = np.array([((points - v)**2).sum(axis=1).max()
                                for v in voronoi.vertices])
            # log the MemoryError
            log.warning('MemoryError: falling back to slower check!')

    # we want the smallest sphere, so we take the min of the radii options
    radii_idx = radii_2.argmin()

    # return voronoi radius and center to global scale
    radius_v = np.sqrt(radii_2[radii_idx]) * points_scale
    center_v = (voronoi.vertices[radii_idx] * points_scale) + points_origin

    if radius_v > fit_R:
        return fit_C, fit_R
    return center_v, radius_v


def fit_nsphere(points, prior=None):
    """
    Fit an n-sphere to a set of points using least squares.

    Parameters
    ---------
    points: (n,d) set of points
    prior:  (d,) float, best guess for center of nsphere

    Returns
    ---------
    center: (d), location of center
    radius: float, mean radius across circle
    error:  float, peak to peak value of deviation from mean radius
    """
    points = np.asanyarray(points, dtype=np.float64)
    ones = np.ones(points.shape[1])

    def residuals(center):
        # do the axis sum with a dot
        radii_sq = np.dot((points - center) ** 2, ones)
        return radii_sq - radii_sq.mean()

    if prior is None:
        center_guess = points.mean(axis=0)
    else:
        center_guess = np.asanyarray(prior)

    center_result, return_code = leastsq(residuals,
                                         center_guess,
                                         xtol=1e-8)

    if not (return_code in [1, 2, 3, 4]):
        raise ValueError('Least square fit failed!')

    radii = np.linalg.norm(points - center_result, axis=1)
    radius = radii.mean()
    error = radii.ptp()
    return center_result, radius, error


def is_nsphere(points):
    """
    Check if a list of points is an nsphere.

    Parameters
    -----------
    points: (n,dimension) float, points in space

    Returns
    -----------
    check: bool, True if input points are on an nsphere
    """
    center, radius, error = fit_nsphere(points)
    check = error < tol.merge
    return check
