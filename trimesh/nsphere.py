import numpy as np

from . import convex

from .constants import log
try:
    from scipy          import spatial
    from scipy.optimize import leastsq
except ImportError:
    log.warning('No scipy!')

def minimum_nsphere(obj):
    '''
    Compute the minimum n- sphere for a mesh or a set of points.

    Uses the fact that the minimum n- sphere will be centered at one of
    the vertices of the furthest site voronoi diagram, which is n*log(n)
    but should be pretty fast due to using the scipy/qhull implementations
    of convex hulls and voronoi diagrams.
   
    Arguments
    ----------
    obj: Trimesh object OR
         (n,d) float, set of points

    Returns
    ----------
    center: (d) float, center of n- sphere
    radius: float, radius of n-sphere
    '''
    # reduce the input points or mesh to the vertices of the convex hull
    # since we are computing the furthest site voronoi diagram this reduces
    # the input complexity substantially and returns the same value
    points = convex.hull_points(obj)

    # if all of the points are on an n-sphere already the voronoi 
    # method will fail so we check a least squares fit before 
    # bothering to compute the voronoi diagram
    fit_C, fit_R, fit_E = fit_nsphere(points)
    radius_fit = ((points - fit_C)**2).sum(axis=1).max() ** .5
    if fit_E < 1e-3:
        log.debug('Points were on an n-sphere, returning fit')
        return fit_C, fit_R

    # calculate a furthest site voronoi diagram
    # this will fail if the points are ALL on the surface of
    # the n-sphere but hopefully the least squares check caught those cases
    voronoi = spatial.Voronoi(points, furthest_site=True)

    # find the maximum radius^2 point for each of the voronoi vertices
    # this is worst case quite expensive, but we have used quick convex
    # hull methods to reduce n for this operation
    # we are doing comparisons on the radius^2 value so as to only do a sqrt once
    r2 = np.array([((points-v)**2).sum(axis=1).max() for v in voronoi.vertices])
    r2_idx = r2.argmin()
    radius_v = np.sqrt(r2[r2_idx])
    center_v = voronoi.vertices[r2_idx]

    if radius_v > radius_fit:
        return fit_C, radius_fit
    return center_v, radius_v

def fit_nsphere(points, prior=None):
    '''
    Fit an n-sphere to a set of points using least squares. 
    
    Arguments
    ---------
    points: (n,d) set of points
    prior:  (d,) float, best guess for center

    Returns
    ---------
    center: (d), location of center
    radius: float, mean radius across circle
    error:  float, peak to peak value of deviation from mean radius
    '''
    
    def residuals(center):
        radii_sq  = ((points-center)**2).sum(axis=1)
        residuals = radii_sq - radii_sq.mean()
        return residuals

    if prior is None: center_guess = np.mean(points, axis=0)
    else:             center_guess = prior

    center_result, return_code = leastsq(residuals, center_guess, gtol=1e-8)
    if not (return_code in [1,2,3,4]):
        raise ValueError('Least square fit failed!')

    radii  = np.linalg.norm(points-center_result, axis=1)
    radius = radii.mean()
    error  = radii.ptp()
    return center_result, radius, error
