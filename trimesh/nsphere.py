import numpy as np

try:
    from scipy import spatial
    from scipy.optimize import leastsq
except ImportError:
    log.warning('No scipy!')

def minimum_nsphere(obj):
    '''
    Compute the minimum n- sphere for a mesh or a set of points.

    Uses the fact that the minimum n- sphere will be centered at one of
    the vertices of the furthest site voronoi diagram.

    Arguments
    ----------
    obj: Trimesh object OR
         (n,d) float, set of points

    Returns
    ----------
    center: (d) float, center of n- sphere
    radius: float, radius of n-sphere
    '''
    
    if hasattr(obj, 'convex_hull_raw'):
        points = obj.convex_hull_raw.vertices
    elif hasattr(obj, 'convex_hull'):
        points = obj.convex_hull.vertices
    else:
        initial = np.asanyarray(obj)
        if len(initial.shape) != 2:
            raise ValueError('Points must be (n, dimension)!')
        hull = spatial.ConvexHull(initial)
        points = hull.points[hull.vertices]

    # if all of the points are on an n-sphere already the voronoi 
    # method will fail so we check a least squares fit before 
    # bothering to compute the voronoi diagram
    fit_C, fit_R, fit_E = fit_nsphere(points)
    if fit_E < 1e-3:
        return fit_C, fit_R

    # calculate a furthest site voronoi diagram
    # this will fail if the points are ALL on the surface of
    # the n-sphere but hopefully the least squares check caught those cases
    voronoi = spatial.Voronoi(points, furthest_site=True)

    # find the maximum radius point for each of the voronoi vertices
    # this is worst case quite expensive, but we have used quick
    # hull methods to reduce n for this operation
    r2 = np.array([((points-v)**2).sum(axis=1).max() for v in voronoi.vertices])

    center = voronoi.vertices[r2.argmin()]
    radius = np.sqrt(r2.min())

    return center, radius

def fit_nsphere(points, prior=None):
    '''
    Fit an n-sphere to a set of points using least squares. 
    
    Arguments
    ---------
    points: (n,d) set of points
    prior:  tuple of best guess for (center, radius)

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

    if prior is None:
        center_guess = np.mean(points, axis=0)
    else: 
        center_guess = prior[0]

    center_result, return_code = leastsq(residuals, center_guess)
    if not (return_code in [1,2,3,4]):
        raise ValueError('Least square fit failed!')

    radii  = np.linalg.norm(points-center_result, axis=1)
    radius = radii.mean()
    error  = radii.ptp()
    return center_result, radius, error
