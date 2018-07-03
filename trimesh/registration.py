"""
registration.py
---------------

Functions for registering (aligning) point clouds with meshes.
"""

from . import base
from . import util
from .transformations import transform_points
import numpy as np
from scipy.spatial import KDTree  

def procrustes(X, Y, reflection=True, translation=True, scale=True, return_cost=True):
    """
    Perform Procrustes' analysis subject to constraints. Finds the 
    transformation T mapping X to Y which minimizes the sums-of-squares 
    distances between TX and Y, also called the cost.

    Parameters
    ----------
    X           : (n,3) float, list of points in space
    Y           : (n,3) float, list of points in space
    reflection  : bool, if the transformation is allowed reflections
    translation : bool, if the transformation is allowed translations
    scale       : bool, if the transformation is allowed scaling
    return_cost : bool, whether to return the cost and transformed X as well
    
    Returns
    ----------
    matrix      : (4,4) float, the transformation matrix sending X to Y
    transformed : (n,3) float, the image of X under the transformation
    cost        : float, the cost of the transformation
    """

    X = np.asanyarray(X, dtype=np.float64)
    Y = np.asanyarray(Y, dtype=np.float64)
    if not util.is_shape(X, (-1, 3)) or not util.is_shape(Y, (-1, 3)):
        raise ValueError('points must be (n,3)!')
        
    if len(X) != len(Y):
        raise ValueError('X and Y must contain same number of points!')
        
    # Remove translation component
    if translation:
        xcenter = X.mean(axis=0)
        ycenter = Y.mean(axis=0)
    else:
        xcenter = np.zeros(X.shape[1])
        ycenter = np.zeros(Y.shape[1])
        
    # Remove scale component
    if scale:
        xscale = np.sqrt(((X-xcenter)**2).sum()/len(X))
        yscale = np.sqrt(((Y-ycenter)**2).sum()/len(Y))
    else:
        xscale = 1
        yscale = 1
        
    # Use SVD to find optimal orthogonal matrix R
    # constrained to det(R) = 1 if necessary.
    u, s, vh = np.linalg.svd(((Y-ycenter)/yscale).T @ ((X-xcenter)/xscale))
    if reflection:
        R = u @ vh
    else:
        R = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
        
    # Compute our 4D transformation matrix encoding
    # X -> (R @ (X - xcenter)/xscale) * yscale + ycenter
    #    = (yscale/xscale)R @ X + (ycenter - (yscale/xscale)R @ xcenter)
    translation = ycenter - (yscale/xscale) * R @ xcenter
    matrix = np.hstack((yscale/xscale * R, translation.reshape(-1, 1)))
    matrix = np.vstack((matrix, np.array([0.]*(X.shape[1]) + [1.]).reshape(1, -1)))
    
    if return_cost:
        transformed = transform_points(X, matrix)
        cost = ((Y - transformed)**2).mean()
        return matrix, transformed, cost
    else:
        return matrix
    
def icp(X, Y, initial=np.identity(4), 
        threshold=1e-5, max_iterations=20, **kwargs):

    """
    Apply the iterative closest point algorithm to align a point cloud with 
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    X              : (n,3) float, list of points in space.
    Y              : (n,3) float or Trimesh, list of points in space or mesh.
    initial        : (4,4) float, initial transformation.
    threshold      : float, stop when change in cost is less than threshold
    max_iterations : int, maximum number of iterations
    kwargs         : dict, args to pass to procrustes
    
    Returns
    ----------
    matrix      : (4,4) float, the transformation matrix sending X to Y
    transformed : (n,3) float, the image of X under the transformation
    cost        : float, the cost of the transformation
    """
    
    X = np.asanyarray(X, dtype=np.float64)
    if not util.is_shape(X, (-1, 3)):
        raise ValueError('points must be (n,3)!')
                
    is_mesh = isinstance(Y, base.Trimesh)
    if not is_mesh:
        Y = np.asanyarray(Y, dtype=np.float64)
        if not util.is_shape(Y, (-1, 3)):
            raise ValueError('points must be (n,3)!')
        if len(X) != len(Y):
            raise ValueError('X and Y must contain same number of points!')
        ytree = KDTree(Y)

    # Transform X under initial_transformation
    X = transform_points(X, initial) 
    total_matrix = initial
    
    n_iteration = 0
    old_cost = np.inf
    while n_iteration < max_iterations:
        n_iteration += 1
        
        # Closest point in Y to each x in X
        if is_mesh:
            closest, distance, faces = Y.nearest.on_surface(X)
        else:
            distances, ix = ytree.query(X, 1)
            closest = Y[ix]
        
        # Align X with closest
        matrix, transformed, cost = procrustes(X, closest, **kwargs)

        # Update X 
        X = transformed
        total_matrix =  matrix @ total_matrix
        
        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost
         
    return total_matrix, transformed, cost
            
        
        