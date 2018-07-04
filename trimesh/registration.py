"""
registration.py
---------------

Functions for registering (aligning) point clouds with meshes.
"""

import numpy as np

from scipy.spatial import cKDTree

from . import util
from .transformations import transform_points


def procrustes(a,
               b,
               reflection=True,
               translation=True,
               scale=True,
               return_cost=True):
    """
    Perform Procrustes' analysis subject to constraints. Finds the
    transformation T mapping a to b which minimizes the sums-of-squares
    distances between Ta and b, also called the cost.

    Parameters
    ----------
    a           : (n,3) float, list of points in space
    b           : (n,3) float, list of points in space
    reflection  : bool, if the transformation is allowed reflections
    translation : bool, if the transformation is allowed translations
    scale       : bool, if the transformation is allowed scaling
    return_cost : bool, whether to return the cost and transformed a as well

    Returns
    ----------
    matrix      : (4,4) float, the transformation matrix sending a to b
    transformed : (n,3) float, the image of a under the transformation
    cost        : float, the cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)) or not util.is_shape(b, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    if len(a) != len(b):
        raise ValueError('a and b must contain same number of points!')

    # Remove translation component
    if translation:
        acenter = a.mean(axis=0)
        bcenter = b.mean(axis=0)
    else:
        acenter = np.zeros(a.shape[1])
        bcenter = np.zeros(b.shape[1])

    # Remove scale component
    if scale:
        ascale = np.sqrt(((a - acenter)**2).sum() / len(a))
        bscale = np.sqrt(((b - bcenter)**2).sum() / len(b))
    else:
        ascale = 1
        bscale = 1

    # Use SVD to find optimal orthogonal matrix R
    # constrained to det(R) = 1 if necessary.
    u, s, vh = np.linalg.svd(
        np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale)))
    if reflection:
        R = np.dot(u, vh)
    else:
        R = np.dot(np.dot(u, np.diag(
            [1, 1, np.linalg.det(np.dot(u, vh))])), vh)

    # Compute our 4D transformation matrix encoding
    # a -> (R @ (a - acenter)/ascale) * bscale + bcenter
    #    = (bscale/ascale)R @ a + (bcenter - (bscale/ascale)R @ acenter)
    translation = bcenter - (bscale / ascale) * np.dot(R, acenter)
    matrix = np.hstack((bscale / ascale * R, translation.reshape(-1, 1)))
    matrix = np.vstack(
        (matrix, np.array([0.] * (a.shape[1]) + [1.]).reshape(1, -1)))

    if return_cost:
        transformed = transform_points(a, matrix)
        cost = ((b - transformed)**2).mean()
        return matrix, transformed, cost
    else:
        return matrix


def icp(a,
        b,
        initial=np.identity(4),
        threshold=1e-5,
        max_iterations=20,
        **kwargs):
    """
    Apply the iterative closest point algorithm to align a point cloud with
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    a              : (n,3) float, list of points in space.
    b              : (m,3) float or Trimesh, list of points in space or mesh.
    initial        : (4,4) float, initial transformation.
    threshold      : float, stop when change in cost is less than threshold
    max_iterations : int, maximum number of iterations
    kwargs         : dict, args to pass to procrustes

    Returns
    ----------
    matrix      : (4,4) float, the transformation matrix sending a to b
    transformed : (n,3) float, the image of a under the transformation
    cost        : float, the cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    is_mesh = util.is_instance_named(b, 'Trimesh')
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError('points must be (n,3)!')
        btree = cKDTree(b)

    # Transform a under initial_transformation
    a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for n_iteration in range(max_iterations):
        # Closest point in b to each point in a
        if is_mesh:
            closest, distance, faces = b.nearest.on_surface(a)
        else:
            distances, ix = btree.query(a, 1)
            closest = b[ix]

        # Align a with closest points
        matrix, transformed, cost = procrustes(a, closest, **kwargs)

        # Update a
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost
