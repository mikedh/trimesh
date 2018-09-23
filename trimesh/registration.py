"""
registration.py
---------------

Functions for registering (aligning) point clouds with meshes.
"""

import numpy as np

from scipy.spatial import cKDTree

from . import util
from . import bounds
from . import transformations

from .transformations import transform_points


def key_points(mesh, count):
    """
    Return a combination of mesh vertices and surface samples
    with vertices chosen by likelyhood to be important to registation
    """
    stack = []
    if len(mesh.vertices) < (count / 2):
        return np.vstack((
            mesh.vertices,
            mesh.sample(count - len(mesh.vertices))))
    else:
        return mesh.sample(count)


def mesh_other(mesh, other, samples=500, icp_first=10, icp_final=50):
    """
    Align a mesh with another mesh or a PointCloud using
    the principal axes of inertia as a starting point which
    is refined by iterative closest point.

    Parameters
    ------------
    mesh : trimesh.Trimesh object
      Mesh to align with other
    other : trimesh.Trimesh or (n, 3) float
      Mesh or points in space
    samples : int
      Number of samples from mesh surface to align
    icp_first : int
      How many ICP iterations for the 9 possible
      combinations of
    icp_final : int
      How many ICP itertations for the closest
      candidate from the wider search

    Returns
    -----------
    mesh_to_other : (4, 4) float
      Transform to align mesh to the other object
    cost : float
      Average squared distance per point
    """

    if not util.is_instance_named(mesh, 'Trimesh'):
        raise ValueError('mesh must be Trimesh object!')

    inverse = True
    search = mesh
    # if both are meshes use the smaller one for searching
    if util.is_instance_named(other, 'Trimesh'):
        if len(mesh.vertices) > len(other.vertices):
            search = other
            inverse = False
            points = key_points(mesh=mesh,
                                count=samples)
            points_mesh = mesh
        else:
            points_mesh = other
            points = key_points(mesh=other,
                                count=samples)

        if points_mesh.is_volume:
            points_PIT = points_mesh.principal_inertia_transform
        else:
            points_PIT = points_mesh.bounding_box_oriented.principal_inertia_transform

    elif util.is_shape(other, (-1, 3)):
        # case where other is just points
        points = other
        points_PIT = bounds.oriented_bounds(points)[0]
    else:
        raise ValueError('other must be mesh or (n, 3) points!')

    if search.is_volume:
        search_PIT = search.principal_inertia_transform
    else:
        search_PIT = search.bounding_box_oriented.principal_inertia_transform

        # move from mesh a to mesh b
    search_to_points = np.dot(np.linalg.inv(points_PIT), search_PIT)

    # permutations of cube rotations
    # the principal inertia transform has arbitrary sign
    # along the 3 major axis so try all combinations of
    # 180 degree rotations with a quick first ICP pass
    cubes = np.array([np.eye(4) * np.append(diag, 1)
                      for diag in [[1, 1, 1],
                                   [1, 1, -1],
                                   [1, -1, 1],
                                   [-1, 1, 1],
                                   [-1, -1, 1],
                                   [-1, 1, -1],
                                   [1, -1, -1],
                                   [-1, -1, -1]]])

    #from IPython import embed
    # embed()

    # loop through permutations and run iterative closest point on each
    costs, transforms = [], []
    centroid = search.centroid
    for flip in cubes:
        a_to_b = np.dot(transformations.transform_around(flip, centroid),
                        np.linalg.inv(search_to_points))

        # import trimesh
        # vpt = trimesh.PointCloud(points)
        # vpt.apply_transform(a_to_b)
        # trimesh.Scene([search, vpt]).show()

        # run first pass ICP
        matrix, junk, cost = icp(a=points,
                                 b=search,
                                 initial=a_to_b,
                                 max_iterations=int(icp_first),
                                 scale=False)
        transforms.append(matrix)
        costs.append(cost)

    # run a final ICP refinement step
    matrix, junk, cost = icp(a=points,
                             b=search,
                             initial=transforms[np.argmin(costs)],
                             max_iterations=int(icp_final),
                             scale=False)

    # convert square sum distance to squared average distance
    cost /= len(points)

    if inverse:
        mesh_to_other = np.linalg.inv(matrix)
    else:
        mesh_to_other = matrix

    return mesh_to_other, cost


def procrustes(a,
               b,
               reflection=True,
               translation=True,
               scale=True,
               return_cost=True):
    """
    Perform Procrustes' analysis subject to constraints. Finds the
    transformation T mapping a to b which minimizes the square sum
    distances between Ta and b, also called the cost.

    Parameters
    ----------
    a : (n,3) float
      List of points in space
    b : (n,3) float
      List of points in space
    reflection : bool
      If the transformation is allowed reflections
    translation : bool
      If the transformation is allowed translations
    scale : bool
      If the transformation is allowed scaling
    return_cost : bool
      Whether to return the cost and transformed a as well

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
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
    a : (n,3) float
      List of points in space.
    b : (m,3) float or Trimesh
      List of points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    kwargs : dict
      Args to pass to procrustes

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
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

    # transform a under initial_transformation
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

        # align a with closest points
        matrix, transformed, cost = procrustes(a=a,
                                               b=closest,
                                               **kwargs)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost
