"""
curvature.py
---------------

Query mesh curvature.
"""
import numpy as np

from . import util

try:
    from scipy.sparse.coo import coo_matrix
except ImportError:
    pass


def face_angles(mesh):
    """
    Returns the angle at each vertex of a face.

    Returns
    --------
    angles: (n, 3) float, angle at each vertex of a face.
    """

    u = mesh.triangles[:, 1] - mesh.triangles[:, 0]
    v = mesh.triangles[:, 2] - mesh.triangles[:, 0]
    w = mesh.triangles[:, 2] - mesh.triangles[:, 1]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    a = np.arccos(np.clip(np.einsum('ij, ij->i', u, v), -1, 1))
    b = np.arccos(np.clip(np.einsum('ij, ij->i', -u, w), -1, 1))
    c = np.pi - a - b

    return np.vstack([a, b, c]).T


def face_angles_sparse(mesh):
    """
    A sparse matrix representation of the face angles.

    Returns
    ----------
    sparse: scipy.sparse.coo_matrix with:
            dtype: float
            shape: (len(mesh.vertices), len(mesh.faces))
    """
    matrix = coo_matrix((mesh.face_angles.flatten(),
                         (mesh.faces_sparse.row, mesh.faces_sparse.col)),
                        mesh.faces_sparse.shape)
    return matrix


def vertex_defects(mesh):
    """
    Return the vertex defects, or (2*pi) minus the sum of the angles
    of every face that includes that vertex.

    If a vertex is only included by coplanar triangles, this
    will be zero. For convex regions this is positive, and
    concave negative.

    Returns
    --------
    vertex_defect : (len(self.vertices), ) float
                     Vertex defect at the every vertex
    """
    angle_sum = np.asarray(mesh.face_angles_sparse.sum(axis=1)).flatten()
    defect = (2 * np.pi) - angle_sum
    return defect


def discrete_gaussian_curvature_measure(mesh, points, radius):
    """
    Return the discrete gaussian curvature measure of a sphere centered
    at a point as detailed in 'Restricted Delaunay triangulations and normal
    cycle', Cohen-Steiner and Morvan.

    Parameters
    ----------
    points : (n,3) float, list of points in space
    radius : float, the sphere radius

    Returns
    --------
    gaussian_curvature: (n,) float, discrete gaussian curvature measure.
    """

    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    nearest = mesh.kdtree.query_ball_point(points, radius)
    gauss_curv = [mesh.vertex_defects[vertices].sum() for vertices in nearest]

    return np.asarray(gauss_curv)


def discrete_mean_curvature_measure(mesh, points, radius):
    """
    Return the discrete mean curvature measure of a sphere centered
    at a point as detailed in 'Restricted Delaunay triangulations and normal
    cycle', Cohen-Steiner and Morvan.

    Parameters
    ----------
    points : (n,3) float, list of points in space
    radius : float, the sphere radius

    Returns
    --------
    mean_curvature: (n,) float, discrete mean curvature measure.
    """

    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    # axis aligned bounds
    bounds = np.column_stack((points - radius,
                              points + radius))

    # line segments that intersect axis aligned bounding box
    candidates = [list(mesh.face_adjacency_tree.intersection(b))
                  for b in bounds]

    mean_curv = np.empty(len(points))
    for i, (x, x_candidates) in enumerate(zip(points, candidates)):
        endpoints = mesh.vertices[mesh.face_adjacency_edges[x_candidates]]
        lengths = line_ball_intersection(
            endpoints[:, 0],
            endpoints[:, 1],
            center=x,
            radius=radius)
        angles = mesh.face_adjacency_angles[x_candidates]
        signs = np.where(mesh.face_adjacency_convex[x_candidates], 1, -1)
        mean_curv[i] = (lengths * angles * signs).sum() / 2

    return mean_curv


def line_ball_intersection(start_points, end_points, center, radius):
    """
    Compute the length of the intersection of a line segment with a ball.

    Parameters
    ----------
    start_points : (n,3) float, list of points in space
    end_points   : (n,3) float, list of points in space
    center       : (3,) float, the sphere center
    radius       : float, the sphere radius

    Returns
    --------
    lengths: (n,) float, the lengths.

    """

    # We solve for the intersection of |x-c|**2 = r**2 and
    # x = o + dL. This yields
    # d = (-l.(o-c) +- sqrt[ l.(o-c)**2 - l.l((o-c).(o-c) - r^**2) ]) / l.l
    L = end_points - start_points
    oc = start_points - center  # o-c
    r = radius
    ldotl = np.einsum('ij, ij->i', L, L)  # l.l
    ldotoc = np.einsum('ij, ij->i', L, oc)  # l.(o-c)
    ocdotoc = np.einsum('ij, ij->i', oc, oc)  # (o-c).(o-c)
    discrims = ldotoc**2 - ldotl * (ocdotoc - r**2)

    # If discriminant is non-positive, then we have zero length
    lengths = np.zeros(len(start_points))
    # Otherwise we solve for the solns with d2 > d1.
    m = discrims > 0  # mask
    d1 = (-ldotoc[m] - np.sqrt(discrims[m])) / ldotl[m]
    d2 = (-ldotoc[m] + np.sqrt(discrims[m])) / ldotl[m]

    # Line segment means we have 0 <= d <= 1
    d1 = np.clip(d1, 0, 1)
    d2 = np.clip(d2, 0, 1)

    # Length is |o + d2 l - o + d1 l|  = (d2 - d1) |l|
    lengths[m] = (d2 - d1) * np.sqrt(ldotl[m])

    return lengths


def sphere_ball_intersection(R, r):
    """
    Compute the surface area of the intersection of sphere of radius R centered
    at (0, 0, 0) with a ball of radius r centered at (R, 0, 0).

    Parameters
    ----------
    R : float, sphere radius
    r : float, ball radius

    Returns
    --------
    area: float, the surface are.
    """
    x = (2 * R**2 - r**2) / (2 * R)  # x coord of plane
    if x >= -R:
        return 2 * np.pi * R * (R - x)
    if x < -R:
        return 4 * np.pi * R**2
