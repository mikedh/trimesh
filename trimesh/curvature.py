"""
curvature.py
---------------

Query mesh curvature.
"""

import numpy as np
from . import util

def vertex_defects(mesh):
    """
    Return the vertex defects

    Returns
    --------
    vertex_defect: (n,) float vertex defect at the given vertex.
                   Each value corresponds with self.vertices
    """
    defects = np.empty(len(mesh.vertices))
    for i in range(len(mesh.vertices)):
        faces, v_ix = np.where(mesh.faces == i) # faces incident at i
        v1 = mesh.triangles[faces, (v_ix+1) % 3] - mesh.vertices[i]
        v2 = mesh.triangles[faces, (v_ix-1) % 3] - mesh.vertices[i]
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
        defects[i] = 2*np.pi - np.arccos(np.clip(np.einsum('ij, ij->i', v1, v2), -1, 1)).sum()
    return defects

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
    candidates = [list(mesh.face_adjacency_tree.intersection(b)) for b in bounds]
    
    mean_curv = np.empty(len(points))
    for i, (x, x_candidates) in enumerate(zip(points, candidates)):
        endpoints = mesh.vertices[mesh.face_adjacency_edges[x_candidates]]
        lengths = line_ball_intersection(endpoints[:, 0], endpoints[:, 1], x, radius)
        angles = mesh.face_adjacency_angles[x_candidates]
        signs = np.where(mesh.face_adjacency_convex[x_candidates], 1, -1)
        mean_curv[i] = (lengths*angles*signs).sum()/2
    
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
    # x = o + dl. This yields
    # d = (-l.(o-c) +- sqrt[ l.(o-c)**2 - l.l((o-c).(o-c) - r^**2) ]) / l.l
    l = end_points - start_points
    oc = start_points - center # o-c
    r = radius
    ldotl = np.einsum('ij, ij->i', l, l) #l.l
    ldotoc = np.einsum('ij, ij->i', l, oc) #l.(o-c)
    ocdotoc = np.einsum('ij, ij->i', oc, oc) #(o-c).(o-c)
    discrims = ldotoc**2 - ldotl*(ocdotoc - r**2)
    
    # If discriminant is non-positive, then we have zero length
    lengths = np.zeros(len(start_points))
    # Otherwise we solve for the solns with d2 > d1.
    m = discrims > 0 # mask
    d1 = (-ldotoc[m] - np.sqrt(discrims[m])) / ldotl[m]
    d2 = (-ldotoc[m] + np.sqrt(discrims[m])) / ldotl[m]
    
    # Line segment means we have 0 <= d <= 1
    d1 = np.clip(d1, 0, 1)
    d2 = np.clip(d2, 0, 1)
    
    # Length is |o + d2 l - o + d1 l|  = (d2 - d1) |l|
    lengths[m] = (d2-d1) * np.sqrt(ldotl[m])
    
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
    x = (2*R**2 - r**2)/(2*R) # x coord of plane
    if x >= -R:
        return 2*np.pi*R*(R-x)
    if x < -R:
        return 4*np.pi*R**2

