'''
Narrow phase ray- triangle intersection
'''
import numpy as np
import time

from ..constants import log, log_time, TOL_ZERO

@log_time
def rays_triangles_id(triangles,
                      rays, 
                      ray_candidates = None,
                      return_any     = False):
    '''
    Intersect a set of rays and triangles. 

    Arguments
    ---------
    triangles:      (n, 3, 3) float array of triangle vertices
    rays:           (m, 2, 3) float array of ray start, ray directions
    ray_candidates: (m, *) int array of which triangles are candidates
                    for the ray. 
    return_any:     bool flag for output format

    Returns
    ---------
    intersections, return_any:      (m) bool of whether the ray hit any triangles
    intersections, not return_any:  (m) sequence of triangle indexes hit by rays
    '''

    if return_any: hits = np.zeros(len(rays), dtype=np.bool)
    else:          hits = [None] * len(rays)
  
    # default set of candidate triangles to be queried 
    # is every triangle. this is very slow
    candidates = np.ones(len(triangles), dtype=np.bool)

    for ray_index, ray in enumerate(rays):
        if ray_candidates is not None:
            candidates = ray_candidates[ray_index]
        # query the triangle candidates
        hit = ray_triangles(triangles[candidates], *ray)
        if return_any: 
            hits[ray_index] = hit.any()
        else:
            hits[ray_index] = np.array(candidates)[hit]
    return np.array(hits)

def ray_triangles(triangles, 
                  ray_origin, 
                  ray_direction):
    '''
    Intersection of multiple triangles and a single ray.

    Uses Moller-Trumbore intersection algorithm
    '''
    candidates = np.ones(len(triangles), dtype=np.bool)

    vert0 = triangles[:,0,:]
    vert1 = triangles[:,1,:]
    vert2 = triangles[:,2,:]

    edge0 = vert1 - vert0
    edge1 = vert2 - vert0

    #P is a vector perpendicular to the ray direction and one
    # triangle edge. 
    P   = np.cross(ray_direction, edge1)

    #if determinant is near zero, ray lies in plane of triangle
    det = _diag_dot(edge0, P)
    
    candidates[np.abs(det) < TOL_ZERO] = False

    if not candidates.any(): return candidates

    inv_det = 1.0 / det[candidates]
    T = ray_origin - vert0[candidates]
    u = _diag_dot(T, P[candidates]) * inv_det

    new_candidates         = np.logical_not(np.logical_or(u < TOL_ZERO,
                                                          u > (1-TOL_ZERO)))
    candidates[candidates] = new_candidates
    if not candidates.any(): return candidates    
    inv_det = inv_det[new_candidates]
    T       = T[new_candidates]
    u       = u[new_candidates]

    Q = np.cross(T, edge0[candidates])
    v = np.dot(ray_direction, Q.T) * inv_det

    new_candidates = np.logical_not(np.logical_or((v     < TOL_ZERO),
                                                  (u + v > (1-TOL_ZERO))))
    candidates[candidates] = new_candidates
    if not candidates.any(): return candidates
    Q       = Q[new_candidates]
    inv_det = inv_det[new_candidates]
    
    t = _diag_dot(edge1[candidates], Q) * inv_det
    candidates[candidates] = t > TOL_ZERO

    return candidates

def _diag_dot(a, b):
    '''
    Dot product by row of a and b.

    Same as np.diag(np.dot(a, b.T)) but without the monstrous 
    intermediate matrix (and is much faster). 
    '''
    result = np.array([np.dot(i,j) for i,j in zip(a,b)])
    return result
