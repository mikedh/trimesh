import numpy as np

from . import util

from .triangles import closest_point

def closest_point_naive(triangles, points):
    '''
    Given a NON- CORRESPONDING list of triangles and points, for each point
    find the closest point on any triangle.
    
    Does this by constructing a very large intermediate array and 
    comparing every point to every triangle.

    Arguments
    ----------
    triangles: (n,3,3) float, triangles in space
    points:    (m,3)   float, points in space

    Returns
    ----------
    closest:     (m,3) float, closest point on triangles for each point
    distance:    (m,)  float, distance
    triangle_id: (m,)  int, index of closest triangle for each point
    '''

    # establish that input triangles and points are sane
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points    = np.asanyarray(points,    dtype=np.float64)
    if not util.is_shape(triangles, (-1,3,3)):
        raise ValueError('triangles shape incorrect')
    if not util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)')

    # create a giant tiled array of each point tiled len(triangles) times
    points_tiled = np.tile(points, (1, len(triangles)))
    on_triangle = np.array([closest_point(triangles, 
                                          i.reshape((-1,3))) for i in points_tiled])

    # distance squared
    distance_2 = [((i-q)**2).sum(axis=1) for i,q in zip(on_triangle, points)]
    triangle_id = np.array([i.argmin() for i in distance_2])
    
    # closest cartesian point
    closest  = np.array([g[i] for i,g in zip(triangle_id, on_triangle)])
    distance = np.array([g[i] for i,g in zip(triangle_id, distance_2)]) ** .5

    return closest, distance, triangle_id
