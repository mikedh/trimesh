import numpy as np

#from .ray_pyembree import RayMeshIntersector
    
from .ray_triangle import RayMeshIntersector    

def contains_points(mesh, points):
    '''
    Check if a mesh contains a set of points, using ray tests.

    If the point is on the surface of the mesh, behavior is undefined.

    Arguments
    ---------
    mesh: Trimesh object
    points: (n,3) points in space

    Returns
    ---------
    contains: (n) boolean array, whether point is inside mesh or not
    '''
    ray_origins = np.asanyarray(points)
    # rays are all going in arbitrary direction
    ray_directions = np.tile([0,0,1.0], (len(points), 1))
    hits = mesh.ray.intersects_location(ray_origins, ray_directions)
    hits_count = np.array([len(i) for i in hits])
    contains = np.mod(hits_count, 2) == 1

    return contains
