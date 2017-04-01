import numpy as np


def contains_points(mesh, points):
    '''
    Check if a mesh contains a set of points, using ray tests.

    If the point is on the surface of the mesh, behavior is undefined.

    Parameters
    ---------
    mesh: Trimesh object
    points: (n,3) points in space

    Returns
    ---------
    contains: (n) boolean array, whether point is inside mesh or not
    '''
    ray_origins = np.asanyarray(points)
    # rays are all going in arbitrary direction
    ray_directions = np.tile([0, 0, 1.0], (len(points), 1))
    locations, index_ray = mesh.ray.intersects_location(
        ray_origins, ray_directions)

    if len(locations) == 0:
        return np.zeros(len(points), dtype=np.bool)

    hit_count = np.bincount(index_ray, minlength=len(points))
    contains = np.mod(hit_count, 2) == 1

    return contains
