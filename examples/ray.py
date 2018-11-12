"""
ray.py
----------------

Do simple mesh- ray queries. Base functionality only
requires numpy, but if you install `pyembree` you get the
same API with a roughly 50x speedup.
"""

import trimesh
import numpy as np

if __name__ == '__main__':

    # test on a sphere mesh
    mesh = trimesh.primitives.Sphere()

    # create some rays
    ray_origins = np.array([[0, 0, -5],
                            [2, 2, -10]])
    ray_directions = np.array([[0, 0, 1],
                               [0, 0, 1]])

    """
    Signature: mesh.ray.intersects_location(ray_origins,
                                            ray_directions,
                                            multiple_hits=True)
    Docstring:

    Return the location of where a ray hits a surface.

    Parameters
    ----------
    ray_origins:    (n,3) float, origins of rays
    ray_directions: (n,3) float, direction (vector) of rays


    Returns
    ---------
    locations: (n) sequence of (m,3) intersection points
    index_ray: (n,) int, list of ray index
    index_tri: (n,) int, list of triangle (face) indexes
    """

    # run the mesh- ray test
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)

    # stack rays into line segments for visualization as Path3D
    ray_visualize = trimesh.load_path(np.hstack((
        ray_origins,
        ray_origins + ray_directions)).reshape(-1, 2, 3))

    # make mesh transparent- ish
    mesh.visual.face_colors = [100, 100, 100, 100]

    # create a visualization scene with rays, hits, and mesh
    scene = trimesh.Scene([
        mesh,
        ray_visualize,
        trimesh.points.PointCloud(locations)])

    # display the scene
    scene.show()
