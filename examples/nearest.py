import trimesh

import numpy as np

import sys

if __name__ == '__main__':
    # print logging messages to current terminal
    trimesh.util.attach_to_log()

    # load a large- ish PLY model with colors    
    mesh = trimesh.load('../models/cycloidal.ply')

    # create a set of 3D points randomly distributed inside the
    # bounding box of the mesh
    points  = np.random.random((20,3)) * mesh.extents
    points += mesh.bounds[0]

    # find the closest point on the mesh to each random point
    (closest_points,
     closest_distances,
     closest_triangle_id) = mesh.nearest.on_surface(points)

    # create a PointCloud object out of each (n,3) list of points
    cloud_original = trimesh.points.PointCloud(points)
    cloud_close    = trimesh.points.PointCloud(closest_points)

    # create a unique color for each point
    colors = np.array([trimesh.visual.random_color() for i in points])

    # set the colors on the random point and its nearest point to be the same
    cloud_original.vertices_color = colors
    cloud_close.vertices_color    = colors

    # create a scene containing the mesh and two sets of points
    scene = trimesh.Scene([mesh,
                           cloud_original,
                           cloud_close])

    # if we are ran without -nw open a pyglet window showing the scene
    if not '-nw' in sys.argv:
        scene.show()
