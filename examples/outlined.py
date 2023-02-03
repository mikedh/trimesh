"""
outlined.py
--------------

Show a mesh with edges highlighted using GL_LINES
"""

import trimesh
import numpy as np

if __name__ == '__main__':
    mesh = trimesh.load('../models/featuretype.STL')

    # get edges we want to highlight by finding edges
    # that have sharp angles between adjacent faces
    edges = mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.radians(
        30)]
    # get a Path3D object for the edges we want to highlight
    path = trimesh.path.Path3D(**trimesh.path.exchange.misc.edges_to_path(
        edges, mesh.vertices.copy()))

    # set the mesh face colors to white
    mesh.visual.face_colors = [255, 255, 255, 255]
    # create a scene with both the mesh and the outline edges
    scene = trimesh.Scene([mesh, path])

    # set the camera resolution
    scene.camera.resolution = (4000, 2000)
    # set the camera transform to look at the mesh
    scene.camera_transform = scene.camera.look_at(
        points=mesh.vertices,
        rotation=trimesh.transformations.euler_matrix(np.pi / 3, 0, np.pi / 5))

    # write a PNG of the render
    with open('outlined.PNG', 'wb') as f:
        f.write(scene.save_image())
