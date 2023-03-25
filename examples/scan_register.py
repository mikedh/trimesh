"""
scan_register.py
-------------

Create some simulated 3D scan data and register
it to a "truth" mesh.
"""

import trimesh
import numpy as np


def simulated_brick(face_count, extents, noise, max_iter=10):
    """
    Produce a mesh that is a rectangular solid with noise
    with a random transform.

    Parameters
    -------------
    face_count : int
      Approximate number of faces desired
    extents : (n,3) float
      Dimensions of brick
    noise : float
      Magnitude of vertex noise to apply
    """

    # create the mesh as a simple box
    mesh = trimesh.creation.box(extents=extents)

    # add some systematic error pre- tessellation
    mesh.vertices[0] += mesh.vertex_normals[0] + (noise * 2)

    # subdivide until we have more faces than we want
    for _i in range(max_iter):
        if len(mesh.vertices) > face_count:
            break
        mesh = mesh.subdivide()

    # apply tessellation and random noise
    mesh = mesh.permutate.noise(noise)

    # randomly rotation with translation
    transform = trimesh.transformations.random_rotation_matrix()
    transform[:3, 3] = (np.random.random(3) - .5) * 1000

    mesh.apply_transform(transform)

    return mesh


if __name__ == '__main__':
    # print log messages to terminal
    trimesh.util.attach_to_log()
    log = trimesh.util.log

    # the size of our boxes
    extents = [6, 12, 2]

    # create a simulated brick with noise and random transform
    scan = simulated_brick(face_count=5000,
                           extents=extents,
                           noise=.05)

    # create a "true" mesh
    truth = trimesh.creation.box(extents=extents)

    # (4, 4) float homogeneous transform from truth to scan
    # this will do an ICP refinement with initial transforms
    # seeded by the principal components of inertia
    truth_to_scan, cost = truth.register(scan)

    log.debug("centroid distance pre-registration:",
              np.linalg.norm(truth.centroid - scan.centroid))

    # apply the registration transform
    truth.apply_transform(truth_to_scan)

    log.debug("centroid distance post-registration:",
              np.linalg.norm(truth.centroid - scan.centroid))

    # find the distance from the truth mesh to each scan vertex
    distance = truth.nearest.on_surface(scan.vertices)[1]

    # color the mesh by distance from truth
    # linear interpolation between two colors
    # scaled between distance.min() and distance.max()
    scan.visual.vertex_colors = trimesh.visual.interpolate(distance)

    # print some quick statistics about the mesh
    log.debug('distance max:', distance.max())
    log.debug('distance mean:', distance.mean())
    log.debug('distance STD:', distance.std())

    # export result with vertex colors for meshlab
    scan.export('scan_new.ply')

    # show in a pyglet window
    scan.show()
