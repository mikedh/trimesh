"""
raytrace.py
----------------

A very simple example of using scene cameras to generate
rays for image reasons.

Install `pyembree` for a speedup (600k+ rays per second)
"""
import PIL

import trimesh
import numpy as np


def camera_to_rays(camera):
    """
    Convert a trimesh.scene.Camera object to ray origins
    and direction vectors.

    Parameters
    --------------
    camera : trimesh.scene.Camera
      Camera with transform defined

    Returns
    --------------
    origins : (n, 3) float
      Ray origins in space
    vectors : (n, 3) float
      Ray direction unit vectors
    angles : (n, 2) float
      Ray spherical coordinate angles in radians
    """
    # radians of half the field of view
    half = np.radians(camera.fov / 2.0)
    # scale it down by two pixels to keep image under resolution
    half *= (camera.resolution - 2) / camera.resolution
    # get FOV angular bounds in radians

    # create an evenly spaced list of angles
    angles = trimesh.util.grid_linspace(bounds=[-half, half],
                                        count=camera.resolution)

    # turn the angles into unit vectors
    vectors = trimesh.unitize(np.column_stack((
        np.sin(angles),
        np.ones(len(angles)))))

    # flip the camera transform to change sign of Z
    transform = np.dot(
        camera.transform,
        trimesh.geometry.align_vectors([1, 0, 0], [-1, 0, 0]))

    # apply the rotation to the direction vectors
    vectors = trimesh.transform_points(vectors,
                                       transform,
                                       translate=False)

    # camera origin is single point, extract from transform
    origin = trimesh.transformations.translation_from_matrix(transform)
    # tile it into corresponding list of ray vectorsy
    origins = np.ones_like(vectors) * origin

    return origins, vectors, angles


if __name__ == '__main__':

    # test on a simple mesh
    mesh = trimesh.load('../models/featuretype.STL')

    # scene will have automatically generated camera and lights
    scene = mesh.scene()

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [640, 480]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = 60 * (scene.camera.resolution /
                             scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origins, vectors, angles = camera_to_rays(scene.camera)

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False)

    # for each hit, find the distance along its vector
    # you could also do this against the single camera Z vector
    depth = trimesh.util.diagonal_dot(points - origins[0],
                                      vectors[index_ray])

    # find the angular resolution, in pixels per radian
    ppr = scene.camera.resolution / np.radians(scene.camera.fov)
    # convert rays to pixel locations
    pixel = (angles * ppr).round().astype(np.int64)
    # make sure we are in the first quadrant
    pixel -= pixel.min(axis=0)
    # find pixel locations of actual hits
    pixel_ray = pixel[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = ((depth - depth.min()) / depth.ptp())

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

    # create a PIL image from the depth queries
    img = PIL.Image.fromarray(a)
