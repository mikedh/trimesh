"""
comparison.py
----------------

Provide methods for quickly hashing and comparing meshes.
"""

import numpy as np

from . import util

from .constants import tol

# how many signifigant figures to use for each field of the identifier
id_sigfig = np.array([5,  # area
                      10,  # euler number
                      5,  # area/volume ratio
                      2,  # convex/mesh area ratio
                      2,  # convex area/volume ratio
                      3])  # max radius squared / area


def identifier_simple(mesh):
    """
    Return a basic identifier for a mesh, consisting of properties
    that are somewhat robust to transformation and noise.

    Parameters
    ----------
    mesh: Trimesh object

    Returns
    ----------
    identifier: (6,) float, identifying values of the mesh
    """

    # pre-allocate identifier so indexes of values can't move around
    # like they might if we used hstack or something else
    identifier = np.zeros(6, dtype=np.float64)

    # start with properties that are valid regardless of watertightness
    # note that we're going to try to make all parameters relative
    # to area so other values don't get blown up at weird scales
    identifier[0] = mesh.area
    # topological constant and the only thing we can really
    # trust in this fallen world
    identifier[1] = mesh.euler_number

    # if we have a watertight mesh include volume and inertia
    if mesh.is_volume:
        # side length of a cube ratio
        # 1.0 for cubes, different values for other things
        identifier[2] = (((mesh.area / 6.0) ** (1.0 / 2.0)) /
                         (mesh.volume ** (1.0 / 3.0)))
        vertices = mesh.vertices - mesh.center_mass

        # we are going to special case radially symmetric meshes
        # to replace their surface area with ratio of their
        # surface area to a primitive sphere or cylinder surface area
        # this is because tesselated curved surfaces are really rough
        # to reliably hash as they are very sensitive to floating point
        # and tesselation error. By making area proportionate to a fit
        # primitive area we are able to reliably hash at more sigfigs
        if mesh.symmetry == 'radial':
            # cylinder height
            h = np.dot(vertices, mesh.symmetry_axis).ptp()
            # section radius
            R2 = (np.dot(vertices, mesh.symmetry_section.T)**2).sum(axis=1).max()
            # area of a cylinder primitive
            area = (2 * np.pi * (R2**.5) * h) + (2 * np.pi * R2)
            # replace area in this case with area ratio
            identifier[0] = mesh.area / area
        elif mesh.symmetry == 'spherical':
            # handle a spherically symmetric mesh
            R2 = (vertices ** 2).sum(axis=1).max()
            area = 4 * np.pi * R2
            identifier[0] = mesh.area / area
    else:
        # if we don't have a watertight mesh add information about the
        # convex hull, which is slow to compute and unreliable
        # just what we're looking for in a hash but hey
        identifier[3] = mesh.area / mesh.convex_hull.area
        # cube side length ratio for the hull
        identifier[4] = (((mesh.convex_hull.area / 6.0) ** (1.0 / 2.0)) /
                         (mesh.convex_hull.volume ** (1.0 / 3.0)))
        vertices = mesh.vertices - mesh.centroid

    # add in max radius^2 to area ratio
    R2 = (vertices ** 2).sum(axis=1).max()
    identifier[5] = R2 / mesh.area

    return identifier


def identifier_hash(identifier, sigfig=None):
    """
    Hash an identifier array to a specified number of signifigant figures.

    Parameters
    ----------
    identifier: (n,) float
    sigfig:     (n,) int

    Returns
    ----------
    md5: str, MD5 hash of identifier
    """
    if sigfig is None:
        sigfig = id_sigfig

    # convert identifer to integers and order of magnitude
    as_int, multiplier = util.sigfig_int(identifier, sigfig)
    # make all scales positive
    if (multiplier < 0).any():
        multiplier += np.abs(multiplier.min())
    hashable = (as_int * (10 ** multiplier)).astype(np.int64)
    md5 = util.md5_object(hashable)
    return md5
