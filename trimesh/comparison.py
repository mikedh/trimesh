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
                      2])  # convex area/volume ratio


def identifier_simple(mesh):
    """
    Return a basic identifier for a mesh, consisting of properties
    that are somewhat robust to transformation and noise.

    Parameters
    ----------
    mesh: Trimesh object

    Returns
    ----------
    identifier: (5,) float, identifying values of the mesh
    """
    # pre-allocate identifier so indexes of values can't move around
    # like they might if we used hstack or something else
    identifier = np.zeros(5, dtype=np.float64)

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
    else:
        # if we don't have a watertight mesh add information about the
        # convex hull, which is slow to compute and unreliable
        # just what we're looking for in a hash but hey
        identifier[3] = mesh.area / mesh.convex_hull.area
        # cube side length ratio for the hull
        identifier[4] = (((mesh.convex_hull.area /6.0) ** (1.0 / 2.0)) /
                         (mesh.convex_hull.volume ** (1.0 / 3.0)))

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
