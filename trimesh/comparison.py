import numpy as np

from . import util

from .constants import tol


identifier_sigfig = (4, 2, 1, 5, 2)


def identifier_simple(mesh):
    '''
    Return a basic identifier for a mesh, consisting of properties
    that are somewhat robust to transformation and noise. 

    These include:
    -volume
    -surface area
    -convex hull surface area
    -euler number
    -average radius

    Arguments
    ----------
    mesh: Trimesh object

    Returns
    ----------
    identifier: (5,) float, properties of mesh
    '''
    identifier = np.array([mesh.volume,
                           mesh.area,
                           mesh.area / mesh.convex_hull_raw.area,
                           mesh.euler_number,
                           0.0],
                          dtype=np.float64)

    if mesh.is_watertight:
        origin = mesh.center_mass
    else:
        origin = mesh.centroid
    vertex_radii = ((mesh.vertices - origin) ** 2).sum(axis=1)
    identifier[-1] = np.percentile(vertex_radii, 99.99)

    return identifier


def identifier_hash(identifier, sigfig=None):
    '''
    Hash an identifier array to a specified number of signifigant figures.

    Arguments
    ----------
    identifier: (n,) float
    sigfig:     (n,) int

    Returns
    ----------
    md5: str, MD5 hash of identifier
    '''
    if sigfig is None:
        sigfig = identifier_sigfig
    as_int, multiplier = util.sigfig_int(identifier, sigfig)
    hashable = (as_int * (10 ** multiplier)).astype(np.int32)
    md5 = util.md5_object(hashable)
    return md5
