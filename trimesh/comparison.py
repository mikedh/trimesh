import numpy as np

from . import util

from .constants import tol


# how many signifigant figures to use for each field of the identifier
identifier_sigfig = (5,  # mesh volume, pretty stable
                     3,  # mesh area
                     1,  # ratio of original mesh volume to convex hull volume
                     6,  # euler number of mesh- topological integer
                     2)  # 99.99 percentile vertex radius


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

    Parameters
    ----------
    mesh: Trimesh object

    Returns
    ----------
    identifier: (5,) float, properties of mesh
    '''
    identifier = np.array([mesh.volume,
                           mesh.area,
                           mesh.volume / mesh.convex_hull.volume,
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

    Parameters
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
    if (multiplier < 0).any():
        multiplier += np.abs(multiplier.min())
    hashable = (as_int * (10 ** multiplier)).astype(np.int64)
    md5 = util.md5_object(hashable)
    return md5
