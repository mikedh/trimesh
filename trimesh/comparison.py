import numpy as np

from . import util

from .constants import tol

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
                           mesh.convex_hull_raw.area,
                           mesh.euler_number,
                           0.0],
                          dtype = np.float64)
    
    if mesh.is_watertight:
        origin = mesh.center_mass
    else:
        origin = mesh.centroid
    
    vertex_radii = ((mesh.vertices - origin) ** 2).sum(axis=1)
    center_radii = ((mesh.triangles_center - origin) ** 2).sum(axis=1)
    radii = np.column_stack((vertex_radii[mesh.faces],
                             center_radii)).reshape(-1)
    weights = np.tile((mesh.area_faces.reshape((-1, 1)) * (1.0 / 4.0)),
                      (1, 4)).reshape(-1)
    identifier[-1] = np.average(radii, weights=weights)

    return identifier

def identifier_hash(identifier, sigfig = [4,2,2,5,2]):
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
    sigfig = np.asanyarray(sigfig, dtype=np.int).reshape(-1)

    if sigfig.shape != identifier.shape:
        raise ValueError('sigfig must match identifier')

    exponent = np.zeros(len(identifier))
    nonzero = np.abs(identifier) > tol.zero
    exponent[nonzero] = np.floor(np.log10(np.abs(identifier[nonzero])))    

    multiplier = exponent.copy()
    multiplier -= sigfig - 1

    hashable = np.append(np.round(identifier / (10**multiplier)),
                         exponent).astype(np.int32)
    md5 = util.md5_object(hashable)
    return md5
