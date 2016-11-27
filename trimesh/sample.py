import numpy as np

from . import util


def sample_surface(mesh, count):
    '''
    Sample the surface of a mesh, returning the specified number of points

    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html

    Arguments
    ---------
    mesh: Trimesh object
    count: number of points to return

    Returns
    ---------
    samples: (count,3) points in space on the surface of mesh

    '''

    # len(mesh.faces) float array of the areas of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples


def sample_volume(mesh, count):
    '''
    Use rejection sampling to produce points randomly distributed in the volume of a mesh.

    Arguments
    ----------
    mesh: Trimesh object
    count: int, number of samples desired

    Returns
    ----------
    samples: (n,3) float, points in the volume of the mesh. 
             (n <= count) as samples are produced by rejecting points 
    '''
    points = (np.random.random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = mesh.contains(points)
    samples = points[contained][:count]
    return samples


def sample_surface_even(mesh, count):
    '''
    Sample the surface of a mesh, returning samples which are
    approximately evenly spaced.
    '''
    from .points import remove_close

    radius = np.sqrt(mesh.area / (2 * count))
    samples = sample_surface(mesh, count * 5)
    result = remove_close(samples, radius)
    return result


def sample_surface_sphere(count):
    '''
    Correctly pick random points on the surface of a unit sphere

    Uses this method:
    http://mathworld.wolfram.com/SpherePointPicking.html

    Arguments
    ----------
    count: int, number of points to return

    Returns
    ----------
    points: (count,3) float, list of random points on a unit sphere
    '''

    u, v = np.random.random((2, count))

    theta = np.pi * 2 * u
    phi = np.arccos((2 * v) - 1)

    points = util.spherical_to_vector(np.column_stack((theta, phi)))
    return points
