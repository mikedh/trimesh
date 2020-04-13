"""
comparison.py
----------------

Provide methods for quickly hashing and comparing meshes.
"""

import numpy as np

from . import util

from .constants import tol

# how many significant figures to use for each field of the identifier
id_sigfig = np.array([5,  # area
                      10,  # euler number
                      5,  # area/volume ratio
                      2,  # convex/mesh area ratio
                      2,  # convex area/volume ratio
                      3])  # max radius squared / area


def identifier_simple(mesh):
    """
    Return a basic identifier for a mesh, consisting of properties
    that have been hand tuned to be somewhat robust to rigid
    transformations and different tesselations.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry

    Returns
    ----------
    identifier : (6,) float
      Identifying values of the mesh
    """
    # verify the cache once
    mesh._cache.verify()

    # don't check hashes during identifier as we aren't
    # changing any data values of the mesh inside block
    # if we did change values in cache block things would break
    with mesh._cache:
        # pre-allocate identifier so indexes of values can't move around
        # like they might if we used hstack or something else
        identifier = np.zeros(6, dtype=np.float64)
        # avoid thrashing the cache unnecessarily
        mesh_area = mesh.area
        # start with properties that are valid regardless of watertightness
        # note that we're going to try to make all parameters relative
        # to area so other values don't get blown up at weird scales
        identifier[0] = mesh_area
        # avoid divide-by-zero later
        if mesh_area < tol.merge:
            mesh_area = 1.0
        # topological constant and the only thing we can really
        # trust in this fallen world
        identifier[1] = mesh.euler_number

        # if we have a watertight mesh include volume and inertia
        if mesh.is_volume:
            # side length of a cube ratio
            # 1.0 for cubes, different values for other things
            identifier[2] = (((mesh_area / 6.0) ** (1.0 / 2.0)) /
                             (mesh.volume ** (1.0 / 3.0)))
            # save vertices for radius calculation
            vertices = mesh.vertices - mesh.center_mass
            # we are going to special case radially symmetric meshes
            # to replace their surface area with ratio of their
            # surface area to a primitive sphere or cylinder surface area
            # this is because tessellated curved surfaces are really rough
            # to reliably hash as they are very sensitive to floating point
            # and tessellation error. By making area proportionate to a fit
            # primitive area we are able to reliably hash at more sigfigs
            if mesh.symmetry == 'radial':
                # cylinder height
                h = np.dot(vertices, mesh.symmetry_axis).ptp()
                # section radius summed per row then overall max
                R2 = np.dot((np.dot(vertices, mesh.symmetry_section.T)
                             ** 2), [1, 1]).max()
                # area of a cylinder primitive
                area = (2 * np.pi * (R2**.5) * h) + (2 * np.pi * R2)
                # replace area in this case with area ratio
                identifier[0] = mesh_area / area
            elif mesh.symmetry == 'spherical':
                # handle a spherically symmetric mesh
                R2 = np.dot((vertices ** 2), [1, 1, 1]).max()
                area = 4 * np.pi * R2
                identifier[0] = mesh_area / area
        else:
            # if we don't have a watertight mesh add information about the
            # convex hull which is slow to compute and unreliable
            try:
                # get the hull area and volume
                hull = mesh.convex_hull
                hull_area = hull.area
                hull_volume = hull.volume
            except BaseException:
                # in-plane or single point geometry has no hull
                hull_area = 6.0
                hull_volume = 1.0
            # just what we're looking for in a hash but hey
            identifier[3] = mesh_area / hull_area
            # cube side length ratio for the hull
            identifier[4] = (((hull_area / 6.0) ** (1.0 / 2.0)) /
                             (hull_volume ** (1.0 / 3.0)))
            # calculate maximum mesh radius
            vertices = mesh.vertices - mesh.centroid
            # add in max radius^2 to area ratio
            R2 = np.dot((vertices ** 2), [1, 1, 1]).max()
            identifier[5] = R2 / mesh_area

    return identifier


def identifier_hash(identifier, sigfig=None):
    """
    Hash an identifier array to a specified number of
    significant figures.

    Parameters
    ------------
    identifier : (n,) float
      Vector of properties
    sigfig : (n,) int
      Number of sigfigs per property

    Returns
    ----------
    md5 : str
      MD5 hash of identifier
    """
    if sigfig is None:
        sigfig = id_sigfig

    # convert identifier to integers and order of magnitude
    as_int, multiplier = util.sigfig_int(identifier, sigfig)
    # make all scales positive
    if (multiplier < 0).any():
        multiplier += np.abs(multiplier.min())
    hashable = (as_int * (10 ** multiplier)).astype(np.int64)
    md5 = util.md5_object(hashable)
    return md5


def face_ordering(mesh):
    """
    Return the size-order of every face in the input mesh.

    Triangles can be considered by the length order:
      [small edge, medium edge, large edge] (SML)
      [small edge, large edge,  medium edge] (SLM)

    This function returns [-1, 0, 1], depending on whether
    the triangle is SML or SLM, and 0 if M == L.

    The reason this is useful as it as a rare property that is
    invariant to translation and rotation but changes when a
    mesh is reflected or inverted. It is NOT invariant to
    different tesselations of the same surface.

    Parameters
    -------------
    mesh : trimesh.Trimesh
      Source geometry to calculate ordering on

    Returns
    --------------
    order : (len(mesh.faces), ) int
      Is each face SML (-1), SLM (+1), or M==L (0)
    """

    # the length of each edge in faces
    norms = mesh.edges_unique_length[
        mesh.edges_unique_inverse].reshape((-1, 3))

    # the per- row index of the shortest edge
    small = norms.argmin(axis=1)

    # the ordered index for the medium and large edge norm
    # arranged to reference flattened norms for indexing
    MLidx = np.column_stack((small + 1, small + 2)) % 3
    MLidx += (np.arange(len(small)) * 3).reshape((-1, 1))

    # subtract the two largest edge lengths from each other
    diff = np.subtract(*norms.reshape(-1)[MLidx.T])

    # mark by sign but keep zero values zero
    order = np.zeros(len(norms), dtype=np.int64)
    order[diff < tol.merge] = -1
    order[diff > tol.merge] = 1

    return order
