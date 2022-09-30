"""
A basic slow implementation of ray- triangle queries.
"""
import numpy as np

from .. import util
from .. import caching
from .. import grouping
from .. import intersections
from .. import triangles as triangles_mod

from ..constants import tol
from .parent import RayParent, _kwarg_deprecate


class RayMeshIntersector(RayParent):
    """
    An object to query a mesh for ray intersections using basic
    numpy, precomputes an r-tree for each triangle on the mesh.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._cache = caching.Cache(self.mesh.crc)

    def __repr__(self):
        return 'basic.RayMesh'

    @_kwarg_deprecate
    def intersects_id(self,
                      origins,
                      vectors,
                      return_locations=False,
                      multiple_hits=True,
                      **kwargs):
        # inherits docstring from parent
        (index_tri,
         index_ray,
         locations) = triangle_id(
             triangles=self.mesh.triangles,
             origins=origins,
             vectors=vectors,
             tree=self.mesh.triangles_tree,
             multiple_hits=multiple_hits,
             triangles_normal=self.mesh.face_normals)
        if return_locations:
            if len(index_tri) == 0:
                return index_tri, index_ray, locations
            unique = grouping.unique_rows(
                np.column_stack((locations, index_ray)))[0]
            return index_tri[unique], index_ray[unique], locations[unique]
        return index_tri, index_ray


def triangle_id(
        triangles,
        origins,
        vectors,
        triangles_normal=None,
        tree=None,
        multiple_hits=True):
    """
    Find the intersections between a group of triangles
    and rays.

    Parameters
    -------------
    triangles : (n, 3, 3) float
      Triangles in space
    origins : (m, 3) float
      Ray origin points
    vectors : (m, 3) float
      Ray direction vectors
    triangles_normal : (n, 3) float
      Normal vector of triangles, optional
    tree : rtree.Index
      Rtree object holding triangle bounds

    Returns
    -----------
    index_triangle : (h,) int
      Index of triangles hit
    index_ray : (h,) int
      Index of ray that hit triangle
    locations : (h, 3) float
      Position of intersection in space
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    origins = np.asanyarray(origins, dtype=np.float64)
    vectors = np.asanyarray(vectors, dtype=np.float64)

    # if we didn't get passed an r-tree for the bounds of each
    # triangle create one here
    if tree is None:
        tree = triangles_mod.bounds_tree(triangles)

    # find the list of likely triangles and which ray they
    # correspond with, via rtree queries
    candidates, ids = ray_triangle_candidates(
        origins=origins,
        vectors=vectors,
        tree=tree)

    # get subsets which are corresponding rays and triangles
    # (c,3,3) triangle candidates
    triangle_candidates = triangles[candidates]
    # (c,3) origins and vectors for the rays
    line_origins = origins[ids]
    line_vectors = vectors[ids]

    # get the plane origins and normals from the triangle candidates
    plane_origins = triangle_candidates[:, 0, :]
    if triangles_normal is None:
        plane_normals, triangle_ok = triangles_mod.normals(
            triangle_candidates)
        if not triangle_ok.all():
            raise ValueError('Invalid triangles!')
    else:
        plane_normals = triangles_normal[candidates]

    # find the intersection location of the rays with the planes
    location, valid = intersections.planes_lines(
        plane_origins=plane_origins,
        plane_normals=plane_normals,
        line_origins=line_origins,
        line_vectors=line_vectors)

    if (len(triangle_candidates) == 0 or
            not valid.any()):
        # we got no hits so return early with empty array
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64))

    # find the barycentric coordinates of each plane intersection on the
    # triangle candidates
    barycentric = triangles_mod.points_to_barycentric(
        triangle_candidates[valid], location)

    # the plane intersection is inside the triangle if all barycentric
    # coordinates are between 0.0 and 1.0
    hit = np.logical_and(
        (barycentric > -tol.zero).all(axis=1),
        (barycentric < (1 + tol.zero)).all(axis=1))

    # the result index of the triangle is a candidate with a valid
    # plane intersection and a triangle which contains the plane
    # intersection point
    index_tri = candidates[valid][hit]
    # the ray index is a subset with a valid plane intersection and
    # contained by a triangle
    index_ray = ids[valid][hit]
    # locations are already valid plane intersections, just mask by hits
    location = location[hit]

    # only return points that are forward from the origin
    vector = location - origins[index_ray]
    distance = util.diagonal_dot(
        vector, vectors[index_ray])
    forward = distance > -1e-6

    index_tri = index_tri[forward]
    index_ray = index_ray[forward]
    location = location[forward]
    distance = distance[forward]

    if multiple_hits:
        return index_tri, index_ray, location

    # since we are not returning multiple hits, we need to
    # figure out which hit is first
    if len(index_ray) == 0:
        return index_tri, index_ray, location

    # find the first hit
    first = np.array(
        [g[distance[g].argmin()] for g in
         grouping.group(index_ray)])

    return index_tri[first], index_ray[first], location[first]


def ray_triangle_candidates(origins, vectors, tree):
    """
    Do broad- phase search for triangles that the rays
    may intersect.

    Does this by creating a bounding box for the ray as it
    passes through the volume occupied by the tree

    Parameters
    ------------
    origins : (m, 3) float
      Ray origin points
    vectors : (m, 3) float
      Ray direction vectors
    tree : rtree.Index
       Contains AABB of each triangle

    Returns
    ----------
    candidates : (n,) int
      Triangle indexes
    id : (n,) int
      Corresponding ray index for a triangle candidate
    """
    bounding = ray_bounds(origins=origins,
                          vectors=vectors,
                          bounds=tree.bounds)
    candidates = [[]] * len(origins)
    ids = [[]] * len(origins)

    for i, bounds in enumerate(bounding):
        candidates[i] = np.array(list(tree.intersection(bounds)),
                                 dtype=np.int64)
        ids[i] = np.ones(len(candidates[i]), dtype=np.int64) * i

    ids = np.hstack(ids)
    candidates = np.hstack(candidates)

    return candidates, ids


def ray_bounds(origins,
               vectors,
               bounds,
               buffer_dist=1e-5):
    """
    Given a set of rays and a bounding box for the volume of
    interest where the rays will be passing through, find the
    bounding boxes of the rays as they pass through the volume.

    Parameters
    ------------
    origins : (m, 3) float
      Ray origin points
    vectors : (m, 3) float
      Ray direction vectors
    bounds : (2, 3) float
      Bounding box (min, max)
    buffer_dist : float
      Distance to pad zero width bounding boxes

    Returns
    ---------
    bounding : (n) set
      AABB of rays passing through volume
    """

    origins = np.asanyarray(origins, dtype=np.float64)
    vectors = np.asanyarray(vectors, dtype=np.float64)

    # bounding box we are testing against
    bounds = np.asanyarray(bounds)

    # find the primary axis of the vector
    axis = np.abs(vectors).argmax(axis=1)
    axis_bound = bounds.reshape((2, -1)).T[axis]
    axis_ori = np.array([origins[i][a]
                         for i, a in enumerate(axis)]).reshape((-1, 1))
    axis_dir = np.array([vectors[i][a]
                         for i, a in enumerate(axis)]).reshape((-1, 1))

    # parametric equation of a line
    # point = direction*t + origin
    # p = dt + o
    # t = (p-o)/d
    t = (axis_bound - axis_ori) / axis_dir

    # prevent the bounding box from including triangles
    # behind the ray origin
    t[t < buffer_dist] = buffer_dist

    # the value of t for both the upper and lower bounds
    t_a = t[:, 0].reshape((-1, 1))
    t_b = t[:, 1].reshape((-1, 1))

    # the cartesion point for where the line hits the plane defined by
    # axis
    on_a = (vectors * t_a) + origins
    on_b = (vectors * t_b) + origins

    on_plane = np.column_stack(
        (on_a, on_b)).reshape(
        (-1, 2, vectors.shape[1]))

    bounding = np.hstack((on_plane.min(axis=1),
                          on_plane.max(axis=1)))
    # pad the bounding box by TOL_BUFFER
    # not sure if this is necessary, but if the ray is  axis aligned
    # this function will otherwise return zero volume bounding boxes
    # which may or may not screw up the r-tree intersection queries
    bounding += np.array([-1, -1, -1, 1, 1, 1]) * buffer_dist

    return bounding
