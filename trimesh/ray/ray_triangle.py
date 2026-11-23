"""
A basic slow implementation of ray- triangle queries.
"""

import itertools

import numpy as np

from .. import caching, grouping
from ..constants import tol
from ..typed import ArrayLike, NDArray
from .ray_util import contains_points


class RayMeshIntersector:
    """
    An object to query a mesh for ray intersections.
    Precomputes a bounding volume hierarchy of the mesh's triangles.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._cache = caching.Cache(self.mesh.__hash__)

    @caching.cache_decorator
    def hierarchy(self):
        """
        The mesh's triangles in a bounding volume hierarchy.

        Returns
        ---------
        hierarchy
          From `bounding_volume_hierarchy(mesh.triangles)`.
        """
        return bounding_volume_hierarchy(self.mesh.triangles)

    def intersects_id(
        self,
        ray_origins,
        ray_directions,
        return_locations=False,
        multiple_hits=True,
        **kwargs,
    ):
        """
        Find the intersections between the current mesh and an
        array of rays.

        Parameters
        ------------
        ray_origins :  (m, 3) float
          Ray origin points
        ray_directions : (m, 3) float
          Ray direction vectors
        multiple_hits :  bool
          Consider multiple hits of each ray or not
        return_locations : bool
          Return hit locations or not

        Returns
        -----------
        index_triangle : (h,) int
          Index of triangles hit
        index_ray : (h,) int
          Index of ray that hit triangle
        locations : (h, 3) float
          [optional] Position of intersection in space
        """
        (index_tri, index_ray, locations) = ray_triangle_id(
            triangles=self.mesh.triangles,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=multiple_hits,
            hierarchy=self.hierarchy,
        )
        if return_locations:
            if len(index_tri) == 0:
                return index_tri, index_ray, locations
            # a hit on a shared edge or vertex is reported once, in ray order
            unique = np.sort(
                grouping.unique_rows(np.column_stack((locations, index_ray)))[0]
            )
            return index_tri[unique], index_ray[unique], locations[unique]
        return index_tri, index_ray

    def intersects_location(self, ray_origins, ray_directions, **kwargs):
        """
        Return unique cartesian locations where rays hit the mesh.
        If you are counting the number of hits a ray had, this method
        should be used as if only the triangle index is used on- edge hits
        will be counted twice.

        Parameters
        ------------
        ray_origins : (m, 3) float
          Ray origin points
        ray_directions : (m, 3) float
          Ray direction vectors

        Returns
        ---------
        locations : (n) sequence of (m,3) float
          Intersection points
        index_ray : (n,) int
          Array of ray indexes
        index_tri: (n,) int
          Array of triangle (face) indexes
        """
        (index_tri, index_ray, locations) = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=True,
            **kwargs,
        )
        return locations, index_ray, index_tri

    def intersects_first(self, ray_origins, ray_directions, **kwargs):
        """
        Find the index of the first triangle a ray hits.


        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        """

        (index_tri, index_ray) = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=False,
            multiple_hits=False,
            **kwargs,
        )

        # put the result into the form of "one triangle index per ray"
        result = np.ones(len(ray_origins), dtype=np.int64) * -1
        result[index_ray] = index_tri

        return result

    def intersects_any(self, ray_origins, ray_directions, **kwargs):
        """
        Find out if each ray hit any triangle on the mesh.

        Parameters
        ------------
        ray_origins : (m, 3) float
          Ray origin points
        ray_directions : (m, 3) float
          Ray direction vectors

        Returns
        ---------
        hit : (m,) bool
          Whether any ray hit any triangle on the mesh
        """
        _index_tri, index_ray = self.intersects_id(
            ray_origins, ray_directions, multiple_hits=False
        )
        return np.bincount(index_ray, minlength=len(ray_origins)) > 0

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh the behavior
        is undefined.

        Parameters
        ------------
        points : (n, 3) float
          Points in space

        Returns
        ---------
        contains : (n,) bool
          Whether point is inside mesh or not
        """

        return contains_points(self, points)


def ray_triangle_id(
    triangles,
    ray_origins,
    ray_directions,
    triangles_normal=None,
    tree=None,
    multiple_hits=True,
    hierarchy=None,
    batch_size=2**12,
):
    """
    Find the intersections between a group of triangles and rays

    Parameters
    -------------
    triangles : (n, 3, 3) float
      Triangles in space
    ray_origins : (m, 3) float
      Ray origin points
    ray_directions : (m, 3) float
      Ray direction vectors
    triangles_normal : None
      Unused, as hits come from the vertices alone
    tree : None
      Unused, as `hierarchy` finds the candidates
    multiple_hits : bool
      Consider multiple hits of each ray or not
    hierarchy : None or tuple
      From `bounding_volume_hierarchy(triangles)`, or built here
    batch_size : int
      The most rays traced at once, which bounds memory

    Returns
    -----------
    index_triangle : (h,) int
      Index of triangles hit, sorted by ray then distance
    index_ray : (h,) int
      Index of ray that hit triangle
    locations : (h, 3) float
      Position of intersection in space
    """
    triangles = np.asanyarray(triangles, dtype=np.float64).reshape((-1, 3, 3))
    origins = np.asanyarray(ray_origins, dtype=np.float64).reshape((-1, 3))
    directions = np.asanyarray(ray_directions, dtype=np.float64).reshape((-1, 3))
    if hierarchy is None:
        hierarchy = bounding_volume_hierarchy(triangles)
    order, levels = hierarchy

    # coordinates first so every gather by ray is contiguous
    rays = np.stack((origins.T, directions.T))
    every = np.arange(len(origins), dtype=np.int64)
    found = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for ray in np.split(every, every[batch_size::batch_size]):
            # every ray of the batch down one depth at a time, from below
            # the root as a ray that misses it misses both its children
            node = np.zeros(len(ray), dtype=np.int64)
            for level in levels:
                ray, node = np.repeat(ray, 2), (node[:, None] * 2 + [0, 1]).ravel()
                # NaN, from a padded node or a ray exactly on a padded face
                # it is parallel to, never enters: no triangle is that close
                origin, direction = np.take(rays, ray, axis=2)
                slabs = (np.take(level, node, axis=2) - origin) / direction
                near = np.maximum(np.minimum(*slabs).max(axis=0), 0.0)
                keep = near <= np.maximum(*slabs).min(axis=0)
                ray, node = ray[keep], node[keep]

            # Moller-Trumbore: distance and barycentric (u, v) in one pass
            triangle = order[node]
            vertices = np.take(triangles, triangle, axis=0).transpose(1, 2, 0)
            edge_one, edge_two = vertices[1:] - vertices[0]
            origin, direction = np.take(rays, ray, axis=2)
            offset = origin - vertices[0]
            cross_direction = np.cross(direction, edge_two, axis=0)
            cross_offset = np.cross(offset, edge_one, axis=0)
            determinant = (edge_one * cross_direction).sum(axis=0)
            u = (offset * cross_direction).sum(axis=0) / determinant
            v = (direction * cross_offset).sum(axis=0) / determinant
            distance = (edge_two * cross_offset).sum(axis=0) / determinant
            # a degenerate or parallel triangle divides by zero and fails
            hit = (u > -tol.zero) & (v > -tol.zero) & (u + v < 1.0 + tol.zero)
            # forward of the origin by the same projection as before
            hit &= distance * (direction * direction).sum(axis=0) > -1e-6
            found.append((triangle[hit], ray[hit], distance[hit]))
    index_tri, index_ray, distance = (np.concatenate(column) for column in zip(*found))

    # sorted by ray then distance, keeping only the nearest if asked
    sorting = np.lexsort((distance, index_ray))
    if not multiple_hits:
        sorting = sorting[np.unique(index_ray[sorting], return_index=True)[1]]
    index_tri, index_ray = index_tri[sorting], index_ray[sorting]
    locations = origins[index_ray] + directions[index_ray] * distance[sorting, None]
    return index_tri, index_ray, locations


def bounding_volume_hierarchy(
    triangles: ArrayLike,
) -> tuple[NDArray[np.int64], list[NDArray[np.float64]]]:
    """
    A complete binary tree of axis aligned boxes with one triangle
    in each leaf, sorted along a Morton curve of their centroids.
    Node `j` at one depth has the children `2j` and `2j + 1`.

    Parameters
    ------------
    triangles : (n, 3, 3) float
      Triangles in space.

    Returns
    ----------
    order : (n,) int
      The triangle in each leaf.
    levels : (depth,) list of (2, 3, 2**k) float
      The lower and upper corner of every node's box at each
      depth below the root, padded by `tol.merge`, and NaN where
      a node holds no triangle so that no ray ever enters it.
    """
    triangles = np.asanyarray(triangles, dtype=np.float64).reshape((-1, 3, 3))
    count = len(triangles)
    # (vertex, axis, triangle) so each reduction is between rows
    vertices = triangles.transpose(1, 2, 0)
    lower = np.minimum(np.minimum(vertices[0], vertices[1]), vertices[2])
    upper = np.maximum(np.maximum(vertices[0], vertices[1]), vertices[2])

    # 16 bits per axis interleaved into a 48 bit Morton code
    centroids = vertices[0] + vertices[1] + vertices[2]
    start = centroids.min(axis=1, initial=np.inf)
    span = max((centroids.max(axis=1, initial=-np.inf) - start).max(), tol.merge)
    scale = np.iinfo(np.uint16).max / span
    quantized = ((centroids.T - start) * scale).astype(">u2", order="C")
    bits = np.unpackbits(quantized.view(np.uint8).reshape((-1, 3, 2)), axis=2)
    code = np.packbits(bits.swapaxes(1, 2).reshape((-1, 48)), axis=1)
    order = np.lexsort(code.T[::-1]).astype(np.int64)

    # every leaf's box as (lower, -upper) so `fmin` merges siblings,
    # with NaN out to a power of two and no root as no ray tests it
    pad = tol.merge * np.abs(triangles).max(initial=0.0, where=np.isfinite(triangles))
    depth = max(count - 1, 1).bit_length()
    leaves = np.full((2, 3, 2**depth), np.nan)
    leaves[0, :, :count] = lower[:, order] - pad
    leaves[1, :, :count] = -upper[:, order] - pad
    merged = itertools.accumulate(
        range(depth - 1),
        lambda child, _: np.fmin(child[:, :, ::2], child[:, :, 1::2]),
        initial=leaves,
    )
    levels = [level * [[[1.0]], [[-1.0]]] for level in merged][::-1]
    return order, levels
