"""
ray/tinybvh.py
---------------

Ray queries with `trimkit` which uses `tinybvh`
and implements fast, multicore float64 ray checks.
"""

import numpy as np

from .. import caching, util
from ..typed import ArrayLike, Integer, NDArray
from .parent import RayParent


class RayMeshIntersector(RayParent):
    def __init__(self, mesh):
        """
        Do ray-mesh queries with `trimkit` which
        packages `tinybvh`.

        Parameters
        -------------
        mesh : Trimesh
          Mesh to do ray tests on
        """
        self.mesh = mesh
        self._cache = caching.Cache(id_function=self.mesh.__hash__)

    @caching.cache_decorator
    def scene(self):
        """
        The mesh built once into a `trimkit.RayScene`.

        Returns
        ---------
        scene : trimkit.RayScene
          Rebuilt only when the mesh changes.
        """
        from trimkit import RayScene

        return RayScene(
            [(self.mesh.vertices.view(np.ndarray), self.mesh.faces.view(np.ndarray))]
        )

    def intersects_id(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        multiple_hits: bool = True,
        return_locations: bool = False,
        max_hits: Integer | None = None,
        return_instance: bool = False,
        **kwargs,
    ):
        """
        Find the triangles hit by each ray, nearest first.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays
        multiple_hits : bool
          Every hit along each ray, or only its first
        return_locations : bool
          Also return where each hit is
        max_hits : None or int
          The most hits per ray, or every hit
        return_instance : bool
          Also return the instance of each hit, always 0

        Returns
        ---------
        index_tri : (m,) int
          Indexes of mesh.faces
        index_ray : (m,) int
          Indexes of ray
        locations : (m, 3) float
          Intersection points, only returned if `return_locations`
        index_instance : (m,) int
          Zeros, only returned if `return_instance`
        """
        origins = np.asarray(ray_origins, dtype=np.float64)
        directions = np.asarray(ray_directions, dtype=np.float64)
        if not multiple_hits:
            max_hits = 1
        return hits_result(
            origins,
            directions,
            self.scene.intersect(origins, directions, maximum_hits=max_hits),
            return_locations=return_locations,
            return_instance=return_instance,
        )

    def intersects_any(self, ray_origins: ArrayLike, ray_directions: ArrayLike):
        """
        Check if each ray hits the surface.

        Parameters
        -----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit the surface
        """
        return self.scene.intersect_any(
            np.asarray(ray_origins, dtype=np.float64),
            np.asarray(ray_directions, dtype=np.float64),
        )


def hits_result(
    origins: NDArray[np.float64],
    directions: NDArray[np.float64],
    hits: tuple,
    return_locations: bool,
    return_instance: bool,
) -> tuple:
    """
    A `trimkit.RayScene.intersect` answer as `RayParent.intersects_id` returns it.

    Parameters
    ----------
    origins : (n, 3) float
      Origins of the rays cast.
    directions : (n, 3) float
      Directions of the rays cast.
    hits
      `(rays, instances, faces, distances)` from trimkit.
    return_locations
      Include where each hit is.
    return_instance
      Include the instance of each hit.

    Returns
    ---------
    result : tuple
      `(index_tri, index_ray[, locations][, index_instance])`
    """
    rays, instances, faces, distances = hits
    result = [faces, rays]
    if return_locations:
        # trimkit's distances are along the ray in world units
        along = util.unitize(directions)[rays] * distances.reshape((-1, 1))
        result.append(origins[rays] + along)
    if return_instance:
        result.append(instances)
    return tuple(result)
