"""
ray/tinybvh.py
---------------

Ray queries with `trimkit` which uses `tinybvh`
and implements fast, multicore float64 ray checks.
"""

import numpy as np

from .. import caching, util
from ..typed import ArrayLike, Integer, NDArray
from .ray_util import contains_points


class RayMeshIntersector:
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
        max_hits: Integer | None = None,
        return_locations: bool = False,
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
        max_hits : None or int
          The most hits per ray, or every hit
        return_locations : bool
          Also return where each hit is

        Returns
        ---------
        index_tri : (m,) int
          Indexes of mesh.faces
        index_ray : (m,) int
          Indexes of ray
        locations : (m, 3) float
          Intersection points, only returned if `return_locations`
        """
        origins = np.asarray(ray_origins, dtype=np.float64)
        directions = np.asarray(ray_directions, dtype=np.float64)
        if not multiple_hits:
            max_hits = 1
        rays, _, faces, distances = self.scene.intersect(
            origins, directions, maximum_hits=max_hits
        )
        if not return_locations:
            return faces, rays
        # trimkit's distances are along the ray in world units
        along = util.unitize(directions)[rays] * distances.reshape((-1, 1))
        return faces, rays, origins[rays] + along

    def intersects_location(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        multiple_hits: bool = True,
        max_hits: Integer | None = None,
    ):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays
        multiple_hits : bool
          Every hit along each ray, or only its first
        max_hits : None or int
          The most hits per ray, or every hit

        Returns
        ---------
        locations : (m, 3) float
          Intersection points
        index_ray : (m,) int
          Indexes of ray
        index_tri : (m,) int
          Indexes of mesh.faces
        """
        index_tri, index_ray, locations = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=multiple_hits,
            max_hits=max_hits,
            return_locations=True,
        )
        return locations, index_ray, index_tri

    def intersects_first(
        self, ray_origins: ArrayLike, ray_directions: ArrayLike
    ) -> NDArray[np.int64]:
        """
        Find the index of the first triangle each ray hits.

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
        index_tri, index_ray = self.intersects_id(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )
        first = np.full(len(np.reshape(ray_origins, (-1, 3))), -1, dtype=np.int64)
        first[index_ray] = index_tri
        return first

    def intersects_any(
        self, ray_origins: ArrayLike, ray_directions: ArrayLike
    ) -> NDArray[np.bool_]:
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

    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh, behavior is undefined.

        Parameters
        ---------
        points : (n, 3) float
          Points in space

        Returns
        ---------
        contains : (n,) bool
          Whether point is inside mesh or not
        """
        return contains_points(self, points)

    def __getstate__(self):
        state = self.__dict__.copy()
        # the trimkit scene is rebuilt from the mesh rather than pickled
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = caching.Cache(id_function=self.mesh.__hash__)
