"""
ray/parent.py
---------------

The abstract base class every ray-mesh intersector implements.
"""

import abc

import numpy as np

from .. import caching
from ..typed import ArrayLike, NDArray
from ..util import ABC
from .ray_util import contains_points


class RayParent(ABC):
    """
    Ray queries against a mesh or every instance in a scene: an
    intersector implements `intersects_id` and `intersects_any`,
    and inherits the rest.
    """

    @abc.abstractmethod
    def intersects_id(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        multiple_hits: bool = True,
        return_locations: bool = False,
        return_instance: bool = False,
        **kwargs,
    ):
        """
        Find the triangles hit by each ray.

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
        return_instance : bool
          Also return which instance each hit is on

        Returns
        ---------
        index_tri : (m,) int
          Indexes of the faces of the instance's mesh
        index_ray : (m,) int
          Indexes of ray
        locations : (m, 3) float
          Intersection points, only returned if `return_locations`
        index_instance : (m,) int
          Instance of each hit, always 0 for a mesh and a position
          in `nodes_geometry` for a scene, only returned if `return_instance`
        """

    @abc.abstractmethod
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

    def intersects_location(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        return_instance: bool = False,
        **kwargs,
    ):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays
        return_instance : bool
          Also return which instance each hit is on

        Returns
        ---------
        locations : (m, 3) float
          Intersection points
        index_ray : (m,) int
          Indexes of ray
        index_tri : (m,) int
          Indexes of the faces of the instance's mesh
        index_instance : (m,) int
          Instance of each hit, only returned if `return_instance`
        """
        index_tri, index_ray, locations, *instance = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=True,
            return_instance=return_instance,
            **kwargs,
        )
        return (locations, index_ray, index_tri, *instance)

    def intersects_first(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        return_instance: bool = False,
        **kwargs,
    ):
        """
        Find the index of the first triangle each ray hits.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays
        return_instance : bool
          Also return which instance each first hit is on

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        instance_index : (n,) int
          Instance of each first hit, or -1 if not hit,
          only returned if `return_instance`
        """
        index_tri, index_ray, *instance = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=False,
            multiple_hits=False,
            return_instance=return_instance,
            **kwargs,
        )
        first = np.full((1 + len(instance), len(ray_origins)), -1, dtype=np.int64)
        first[:, index_ray] = [index_tri, *instance]
        if return_instance:
            return first[0], first[1]
        return first[0]

    @property
    def bounds(self) -> NDArray[np.float64]:
        """
        The axis aligned bounds of everything rays can hit.

        Returns
        ----------
        bounds : (2, 3) float
          Minimum and maximum corner.
        """
        return self.mesh.bounds

    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """
        Check if the mesh contains each point, using ray tests.

        If the point is on the surface of the mesh, behavior is undefined.

        Parameters
        ---------
        points : (n, 3) float
          Points in space

        Returns
        ---------
        contains : (n,) bool
          Whether each point is inside the mesh
        """
        return contains_points(self, points)

    def __getstate__(self):
        # the cache rebuilds from the mesh rather than pickling
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = caching.Cache(id_function=self.mesh.__hash__)

    def __copy__(self, *args):
        return type(self)(self.mesh)

    def __deepcopy__(self, *args):
        return self.__copy__()
