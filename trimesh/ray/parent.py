"""
ray/parent.py
-----------------

The abstract base class for RayMeshIntersector objects
which take meshes and allow queries to be run.
"""
import abc
from functools import wraps

import numpy as np

from ..util import ABC
from .util import contains_points

import logging
log = logging.getLogger('trimesh')


def _kwarg_deprecate(function):
    """
    A decorator which replaces `ray_vectors` with
    `vectors` until we can deprecate the arguments.
    """
    # use wraps to preserve docstring
    @wraps(function)
    def kwarg_wrap(*args, **kwargs):
        """
        Only execute the function if its value isn't stored
        in cache already.
        """
        if 'ray_origins' in kwargs:
            log.warning(
                "Deprecation! The `ray_origins` kwarg for " +
                "*all ray operations* has been renamed to `origins`. " +
                "Versions of trimesh released after December 2022" +
                "will not include this warning and calls will fail if " +
                "you don't rename your kwargs! " +
                "Called from: `{}`.".format(function.__name__))
            kwargs['origins'] = kwargs.pop('ray_origins')
        if 'ray_directions' in kwargs:
            log.warning(
                "Deprecation! The `ray_vectors` kwarg for " +
                "*all ray operations* has been renamed to `vectors`. " +
                "Versions of trimesh released after December 2022 " +
                "will not include this warning and will fail if " +
                "you don't rename your kwargs! " +
                "Called from `{}`.".format(function.__name__))
            kwargs['vectors'] = kwargs.pop('ray_directions')
        # value not in cache so execute the function
        return function(*args, **kwargs)

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return kwarg_wrap


class RayParent(ABC):

    @_kwarg_deprecate
    def intersects_location(self,
                            origins,
                            vectors,
                            **kwargs):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        origins : (n, 3) float
          Origins of rays
        vectors : (n, 3) float
          Direction (vector) of rays

        Returns
        ---------
        locations : (m) sequence of (p, 3) float
          Intersection points
        index_ray : (m,) int
          Indexes of ray
        index_tri : (m,) int
          Indexes of mesh.faces
        """

        # inherits docstring from parent
        (index_tri,
         index_ray,
         locations) = self.intersects_id(
             origins=origins,
             vectors=vectors,
             return_locations=True,
             **kwargs)
        return locations, index_ray, index_tri

    @_kwarg_deprecate
    def intersects_first(self, origins, vectors, **kwargs):
        """
        Find the index of the first triangle a ray hits.

        Parameters
        ----------
        origins : (n, 3) float
          Origins of rays
        vectors : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        """
        # inherits docstring from parent
        (index_tri,
         index_ray) = self.intersects_id(
             origins=origins,
             vectors=vectors,
             return_locations=False,
             multiple_hits=False,
             **kwargs)

        # put the result into the form of
        # "one triangle index per ray"
        result = np.ones(len(origins), dtype=np.int64) * -1
        result[index_ray] = index_tri

        return result

    @_kwarg_deprecate
    def intersects_any(self, origins, vectors):
        """
        Check if rays hit anything.


        Parameters
        -----------
        origins : (n, 3) float
          Origins of rays
        vectors : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit the surface
        """
        # inherits docstring from parent
        index_tri, index_ray = self.intersects_id(
            origins, vectors)
        hit_any = np.zeros(len(origins), dtype=bool)
        hit_idx = np.unique(index_ray)
        if len(hit_idx) > 0:
            hit_any[hit_idx] = True
        return hit_any

    @abc.abstractmethod
    def intersects_id(self,
                      origins,
                      vectors,
                      multiple_hits=True,
                      max_hits=20,
                      return_locations=False):
        """
        Find the triangles hit by a list of rays, including
        optionally multiple hits along a single ray.


        Parameters
        ----------
        origins : (n, 3) float
          Origins of rays
        vectors : (n, 3) float
          Direction (vector) of rays
        multiple_hits : bool
          If True will return every hit along the ray
          If False will only return first hit
        max_hits : int
          Maximum number of hits per ray
        return_locations : bool
          Should we return hit locations or not

        Returns
        ---------
        index_tri : (m,) int
          Indexes of mesh.faces
        index_ray : (m,) int
          Indexes of ray
        locations : (m) sequence of (p, 3) float
          Intersection points, only returned if return_locations
        """

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points using
        ray tests. If the point is on the surface of the mesh
        behavior is undefined.

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

    @abc.abstractmethod
    def __repr__(self):
        pass
