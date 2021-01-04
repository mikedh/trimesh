"""
ray/parent.py
-----------------

The abstract base class for RayMeshIntersector objects
which take meshes and allow queries to be run.
"""
import abc
from functools import wraps

from ..util import ABC
from .util import contains_points

import logging
log = logging.getLogger('trimesh')


class RayMeshParent(ABC):

    @abc.abstractmethod
    def intersects_location(self,
                            origins,
                            directions,
                            multiple_hits=True):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        origins : (n, 3) float
          Origins of rays
        directions : (n, 3) float
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

    @abc.abstractmethod
    def intersects_id(self,
                      origins,
                      directions,
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
        directions : (n, 3) float
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

    @abc.abstractmethod
    def intersects_first(
            self, origins, directions):
        """
        Find the index of the first triangle a ray hits.


        Parameters
        ----------
        origins : (n, 3) float
          Origins of rays
        directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        triangle_index : (n,) int
          Index of triangle ray hit, or -1 if not hit
        """

    @abc.abstractmethod
    def intersects_any(self,
                       origins,
                       directions):
        """
        Check if a list of rays hits the surface.


        Parameters
        -----------
        origins : (n, 3) float
          Origins of rays
        directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit the surface
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


def _kwarg_deprecated(function):
    """
    A decorator which replaces `ray_directions` with
    `directions` until we can deprecate the arguments.
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
                "Deprecation! The `ray_origins` kwarg for "
                "*all ray operations* has been renamed to `origins`. "
                "Versions of trimesh released after September 2021 "
                "will not include this warning and calls will fail if "
                "you don't rename your kwargs! "
                "Called from: `{}`.".format(function.__name__))
            kwargs['origins'] = kwargs.pop('ray_origins')
        if 'ray_directions' in kwargs:
            log.warning(
                "Deprecation! The `ray_directions` kwarg for "
                "*all ray operations* has been renamed to `directions`. "
                "Versions of trimesh released after September 2021 "
                "will not include this warning and will fail if "
                "you don't rename your kwargs! "
                "Called from `{}`.".format(function.__name__))
            kwargs['directions'] = kwargs.pop('ray_directions')
        # value not in cache so execute the function
        return function(*args, **kwargs)

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return kwarg_wrap
