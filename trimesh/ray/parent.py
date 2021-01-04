"""
ray/parent.py
-----------------

The abstract base class for RayMeshIntersector objects
which take meshes and allow queries to be run.
"""
import abc
from ..util import ABC
from .util import contains_points


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
    def intersects_first(self,
                         origins,
                         directions):
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

        origins = np.asanyarray(deepcopy(origins))
        directions = np.asanyarray(directions)

        triangle_index = self._scene.run(origins,
                                         directions)
        return triangle_index

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

        first = self.intersects_first(origins=origins,
                                      directions=directions)
        hit = first != -1
        return hit

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
