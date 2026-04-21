"""
Ray queries using the embreex package with the
API wrapped to match our native raytracer.
"""

import numpy as np

# `pip install embreex` installs from wheels
from embreex import rtcore_scene
from embreex.mesh_construction import TriangleMesh

from .. import caching, intersections, util
from ..constants import log_time
from ..typed import ArrayLike, Integer
from .ray_util import contains_points

# embree operates on float32 values
_embree_dtype = np.float32

# when calculating multiple hits we offset the hit to
# advance the ray past the plane of the triangle we hit
# hardcode offset for rays larger than our resolution:
#   np.finfo(_embree_dtype).resolution * 10
_ray_offset = 1e-5


class RayMeshIntersector:
    def __init__(self, geometry, scale_to_box: bool = True):
        """
        Do ray- mesh queries.

        Parameters
        -------------
        geometry : Trimesh object
          Mesh to do ray tests on
        scale_to_box : bool
          If true, will scale mesh to approximate
          unit cube to avoid problems with extreme
          large or small meshes.
        """
        self.mesh = geometry
        self._scale_to_box = scale_to_box
        self._cache = caching.Cache(id_function=self.mesh.__hash__)

    @property
    def _scale(self):
        """
        Scaling factor for precision.
        """
        if self._scale_to_box:
            # scale vertices to approximately a cube to help with
            # numerical issues at very large/small scales
            scale = 100.0 / self.mesh.scale
        else:
            scale = 1.0
        return scale

    @caching.cache_decorator
    def _scene(self):
        """
        A cached version of the embreex scene.
        """
        return _EmbreeWrap(
            vertices=self.mesh.vertices, faces=self.mesh.faces, scale=self._scale
        )

    def intersects_location(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        multiple_hits: bool = True,
    ):
        """
        Return the location of where a ray hits a surface.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
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
        (index_tri, index_ray, locations) = self.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=multiple_hits,
            return_locations=True,
        )

        return locations, index_ray, index_tri

    @log_time
    def intersects_id(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        multiple_hits: bool = True,
        max_hits: Integer = 20,
        return_locations: bool = False,
    ):
        """
        Find the triangles hit by a list of rays, including
        optionally multiple hits along a single ray.


        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
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

        # make sure input is _dtype for embree
        ray_origins = np.array(ray_origins, dtype=np.float64)
        ray_directions = np.array(ray_directions, dtype=np.float64)
        if ray_origins.shape != ray_directions.shape:
            raise ValueError("Ray origin and direction don't match!")
        ray_directions = util.unitize(ray_directions)

        # since we are constructing all hits, save them to a deque then
        # stack into (depth, len(rays)) at the end
        result_triangle = []
        result_ray_idx = []
        result_locations = []

        if multiple_hits or return_locations:
            # how much to offset ray to transport to the other side of face
            ray_offsets = ray_directions * _ray_offset

            # grab the planes from triangles
            plane_origins = self.mesh.triangles[:, 0, :]
            plane_normals = self.mesh.face_normals

        # save the last hit by each ray in case the offsetting and precision
        # issue result in a duplicate being returned.
        last_hit = np.full(len(ray_origins), -1, dtype=np.int64)

        # if we're stuck on on triangle we need to offset more
        count_tri = np.zeros(len(mesh.faces), dtype=np.int64)
        # for translating boolean masks into indexes in the loop
        range_mask = np.arange(len(ray_origins))
        # the mask for which rays are still active
        current = np.ones(len(ray_origins), dtype=bool)

        # use a for loop rather than a while to ensure this exits
        # if a ray is offset from a triangle and then is reported
        # hitting itself this could get stuck on that one triangle
        for _ in range(max_hits):
            # run the embreex query
            # if you set output=1 it will calculate distance along
            # ray which is bizzarely slower than our calculation
            # TODO : FIXED IN embreex>=4.4.0rc1 ;)
            # when that's settled for a while we should probably
            # switch out our python hit location with theirs
            query = self._scene.run(ray_origins[current], ray_directions[current])

            # a ray that hit nothing will be -1
            hit = query != -1

            # we didn't hit anything so we can exit immediately
            if not hit.any():
                break

            # check for duplicates in case we're stuck
            hit_dupe = hit & (last_hit[current] == query)

            # save the index of the triangle this hit
            last_hit[current] = query

            # keep track of how many times we've hit something
            count_tri[query[hit]] += 1

            # it hit something and is unique
            hit_ok = hit & ~hit_dupe

            # if we don't need all of the hits return
            if not multiple_hits and not return_locations:
                # append the index of the triangle hit
                result_triangle.append(query[hit_ok])
                # append the index of the ray that hit
                result_ray_idx.append(range_mask[current][hit_ok])

                break

            # TODO : when embreex>=4.4.1rc1 has stabalized
            # we can use the `run(... output=True)` to get this
            # find the location of where the ray hit the triangle
            hit_triangle = query[hit]
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=ray_origins[current],
                line_directions=ray_directions[current],
            )

            hit[ok] &= valid

            if not valid.all():
                assert 0
                # since a plane intersection was invalid we have to go back and
                # fix some stuff, we pop the ray index and triangle index,
                # apply the valid mask then append it right back to keep our
                # indexes intact
                result_ray_idx.append(result_ray_idx.pop()[valid])
                result_triangle.append(result_triangle.pop()[valid])

                # update the current rays to reflect that we couldn't find a
                # new origin
                current[current_index_hit[np.logical_not(valid)]] = False

            # since we had to find the intersection point anyway we save it
            # even if we're not going to return it
            result_locations.extend(new_origins)

            if multiple_hits:
                # move the ray origin to the other side of the triangle
                ray_origins[current] = new_origins + ray_offsets[current]
                print(ray_origins[current])
            else:
                break

        # stack the dequeues into nice 1D numpy arrays
        index_tri = np.hstack(result_triangle)
        index_ray = np.hstack(result_ray_idx)

        if return_locations:
            locations = (
                np.zeros((0, 3), float)
                if len(result_locations) == 0
                else np.array(result_locations)
            )

            return index_tri, index_ray, locations
        return index_tri, index_ray

    @log_time
    def intersects_first(self, ray_origins, ray_directions):
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
        # make sure our arrays are in the `embree` dtype
        ray_origins = np.array(ray_origins, dtype=_embree_dtype)
        ray_directions = np.array(ray_directions, dtype=_embree_dtype)
        if ray_origins.shape != ray_directions.shape:
            raise ValueError("Ray origin and direction don't match!")
        ray_directions = util.unitize(ray_directions)

        return self._scene.run(ray_origins, ray_directions)

    def intersects_any(self, ray_origins, ray_directions):
        """
        Check if a list of rays hits the surface.


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

        first = self.intersects_first(
            ray_origins=ray_origins, ray_directions=ray_directions
        )
        hit = first != -1
        return hit

    def contains_points(self, points):
        """
        Check if a mesh contains a list of points, using ray tests.

        If the point is on the surface of the mesh, behavior is undefined.

        Parameters
        ---------
        points: (n, 3) points in space

        Returns
        ---------
        contains: (n,) bool
                         Whether point is inside mesh or not
        """
        return contains_points(self, points)

    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle cache
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add cache back since it doesn't exist in the pickle
        self._cache = caching.Cache(id_function=self.mesh.__hash__)

    def __deepcopy__(self, *args):
        return self.__copy__()

    def __copy__(self, *args):
        return RayMeshIntersector(geometry=self.mesh, scale_to_box=self._scale_to_box)


class _EmbreeWrap:
    """
    A light wrapper for Embreex scene objects which
    allows queries to be scaled to help with precision
    issues, as well as selecting the correct dtypes.
    """

    def __init__(self, vertices, faces, scale):
        scaled = np.array(vertices, dtype=np.float64)
        self.origin = scaled.min(axis=0)
        self.scale = float(scale)
        scaled = (scaled - self.origin) * self.scale

        self.scene = rtcore_scene.EmbreeScene()
        # assign the geometry to the scene
        TriangleMesh(
            scene=self.scene,
            vertices=scaled.astype(_embree_dtype),
            indices=faces.view(np.ndarray).astype(np.int32),
        )

    def run(self, origins, normals, **kwargs):
        scaled = (np.array(origins, dtype=np.float64) - self.origin) * self.scale

        return self.scene.run(
            scaled.astype(_embree_dtype), normals.astype(_embree_dtype), **kwargs
        )
