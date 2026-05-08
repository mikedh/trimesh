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

# scale-aware base ray offset to step past hit triangles above f32 ULP
_ray_offset_factor = 1e-6
_ray_offset_floor = 1e-8


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
        max_hits: Integer = 100,
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

        # stack results for multiple hits into a sequence
        result_triangle = [np.zeros(0, dtype=np.int64)]
        result_ray_idx = [np.zeros(0, dtype=np.int64)]
        result_locations = [np.zeros((0, 3), dtype=np.float64)]

        if multiple_hits or return_locations:
            # how much to offset ray to transport to the other side of face
            base_offset = max(_ray_offset_floor, self.mesh.scale * _ray_offset_factor)
            ray_offsets = ray_directions * base_offset

            # grab the planes from triangles
            plane_origins = self.mesh.triangles[:, 0, :]
            plane_normals = self.mesh.face_normals

        # what each ray hit last iteration; -1 means nothing, used to
        # detect a ray stuck on the same face due to precision issues
        last_hit = np.full(len(ray_origins), -1, dtype=np.int64)
        # absolute indices of rays still being queried; shrinks each
        # iteration as rays miss, escape, or get culled
        live = np.arange(len(ray_origins))

        # use a for loop rather than a while to ensure this exits
        # if a ray is offset from a triangle and then is reported
        # hitting itself this could get stuck on that one triangle
        for _depth in range(max_hits):
            # if you set output=1 embreex returns distance along the ray
            # which is bizarrely slower than our own plane-line calc
            # TODO: switch `run(..., output=True)` once embreex>=4.4.0rc1 is stable
            query = self._scene.run(ray_origins[live], ray_directions[live])
            hit = query != -1
            if not hit.any():
                break

            # absolute indices and triangle indices for rays that hit
            hit_rays = live[hit]
            hit_tris = query[hit]

            # first-hit-only fast path: no duplicates or locations to track
            if not multiple_hits and not return_locations:
                result_triangle.append(hit_tris)
                result_ray_idx.append(hit_rays)
                break

            # rays that hit the same triangle as last iteration are stuck
            dupe = last_hit[hit_rays] == hit_tris
            # store the last hit so we can track duplicates
            last_hit[hit_rays] = hit_tris
            # subset to separate duplicates and non-duplicates
            ok_rays = hit_rays[~dupe]
            ok_tris = hit_tris[~dupe]
            dupe_rays = hit_rays[dupe]

            # compute where clean hits actually land on their triangle;
            # planes_lines silently drops near-parallel rays, so `valid`
            # shortens new_origins — we must trim ok_rays / ok_tris to match
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[ok_tris],
                plane_normals=plane_normals[ok_tris],
                line_origins=ray_origins[ok_rays],
                line_directions=ray_directions[ok_rays],
            )
            ok_rays, ok_tris = ok_rays[valid], ok_tris[valid]

            result_locations.append(new_origins)
            result_triangle.append(ok_tris)
            result_ray_idx.append(ok_rays)

            if not multiple_hits:
                break

            # clean hits step onto the new face with a fresh base offset
            ray_origins[ok_rays] = new_origins + ray_offsets[ok_rays]
            ray_offsets[ok_rays] = ray_directions[ok_rays] * base_offset
            # stuck rays double their offset and step further to try to clear
            ray_offsets[dupe_rays] *= 2.0
            ray_origins[dupe_rays] += ray_offsets[dupe_rays]

            # carry forward only rays we successfully advanced;
            # dropped rays (misses, near-parallel planes) die here
            live = np.concatenate([ok_rays, dupe_rays])

        index_tri = np.concatenate(result_triangle)
        index_ray = np.concatenate(result_ray_idx)
        if return_locations:
            return index_tri, index_ray, np.concatenate(result_locations)
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
