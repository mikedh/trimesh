"""
Ray queries using the pyembree package with the
API wrapped to match our native raytracer.
"""
from . import parent

import numpy as np

# bindings for embree3
import embree

raise ValueError('nah')


class RayMeshIntersector(parent.RayMeshParent):

    def __init__(self,
                 geometry,
                 scale_to_box=True):
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

    def intersects_location(self,
                            origins,
                            directions,
                            multiple_hits=True):
        # inherits docstring from parent
        (index_tri,
         index_ray,
         locations) = self.intersects_id(
             origins=origins,
             directions=directions,
             multiple_hits=multiple_hits,
             return_locations=True)

        return locations, index_ray, index_tri

    def intersects_id(self,
                      origins,
                      directions,
                      multiple_hits=True,
                      max_hits=20,
                      return_locations=False):
        # inherits docstring from parent
        origins = np.asanyarray(
            deepcopy(origins),
            dtype=np.float64)
        directions = np.asanyarray(directions,
                                   dtype=np.float64)
        directions = util.unitize(directions)

        # since we are constructing all hits save them to a
        # deque then stack into (depth, len(rays)) at the end
        result_triangle = []
        result_idx = []
        result_locations = []

        # the mask for which rays are still active
        current = np.ones(len(origins), dtype=np.bool)

        if multiple_hits or return_locations:
            # how much to offset ray to transport to the other side of face
            distance = np.clip(_offset_factor * self._scale,
                               _offset_floor,
                               np.inf)
            offsets = directions * distance

            # grab the planes from triangles
            plane_origins = self.mesh.triangles[:, 0, :]
            plane_normals = self.mesh.face_normals

        # use a for loop rather than a while to ensure this exits
        # if a ray is offset from a triangle and then is reported
        # hitting itself this could get stuck on that one triangle
        for query_depth in range(max_hits):
            # run the pyembree query
            # if you set output=1 it will calculate distance along
            # ray, which is bizzarely slower than our calculation
            query = self._scene.run(
                origins[current],
                directions[current])

            # basically we need to reduce the rays to the ones that hit
            # something
            hit = query != -1
            # which triangle indexes were hit
            hit_triangle = query[hit]

            # eliminate rays that didn't hit anything from future queries
            current_index = np.nonzero(current)[0]
            current_index_no_hit = current_index[np.logical_not(hit)]
            current_index_hit = current_index[hit]
            current[current_index_no_hit] = False

            # append the triangle and ray index to the results
            result_triangle.append(hit_triangle)
            result_idx.append(current_index_hit)

            # if we don't need all of the hits, return the first one
            if ((not multiple_hits and
                 not return_locations) or
                    not hit.any()):
                break

            # find the location of where the ray hit the triangle plane
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=origins[current],
                line_directions=directions[current])

            if not valid.all():
                # since a plane intersection was invalid we have to go back and
                # fix some stuff, we pop the ray index and triangle index,
                # apply the valid mask then append it right back to keep our
                # indexes intact
                result_idx.append(result_idx.pop()[valid])
                result_triangle.append(result_triangle.pop()[valid])

                # update the current rays to reflect that we couldn't find a
                # new origin
                current[current_index_hit[np.logical_not(valid)]] = False

            # since we had to find the intersection point anyway we save it
            # even if we're not going to return it
            result_locations.extend(new_origins)

            if multiple_hits:
                # move the ray origin to the other side of the triangle
                origins[current] = new_origins + offsets[current]
            else:
                break

        # stack the deques into nice 1D numpy arrays
        index_tri = np.hstack(result_triangle)
        index_ray = np.hstack(result_idx)

        if return_locations:
            locations = (
                np.zeros((0, 3), float) if len(result_locations) == 0
                else np.array(result_locations))

            return index_tri, index_ray, locations
        return index_tri, index_ray

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

        origins = np.asanyarray(origins)
        directions = np.asanyarray(directions)

        from IPython import embed
        embed()

        triangle_index = self._scene.run(origins,
                                         directions)
        return triangle_index

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
