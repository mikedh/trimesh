import numpy as np

from collections import deque
from copy import deepcopy

from pyembree import rtcore_scene
from pyembree.mesh_construction import TriangleMesh

from .. import util
from .. import intersections

# based on an internal tolerance of embree?
# 1e-4 definetly doesn't work
_ray_offset_distance = 5e-3

class RayMeshIntersector:

    def __init__(self, geometry):
        self._geometry = geometry
        self._cache = util.Cache(id_function=self._geometry.crc)

    @util.cache_decorator
    def _scene(self):
        '''
        A cached version of the pyembree scene.
        '''
        scene = rtcore_scene.EmbreeScene()
        mesh = TriangleMesh(scene, self._geometry.triangles)
        return scene

    def intersects_location(self,
                            ray_origins,
                            ray_directions,
                            multiple_hits=True):
        '''
        Return the location of where a ray hits a surface.

        Arguments
        ----------
        ray_origins:    (n,3) float, origins of rays
        ray_directions: (n,3) float, direction (vector) of rays


        Returns
        ---------
        locations: (m,3) float, points where the ray intersects the surface
        ray_index: (m,) int, index of which ray location is from
        '''
        (index_tri,
         index_ray,
         locations) = self.intersects_id(ray_origins=ray_origins,
                                         ray_directions=ray_directions,
                                         multiple_hits=multiple_hits,
                                         return_locations=True)
        
        return locations, index_ray

    def intersects_id(self,
                      ray_origins,
                      ray_directions,
                      multiple_hits=True,
                      return_locations=False):
        '''
        Find the triangles hit by a list of rays, including optionally 
        multiple hits along a single ray. 

        Arguments
        ----------
        ray_origins:      (n,3) float, origins of rays
        ray_directions:   (n,3) float, direction (vector) of rays
        multiple_hits:    bool, if True will return every hit along the ray
                                if False will only return first hit
        return_locations: bool, should we return hit locations or not

        Returns
        ----------
        index_tri: (m,) int, index of triangle the ray hit
        index_ray: (m,) int, index of ray
        locations: (m,3) float, locations in space
        '''
        # make sure input is float64 for embree
        ray_origins = np.asanyarray(deepcopy(ray_origins), dtype=np.float64)
        ray_directions = np.asanyarray(ray_directions, dtype=np.float64)
        ray_directions = util.unitize(ray_directions)

        # since we are constructing all hits, save them to a deque then
        # stack into (depth, len(rays)) at the end
        result_triangle = deque()
        result_ray_idx = deque()
        result_locations = deque()

        # the mask for which rays are still active
        current = np.ones(len(ray_origins), dtype=np.bool)

        if multiple_hits or return_locations:
            # how much to offset ray to transport to the other side of it
            ray_offset = ray_directions * _ray_offset_distance

            # grab the planes from triangles
            plane_origins = self._geometry.triangles[:, 0, :]
            plane_normals = self._geometry.face_normals

        while True:
            # run the pyembree query
            query = self._scene.run(ray_origins[current],
                                    ray_directions[current])
           
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
            result_ray_idx.append(current_index_hit)

            # if we don't need all of the hits, return the first one
            if ((not multiple_hits and
                 not return_locations) or
                    not hit.any()):
                break

            # find the location of where the ray hit the triangle plane
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=ray_origins[current],
                line_directions=ray_directions[current])

            if not valid.all():
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
                ray_origins[current] = new_origins + ray_offset[current]
            else:
                break

        # stack the deques into nice 1D numpy arrays
        index_tri = np.hstack(result_triangle)
        index_ray = np.hstack(result_ray_idx)

        if return_locations:
            return index_tri, index_ray, np.array(result_locations)
        return index_tri, index_ray

    def intersects_first(self,
                         ray_origins,
                         ray_directions):
        '''
        Find the index of the first triangle a ray hits. 


        Arguments
        ----------
        ray_origins:    (n,3) float, origins of rays
        ray_directions: (n,3) float, direction (vector) of rays

        Returns
        ----------
        triangle_index: (n,) int, index of triangle ray hit, or -1 if not hit
        '''

        ray_origins = np.asanyarray(deepcopy(ray_origins), dtype=np.float64)
        ray_directions = np.asanyarray(ray_directions, dtype=np.float64)

        triangle_index = self._scene.run(ray_origins, ray_directions)
        return triangle_index

    def intersects_any(self,
                       ray_origins,
                       ray_directions):
        '''
        Check if a list of rays hits the surface.


        Arguments
        ----------
        ray_origins:    (n,3) float, origins of rays
        ray_directions: (n,3) float, direction (vector) of rays

        Returns
        ----------
        hit:            (n,) bool, did each ray hit the surface
        '''

        first = self.intersects_first(ray_origins=ray_origins,
                                      ray_directions=ray_directions)
        hit = first != -1
        return hit
