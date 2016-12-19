import numpy as np

from collections import deque
from copy import deepcopy

from pyembree import rtcore_scene
from pyembree.mesh_construction import TriangleMesh

from .. import util
from .. import intersections

class RayMeshIntersector:
    def __init__(self, geometry):
        self._geometry = geometry

        self._cache = util.Cache(id_function=self._geometry.crc)

    @util.cache_decorator
    def _scene(self):
        # set up the embree scene
        scene = rtcore_scene.EmbreeScene()
        mesh = TriangleMesh(scene, self._geometry.triangles)
        return scene
        
    def intersects_id(self,
                      ray_origins,
                      ray_directions,
                      return_all = False):
            
        # make sure input is float64 for embree
        ray_origins = np.asanyarray(deepcopy(ray_origins), dtype=np.float64)
        ray_directions = np.asanyarray(ray_directions, dtype=np.float64)
        ray_directions = util.unitize(ray_directions)

        # since we are constructing all hits, save them to a deque then 
        # stack into (depth, len(rays)) at the end
        faces_hit = deque()
        # the mask for which rays are still active
        current = np.ones(len(ray_origins), dtype=np.bool)

        if return_all:
            # how much to offset ray to transport to the other side of it
            ray_offset = ray_directions * 1e-8

            # grab the planes from triangles
            plane_origins = self._geometry.triangles[:,0,:]
            plane_normals = self._geometry.face_normals
            
        while True:
            # run the pyembree query
            query = self._scene.run(ray_origins[current],
                                    ray_directions[current])

            # if we don't need all of the hits, return the first one
            if not return_all:
                return query
            
            # basically we need to reduce the rays to the ones that hit something
            hit = query != -1

            if not hit.any():
                break

            # stack the query to include non- active rays so the dimensions stay the same
            query_stacked = np.ones(len(current), dtype=np.int32)*-1
            query_stacked[current] = query

            # append the stacked query to the list of hits
            faces_hit.append(query_stacked)

            # eliminate rays that didn't hit anything from future queries
            current_index = np.nonzero(current)[0][np.logical_not(hit)]
            current[current_index] = False

            # actual triangle indexes that were hit
            hit_triangle = query[hit]

            # transport the ray origin to where we hit something
            new_origins, valid = intersections.planes_lines(
                plane_origins=plane_origins[hit_triangle],
                plane_normals=plane_normals[hit_triangle],
                line_origins=ray_origins[current],
                line_directions=ray_directions[current])

            if not valid.any(): 
                break
            elif not valid.all():
                current[current_index[np.logical_not(valid)]] = False

            # move the ray origin to the other side of the triangle
            ray_origins[current] = new_origins + ray_offset[current]
        faces_hit = np.vstack(faces_hit)
        
        return faces_hit

    def intersects_any(self, 
                       ray_origins, 
                       ray_directions):
        first = self.intersects_id(ray_origins=ray_origins,
                                   ray_directions=ray_directions,
                                   return_all=False)
        hit = first != -1
        return hit
        
