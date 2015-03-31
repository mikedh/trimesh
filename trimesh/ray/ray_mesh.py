import numpy as np
import time

import rtree.index as rtree
from copy import deepcopy

from ..constants       import *
from ..intersections   import plane_line_intersection
from ..geometry        import unitize
from .ray_triangle_cpu import ray_triangle_boolean

class RayMeshIntersector:
    '''
    An object to query a mesh for ray intersections. 
    Precomputes an r-tree for each triangle on the mesh.
    '''
    def __init__(self, mesh):
        self.mesh = mesh

        # triangles are (n, 3, 3)
        self.triangles = mesh.vertices[mesh.faces]

        # create an r-tree that contains every triangle
        # this is moderately expensive and can be reused
        self.tree      = create_tree(self.triangles)

    def intersect_boolean(self, rays):
        '''
        Find out whether the rays in question hit any triangle on the mesh.

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions

        Returns
        ---------
        intersections: (n) boolean array of whether or not the ray hit a triangle
        '''
        #tic = time_function()
        ray_candidates = ray_triangle_candidates(rays = rays, 
                                                 tree = self.tree)
        intersections  = ray_triangle_boolean(triangles      = self.triangles, 
                                              rays           = rays, 
                                              ray_candidates = ray_candidates)
        #toc = time_function()
        #log.debug('Queried %i rays at %f rays/second',
        #          len(rays),
        #          len(rays) / (toc-tic))
        return intersections
        
def ray_triangle_candidates(rays, triangles=None, tree=None):
    '''
    Do broad- phase search for triangles that the rays
    may intersect. 

    Does this by creating a bounding box for the ray as it 
    passes through the volume occupied by 
    '''
    if tree is None:
        tree = create_tree(triangles)

    ray_bounding   = ray_bounds(rays, tree.bounds)
    ray_candidates = [None] * len(rays)
    for ray_index, bounds in enumerate(ray_bounding):
        ray_candidates[ray_index] = list(tree.intersection(bounds))
    return ray_candidates

def create_tree(triangles):
    '''
    Given a set of triangles, create an r-tree for broad- phase 
    collision detection

    Arguments
    ---------
    triangles: (n, 3, 3) list of vertices

    Returns
    ---------
    tree: Rtree object 
    '''
  
    # the property object required to get a 3D r-tree index
    properties = rtree.Property()
    properties.dimension = 3
    # the (n,6) interleaved bounding box for every triangle
    tri_bounds = np.column_stack((triangles.min(axis=1), triangles.max(axis=1)))
  
    # stream loading wasn't getting proper index
    tree       = rtree.Index(properties=properties)  
    for i, bounds in enumerate(tri_bounds):
        tree.insert(i, bounds)
    
    return tree

def ray_bounds(rays, bounds, buffer_dist = 1e-5):
    '''
    Given a set of rays and a bounding box for the volume of interest
    where the rays will be passing through, find the bounding boxes 
    of the rays as they pass through the volume. 
    '''
    # separate out the (n, 2, 3) rays array into (n, 3) 
    # origin/direction arrays
    ray_origins    = rays[:,0,:]
    ray_directions = unitize(rays[:,1,:])

    # bounding box we are testing against
    bounds  = np.reshape(bounds, (2,3))

    # the projection of the vector (ray_origin -> bounding box corner) 
    #onto ray_directions
    project = np.column_stack((np.diag(np.dot(bounds[0] - ray_origins, ray_directions.T)),
                               np.diag(np.dot(bounds[1] - ray_origins, ray_directions.T))))

    project_min = project.min(axis=1).reshape((-1,1))
    project_max = project.max(axis=1).reshape((-1,1))

    
    project_min[project_min < buffer_dist] = -buffer_dist
    project_max[project_max < buffer_dist] =  buffer_dist
    

    # stack the rays with min/max projections
    ray_bounds = np.column_stack((ray_origins + ray_directions*project_min,
                                  ray_origins + ray_directions*project_max)).reshape((-1,2,3))
    # reformat bounds into (n,6) interleaved bounds
    ray_bounds = np.column_stack((ray_bounds.min(axis=1), ray_bounds.max(axis=1)))

    # pad the bounding box by TOL_BUFFER
    # not sure if this is necessary, but if the ray is  axis aligned
    # this function will otherwise return zero volume bounding boxes
    # which may or may not screw up the r-tree intersection queries
    ray_bounds += np.array([-1,-1,-1,1,1,1]) * buffer_dist
    
    return ray_bounds
