import numpy as np
import time

from ..util            import Cache
from ..points          import unitize
from ..grouping        import unique_rows
from ..intersections   import plane_lines
from .ray_triangle_cpu import rays_triangles_id

class RayMeshIntersector:
    '''
    An object to query a mesh for ray intersections. 
    Precomputes an r-tree for each triangle on the mesh.
    '''
    def __init__(self, mesh):
        self.mesh   = mesh
        self._cache = Cache(self.mesh.md5)

    @property
    def tree(self):
        if 'tree' in self._cache:
            return self._cache.get('tree')
        else:
            return self._cache.set('tree',
                                   self.mesh.triangles_tree())

    def intersects_id(self, rays, return_any=False):
        '''
        Find the indexes of triangles the rays intersect

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions

        Returns
        ---------        
        hits: (n) sequence of triangle indexes which hit the ray
        '''
        rays       = np.array(rays, dtype=np.float)
        candidates = ray_triangle_candidates(rays = rays, 
                                             tree = self.tree)
        hits  = rays_triangles_id(triangles      = self.mesh.triangles, 
                                  rays           = rays, 
                                  ray_candidates = candidates,
                                  return_any     = return_any)
        return hits
            
    def intersects_location(self, rays, return_id=False):
        '''
        Return unique cartesian locations where rays hit the mesh.
        If you are counting the number of hits a ray had, this method 
        should be used as if only the triangle index is used on- edge hits
        will be counted twice. 

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions
        return_id: boolean flag, if True return triangle indexes
        Returns
        ---------
        locations: (n) sequence of (m,3) intersection points
        hits:      (n) list of face ids 
        '''
        rays      = np.array(rays, dtype=np.float)
        hits      = self.intersects_id(rays)
        locations = ray_triangle_locations(triangles     = self.mesh.triangles,
                                                  rays          = rays,
                                                  intersections = hits,
                                                  tri_normals   = self.mesh.face_normals)
        if return_id:
            return locations, hits
        return locations

    def intersects_any_triangle(self, rays):
        '''
        Find out whether the rays in question hit *any* triangle on the mesh.

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions

        Returns
        ---------
        hits_any: (n) boolean array of whether or not each ray hit any triangle
        '''
        hits     = self.intersects_id(rays)
        hits_any = np.array([len(i) > 0 for i in hits])
        return hits_any

    def intersects_any(self, rays):
        '''
        Find out whether *any* ray hit *any* triangle on the mesh.
        Equivilant to but signifigantly faster than (due to early exit):
            intersects_any_triangle(rays).any()

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions

        Returns
        ---------
        hit: boolean, whether any ray hit any triangle on the mesh
        '''
        hit = self.intersects_id(rays, return_any=True)
        return hit

def ray_triangle_candidates(rays, tree):
    '''
    Do broad- phase search for triangles that the rays
    may intersect. 

    Does this by creating a bounding box for the ray as it 
    passes through the volume occupied by the tree
    '''
    ray_bounding   = ray_bounds(rays, tree.bounds)
    ray_candidates = [None] * len(rays)
    for ray_index, bounds in enumerate(ray_bounding):
        ray_candidates[ray_index] = list(tree.intersection(bounds))
    return ray_candidates

def ray_bounds(rays, bounds, buffer_dist = 1e-5):
    '''
    Given a set of rays and a bounding box for the volume of interest
    where the rays will be passing through, find the bounding boxes 
    of the rays as they pass through the volume. 
    
    Arguments
    ---------
    rays: (n,2,3) array of ray origins and directions
    bounds: (2,3) bounding box (min, max)
    buffer_dist: float, distance to pad zero width bounding boxes

    Returns
    ---------
    ray_bounding: (n) set of AABB of rays passing through volume
    '''
    # separate out the (n, 2, 3) rays array into (n, 3) 
    # origin/direction arrays
    ray_ori = rays[:,0,:]
    ray_dir = unitize(rays[:,1,:])

    # bounding box we are testing against
    bounds  = np.array(bounds)

    # find the primary axis of the vector
    axis       = np.abs(ray_dir).argmax(axis=1)
    axis_bound = bounds.reshape((2,-1)).T[axis]
    axis_ori   = np.array([ray_ori[i][a] for i, a in enumerate(axis)]).reshape((-1,1))
    axis_dir   = np.array([ray_dir[i][a] for i, a in enumerate(axis)]).reshape((-1,1))

    # parametric equation of a line
    # point = direction*t + origin
    # p = dt + o
    # t = (p-o)/d
    t = (axis_bound - axis_ori) / axis_dir

    # prevent the bounding box from including triangles
    # behind the ray origin
    t[t < buffer_dist] = buffer_dist

    # the value of t for both the upper and lower bounds
    t_a = t[:,0].reshape((-1,1))
    t_b = t[:,1].reshape((-1,1))

    # the cartesion point for where the line hits the plane defined by
    # axis
    on_a = (ray_dir * t_a) + ray_ori
    on_b = (ray_dir * t_b) + ray_ori

    on_plane = np.column_stack((on_a, on_b)).reshape((-1,2,ray_dir.shape[1]))
    
    ray_bounding = np.hstack((on_plane.min(axis=1), on_plane.max(axis=1)))
    # pad the bounding box by TOL_BUFFER
    # not sure if this is necessary, but if the ray is  axis aligned
    # this function will otherwise return zero volume bounding boxes
    # which may or may not screw up the r-tree intersection queries
    ray_bounding += np.array([-1,-1,-1,1,1,1]) * buffer_dist

    return ray_bounding

def ray_triangle_locations(triangles, 
                           rays, 
                           intersections, 
                           tri_normals):
    '''
    Given a set of triangles, rays, and intersections between the two,
    find the cartesian locations of the intersections points. 

    Arguments
    ----------
    triangles:     (n, 3, 3) set of triangle vertices
    rays:          (m, 2, 3) set of ray origins/ray direction pairs
    intersections: (m) sequence of intersection indidices which triangles
                    each ray hits. 

    Returns
    ----------
    locations: (m) sequence of (p,3) cartesian points
    '''
    
    ray_origin   = rays[:,0,:]
    ray_vector   = rays[:,1,:]
    ray_segments = np.array([ray_origin,
                             ray_origin + ray_vector])
    locations = [[]] * len(rays)

    for r, tri_group in enumerate(intersections):
        group_locations = np.zeros((len(tri_group), 3))
        valid = np.zeros(len(tri_group), dtype=np.bool)
        for i, tri_index in enumerate(tri_group):
            origin  = triangles[tri_index][0]
            normal  = tri_normals[tri_index]
            segment = ray_segments[:,r,:].reshape((2,-1,3))
            point, ok = plane_lines(plane_origin  = origin,
                                   plane_normal  = normal,
                                   endpoints     = segment,
                                   line_segments = False)
            if ok:
                valid[i] = True
                group_locations[i] = point
        group_locations = group_locations[valid]
        unique = unique_rows(group_locations)[0]
        locations[r] = group_locations[unique]
    return np.array(locations)

def contains_points(mesh, points):
    '''
    Check if a mesh contains a set of points, using ray tests.

    If the point is on the surface of the mesh, behavior is undefined.

    Arguments
    ---------
    mesh: Trimesh object
    points: (n,3) points in space

    Returns
    ---------
    contains: (n) boolean array, whether point is inside mesh or not
    '''
    points = np.asanyarray(points)
    vector = unitize([0,0,1])
    rays = np.column_stack((points, 
                            np.tile(vector,(len(points),1)))).reshape((-1,2,3))
    hits = mesh.ray.intersects_location(rays)
    hits_count = np.array([len(i) for i in hits])
    contains = np.mod(hits_count, 2) == 1
  
    return contains
