'''
A basic, slow implementation of ray- triangle queries. 
'''
import numpy as np

from ..constants import tol
from ..grouping import unique_rows
from ..intersections import plane_lines

from .. import util


class RayMeshIntersector:
    '''
    An object to query a mesh for ray intersections.
    Precomputes an r-tree for each triangle on the mesh.
    '''

    def __init__(self, mesh):
        self.mesh = mesh
        self._cache = util.Cache(self.mesh.crc)

    @property
    def tree(self):
        if 'tree' in self._cache:
            return self._cache.get('tree')
        else:
            return self._cache.set('tree',
                                   self.mesh.triangles_tree())

    def intersects_id(self,
                      ray_origins,
                      ray_directions,
                      return_any=False):
        '''
        Find the indexes of triangles the rays intersect

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions

        Returns
        ---------
        hits: (n) sequence of triangle indexes which hit the ray
        '''
        hits = rays_triangles_id(triangles=self.mesh.triangles,
                                 ray_origins=ray_origins,
                                 ray_directions=ray_directions,
                                 tree=self.tree,
                                 return_any=return_any)
        return hits

    def intersects_location(self,
                            ray_origins,
                            ray_directions,
                            return_id=False):
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
        hits = self.intersects_id(ray_origins=ray_origins,
                                  ray_directions=ray_directions)
        locations = ray_triangle_locations(triangles=self.mesh.triangles,
                                           ray_origins=ray_origins,
                                           ray_directions=ray_directions,
                                           intersections=hits,
                                           tri_normals=self.mesh.face_normals)
        if return_id:
            return locations, hits
        return locations

    def intersects_any_triangle(self,
                                ray_origins,
                                ray_directions):
        '''
        Find out whether the rays in question hit *any* triangle on the mesh.

        Arguments
        ---------
        rays: (n, 2, 3) array of ray origins and directions

        Returns
        ---------
        hits_any: (n) boolean array of whether or not each ray hit any triangle
        '''
        hits = self.intersects_id(ray_origins, ray_directions)
        hits_any = np.array([len(i) > 0 for i in hits])
        return hits_any

    def intersects_any(self,
                       ray_origins,
                       ray_directions):
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
        hit = self.intersects_id(ray_origins,
                                 ray_directions,
                                 return_any=True)
        return hit


def rays_triangles_id(triangles,
                      ray_origins,
                      ray_directions,
                      ray_candidates=None,
                      tree=None,
                      return_any=False):
    '''
    Intersect a set of rays and triangles.

    Arguments
    ---------
    triangles:      (n, 3, 3) float array of triangle vertices
    ray_origins:    (m,3) float, array of ray origins
    ray_directions: (m,3) float, array of ray directions
    ray_candidates: (m, *) int array of which triangles are candidates
                    for the ray.
    return_any:     bool, exit loop early if any ray hits any triangle
                    and change output of function to bool

    Returns
    ---------
    if return_any:
        hit:           bool, whether the set of rays hit any triangle
    else:
        intersections: (m) sequence of triangle indexes hit by rays
    '''

    if ray_candidates is None:
        ray_candidates = ray_triangle_candidates(ray_origins=ray_origins,
                                                 ray_directions=ray_directions,
                                                 tree=tree)
    # default set of candidate triangles to be queried
    # is every triangle. this is very slow
    candidates = np.ones(len(triangles), dtype=np.bool)
    hits = [[]] * len(ray_origins)

    for ray_index, ray_ori, ray_dir in zip(range(len(ray_origins)),
                                           ray_origins,
                                           ray_directions):
        # query the triangle candidates
        candidates = ray_candidates[ray_index]
        if len(candidates) == 0: 
            continue
        hit = ray_triangles(triangles[candidates],
                            ray_ori,
                            ray_dir)
        if return_any:
            if hit.any():
                return True
        else:
            hits[ray_index] = np.array(candidates)[hit]

    if return_any:
        return False
    return np.array(hits)


def ray_triangles(triangles,
                  ray_origin,
                  ray_direction):
    '''
    Intersection of multiple triangles and a single ray.

    Moller-Trumbore intersection algorithm.
    '''
    candidates = np.ones(len(triangles), dtype=np.bool)

    # edge vectors and vertex locations in (n,3) format
    vert0 = triangles[:, 0, :]
    vert1 = triangles[:, 1, :]
    vert2 = triangles[:, 2, :]
    edge0 = vert1 - vert0
    edge1 = vert2 - vert0

    # P is a vector perpendicular to the ray direction and one
    # triangle edge.
    P = np.cross(ray_direction, edge1)

    # if determinant is near zero, ray lies in plane of triangle
    det = util.diagonal_dot(edge0, P)
    candidates[np.abs(det) < tol.zero] = False

    if not candidates.any():
        return candidates
    # remove previously calculated terms which are no longer candidates
    inv_det = 1.0 / det[candidates]
    T = ray_origin - vert0[candidates]
    u = util.diagonal_dot(T, P[candidates]) * inv_det

    new_candidates = np.logical_not(np.logical_or(u < -tol.zero,
                                                  u > (1 + tol.zero)))
    candidates[candidates] = new_candidates
    if not candidates.any():
        return candidates
    inv_det = inv_det[new_candidates]
    T = T[new_candidates]
    u = u[new_candidates]

    Q = np.cross(T, edge0[candidates])
    v = np.dot(ray_direction, Q.T) * inv_det

    new_candidates = np.logical_not(np.logical_or((v < -tol.zero),
                                                  (u + v > (1 + tol.zero))))

    candidates[candidates] = new_candidates
    if not candidates.any():
        return candidates

    Q = Q[new_candidates]
    inv_det = inv_det[new_candidates]

    t = util.diagonal_dot(edge1[candidates], Q) * inv_det
    candidates[candidates] = t > tol.zero

    return candidates


def ray_triangle_candidates(ray_origins,
                            ray_directions,
                            tree):
    '''
    Do broad- phase search for triangles that the rays
    may intersect.

    Does this by creating a bounding box for the ray as it
    passes through the volume occupied by the tree
    '''
    ray_bounding = ray_bounds(ray_origins=ray_origins,
                              ray_directions=ray_directions,
                              bounds=tree.bounds)
    ray_candidates = [None] * len(ray_origins)
    for ray_index, bounds in enumerate(ray_bounding):
        ray_candidates[ray_index] = np.array(list(tree.intersection(bounds)))
    return ray_candidates


def ray_bounds(ray_origins,
               ray_directions,
               bounds,
               buffer_dist=1e-5):
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

    # bounding box we are testing against
    bounds = np.array(bounds)

    # find the primary axis of the vector
    axis = np.abs(ray_directions).argmax(axis=1)
    axis_bound = bounds.reshape((2, -1)).T[axis]
    axis_ori = np.array([ray_origins[i][a]
                         for i, a in enumerate(axis)]).reshape((-1, 1))
    axis_dir = np.array([ray_directions[i][a]
                         for i, a in enumerate(axis)]).reshape((-1, 1))

    # parametric equation of a line
    # point = direction*t + origin
    # p = dt + o
    # t = (p-o)/d
    t = (axis_bound - axis_ori) / axis_dir

    # prevent the bounding box from including triangles
    # behind the ray origin
    t[t < buffer_dist] = buffer_dist

    # the value of t for both the upper and lower bounds
    t_a = t[:, 0].reshape((-1, 1))
    t_b = t[:, 1].reshape((-1, 1))

    # the cartesion point for where the line hits the plane defined by
    # axis
    on_a = (ray_directions * t_a) + ray_origins
    on_b = (ray_directions * t_b) + ray_origins

    on_plane = np.column_stack(
        (on_a, on_b)).reshape(
        (-1, 2, ray_directions.shape[1]))

    ray_bounding = np.hstack((on_plane.min(axis=1), on_plane.max(axis=1)))
    # pad the bounding box by TOL_BUFFER
    # not sure if this is necessary, but if the ray is  axis aligned
    # this function will otherwise return zero volume bounding boxes
    # which may or may not screw up the r-tree intersection queries
    ray_bounding += np.array([-1, -1, -1, 1, 1, 1]) * buffer_dist

    return ray_bounding


def ray_triangle_locations(triangles,
                           ray_origins,
                           ray_directions,
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

    ray_segments = np.array([ray_origins,
                             ray_origins + ray_directions])
    locations = [[]] * len(ray_origins)

    for r, tri_group in enumerate(intersections):
        group_locations = np.zeros((len(tri_group), 3))
        valid = np.zeros(len(tri_group), dtype=np.bool)
        for i, tri_index in enumerate(tri_group):
            origin = triangles[tri_index][0]
            normal = tri_normals[tri_index]
            segment = ray_segments[:, r, :].reshape((2, -1, 3))
            point, ok = plane_lines(plane_origin=origin,
                                    plane_normal=normal,
                                    endpoints=segment,
                                    line_segments=False)
            if ok:
                valid[i] = True
                group_locations[i] = point
        group_locations = group_locations[valid]
        unique = unique_rows(group_locations)[0]
        locations[r] = group_locations[unique]
    return np.array(locations)
