import numpy as np

from .constants import *
from .geometry import unitize, project_to_plane, faces_to_edges

def mesh_plane_intersection(mesh, 
                            plane_normal,
                            plane_origin  = None,
                            return_planar = False):
    '''
    Return a cross section of the trimesh based on plane origin and normal. 
    Basically a bunch of plane-line intersection queries

    origin:        (3) array of plane origin
    normal:        (3) array for plane normal
    return_planar: bool, True returns:
                         (m,2,2) list of 2D line segments
                         False returns:
                         (m,2,3) list of 3D line segments
    '''
    if len(mesh.faces) == 0: 
        raise NameError("Cannot compute cross section of empty mesh.")
    if plane_origin is None: 
        plane_origin = [0,0,0]

    edges                 = faces_to_edges(mesh.faces, sort=True)
    intersections, valid  = plane_line_intersection(plane_origin, 
                                                    plane_normal, 
                                                    mesh.vertices[[edges.T]],
                                                    line_segments = True)
    log.debug('mesh_cross_section found %i intersections', len(intersections))
    if return_planar:
        return project_to_plane(intersections.reshape((-1,3)),
                                plane_normal = plane_normal,
                                plane_origin = plane_origin).reshape((-1,2,2))
    return intersections.reshape(-1,2,3)

def points_in_mesh(points, mesh):
    from rtree import Rtree
    points = np.array(points)
    tri_3D = mesh.vertices[mesh.faces]
    tri_2D = tri_3D[:,:,0:2]
    z_bounds = np.column_stack((np.min(tri_3D[:,:,2], axis=1),
                                np.max(tri_3D[:,:,2], axis=1)))
    bounds = np.column_stack((np.min(tri_2D, axis=1), 
                              np.max(tri_2D, axis=1)))
    tree = Rtree()
    for i, bound in enumerate(bounds):
        tree.insert(i, bound)

    result = np.zeros(len(points), dtype=np.bool)
    for point_index, point in enumerate(points):
        intersections = np.array(list(tree.intersection(point[0:2].tolist())))
        in_triangle   = [point_in_triangle_2D(point[0:2], tri_2D[i]) for i in intersections]
        result[point_index] = np.int(np.mod(np.sum(in_triangle), 2)) == 0
    return result

def plane_line_intersection(plane_origin, 
                            plane_normal, 
                            endpoints,                            
                            line_segments = True):
    '''
    Calculates plane-line intersections

    Arguments
    ---------
    plane_origin:  plane origin, (3) list
    plane_normal:  plane direction (3) list
    endpoints:     points defining lines to be intersected, (2,n,3)
    line_segments: if True, only returns intersections as valid if
                   vertices from endpoints are on different sides
                   of the plane.

    Returns
    ---------
    intersections: (m, 3) list of cartesian intersection points
    valid        : (n, 3) list of booleans indicating whether a valid
                   intersection occurred
    '''
    endpoints = np.array(endpoints)
    line_dir  = unitize(endpoints[1] - endpoints[0])
    plane_normal = unitize(plane_normal)

    t = np.dot(plane_normal, np.transpose(plane_origin - endpoints[0]))
    b = np.dot(plane_normal, np.transpose(line_dir))
    
    # If the plane normal and line direction are perpendicular, it means
    # the vector is 'on plane', and there isn't a valid intersection.
    # We discard on-plane vectors by checking that the dot product is nonzero
    valid = np.abs(b) > TOL_ZERO
    if line_segments:
        test = np.dot(plane_normal, np.transpose(plane_origin - endpoints[1]))
        different_sides = np.sign(t) != np.sign(test)
        valid           = np.logical_and(valid, different_sides)
        
    d  = np.divide(t[valid], b[valid])
    intersection  = endpoints[0][valid]
    intersection += np.reshape(d, (-1,1)) * line_dir[valid]
    return intersection, valid

def point_in_triangle_2D(point, triangle):
    # http://www.blackpawn.com/texts/pointinpoly/

    v0, v1 = triangle[1:] - triangle[0]
    v2     = point        - triangle[0]
 
    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)
    
