import numpy as np

from .constants import log, tol
from .geometry  import faces_to_edges
from .points    import  unitize, project_to_plane 

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

    edges = np.sort(faces_to_edges(mesh.faces), axis=1)
    intersections, valid  = plane_line_intersection(plane_origin, 
                                                    plane_normal, 
                                                    mesh.vertices[edges.T],
                                                    line_segments = True)
    log.debug('mesh_cross_section found %i intersections', len(intersections))
    if return_planar:
        return project_to_plane(intersections.reshape((-1,3)),
                                plane_normal = plane_normal,
                                plane_origin = plane_origin).reshape((-1,2,2))
    return intersections.reshape(-1,2,3)

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
    endpoints    = np.array(endpoints)
    line_dir     = unitize(endpoints[1] - endpoints[0])
    plane_normal = unitize(np.asanyarray(plane_normal).reshape(3))
    plane_origin = np.asanyarray(plane_origin).reshape(3)

    t = np.dot(plane_normal, np.transpose(plane_origin - endpoints[0]))
    b = np.dot(plane_normal, np.transpose(line_dir))
    
    # If the plane normal and line direction are perpendicular, it means
    # the vector is 'on plane', and there isn't a valid intersection.
    # We discard on-plane vectors by checking that the dot product is nonzero
    valid = np.abs(b) > tol.merge
    if line_segments:
        test = np.dot(plane_normal, np.transpose(plane_origin - endpoints[1]))
        different_sides = np.sign(t) != np.sign(test)
        nonzero = np.logical_or(np.abs(t) > tol.merge,
                                np.abs(test) > tol.merge)
        valid = np.logical_and(valid, different_sides)
        valid = np.logical_and(valid, nonzero)

    d  = np.divide(t[valid], b[valid])
    intersection  = endpoints[0][valid]
    intersection += np.reshape(d, (-1,1)) * line_dir[valid]

    return intersection, valid
