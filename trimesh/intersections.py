import numpy as np

from .constants import log, tol
from .grouping  import unique_value_in_row
from .points    import unitize

def mesh_plane(mesh, 
               plane_normal,
               plane_origin  = None):
    '''
    Find a the intersections between a mesh and a plane, 
    returning a set of line segments on that plane.

    Arguments
    ---------
    mesh:          Trimesh object
    plane_normal:  (3,) float, plane normal
    plane_origin:  (3,) float, plane origin

    Returns
    ----------
    (m, 2, 3) float, list of 3D line segments
    '''

    def _triangle_cases(signs):
        '''
        Figure out which faces correspond to which intersection case from 
        the signs of the dot product of each vertex.
        Does this by bitbang each row of signs into an 8 bit integer.

        code : signs      : intersects
        0    : [-1 -1 -1] : No
        2    : [-1 -1  0] : No
        4    : [-1 -1  1] : Yes; 2 on one side, 1 on the other
        6    : [-1  0  0] : Yes; one edge fully on plane
        8    : [-1  0  1] : Yes; one vertex on plane, 2 on different sides
        12   : [-1  1  1] : Yes; 2 on one side, 1 on the other
        14   : [0 0 0]    : No (on plane fully)
        16   : [0 0 1]    : Yes; one edge fully on plane
        20   : [0 1 1]    : No
        28   : [1 1 1]    : No
        '''

        signs_sorted = np.sort(signs, axis=1)
        coded = np.zeros(len(signs_sorted), dtype=np.int8) + 14
        for i in range(3):
            coded += signs_sorted[:,i] << 3-i

        # one edge fully on the plane
        key = np.zeros(29, dtype=np.bool)
        key[[6,16]] = True
        one_edge = key[coded]

        # one vertex on plane, other two on different sides
        key[:] = False
        key[8] = True
        one_vertex = key[coded]

        # one vertex on one side of the plane, two on the other
        key[:]      = False
        key[[4,12]] = True
        basic = key[coded]

        return basic, one_vertex, one_edge

    def _handle_on_vertex(signs, faces, vertices):
        # case where one vertex is on plane, two are on different sides
        vertex_plane = faces[signs == 0]
        edge_thru    = faces[signs != 0].reshape((-1,2))
        point_intersect, valid  = plane_lines(plane_origin, 
                                              plane_normal, 
                                              vertices[edge_thru.T],
                                              line_segments = False)
        lines = np.column_stack((vertices[vertex_plane[valid]],
                                 point_intersect)).reshape((-1,2,3))
        return lines

    def _handle_on_edge(signs, faces, vertices):
        # case where two vertices are on the plane and one is off
        edges  = faces[signs == 0].reshape((-1,2))
        points = vertices[edges]
        return points

    def _handle_basic(signs, faces, vertices):
        #case where one vertex is on one side and two are on the other
        unique_element = unique_value_in_row(signs, unique = [-1,1])
        edges = np.column_stack((faces[unique_element],
                                 faces[np.roll(unique_element, 1, axis=1)],
                                 faces[unique_element],
                                 faces[np.roll(unique_element, 2, axis=1)])).reshape((-1,2))
        intersections, valid  = plane_lines(plane_origin, 
                                            plane_normal, 
                                            vertices[edges.T],
                                            line_segments = False)
        # since the data has been pre- culled, any invalid intersections at all
        # means the culling was done incorrectly and thus things are mega-fucked
        assert valid.all()
        return intersections.reshape((-1,2,3))

    # dot product of each vertex with the plane normal, indexed by face
    # so for each face the dot product of each vertex is a row
    # shape is the same as mesh.faces (n,3)
    dots = np.dot(plane_normal, (mesh.vertices-plane_origin).T)[mesh.faces]

    # sign of the dot product is -1, 0, or 1
    # shape is the same as mesh.faces (n,3)
    signs = np.zeros(mesh.faces.shape, dtype=np.int8)
    signs[dots < -tol.merge] = -1
    signs[dots >  tol.merge] =  1

    cases = _triangle_cases(signs)
    handlers = (_handle_basic, 
                _handle_on_vertex, 
                _handle_on_edge)    

    lines = np.vstack([h(signs[c],
                         mesh.faces[c],
                         mesh.vertices) for c, h in zip(cases, handlers)])

    log.debug('mesh_cross_section found %i intersections', len(lines))

    return lines

def plane_lines(plane_origin, 
                plane_normal, 
                endpoints,                            
                line_segments = True):
    '''
    Calculate plane-line intersections

    Arguments
    ---------
    plane_origin:  plane origin, (3) list
    plane_normal:  plane direction (3) list
    endpoints:     (2, n, 3) points defining lines to be intersect tested
    line_segments: if True, only returns intersections as valid if
                   vertices from endpoints are on different sides
                   of the plane.

    Returns
    ---------
    intersections: (m, 3) list of cartesian intersection points
    valid        : (n, 3) list of booleans indicating whether a valid
                   intersection occurred
    '''
    endpoints    = np.asanyarray(endpoints)
    plane_origin = np.asanyarray(plane_origin).reshape(3)
    line_dir     = unitize(endpoints[1] - endpoints[0])
    plane_normal = unitize(np.asanyarray(plane_normal).reshape(3))
    
    t = np.dot(plane_normal, (plane_origin - endpoints[0]).T)
    b = np.dot(plane_normal, line_dir.T)
    
    # If the plane normal and line direction are perpendicular, it means
    # the vector is 'on plane', and there isn't a valid intersection.
    # We discard on-plane vectors by checking that the dot product is nonzero
    valid = np.abs(b) > tol.zero
    if line_segments:
        test = np.dot(plane_normal, np.transpose(plane_origin - endpoints[1]))
        different_sides = np.sign(t) != np.sign(test)
        nonzero = np.logical_or(np.abs(t) > tol.zero,
                                np.abs(test) > tol.zero)
        valid = np.logical_and(valid, different_sides)
        valid = np.logical_and(valid, nonzero)

    d = np.divide(t[valid], b[valid])
    intersection  = endpoints[0][valid]
    intersection += np.reshape(d, (-1,1)) * line_dir[valid]

    return intersection, valid
