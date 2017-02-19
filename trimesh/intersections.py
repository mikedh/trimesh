import numpy as np

from .constants import log, tol

from . import util
from . import grouping

def mesh_plane(mesh,
               plane_normal,
               plane_origin):
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
    (m, 3)    int, faces that were intersected
    (m, 2, 3) float, contribution of the vertices on each
              intersected triangle to the vertices of the
              intersection line.
    '''

    def triangle_cases(signs):
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

        Arguments
        ----------
        signs: (n,3) int, all values are -1,0, or 1
               Each row contains the dot product of all three vertices
               in a face with respect to the plane

        Returns
        ---------
        basic1:     (n,) bool, which faces are in the basic intersection case
                    (code 4)
        basic2:     (n,) bool, which faces are in the basic intersection case
                    (code 12)
        one_vertex: (n,) bool, which faces are in the one vertex case
        one_edge:   (n,) bool, which faces are in the one edge case
        '''

        signs_sorted = np.sort(signs, axis=1)
        coded = np.zeros(len(signs_sorted), dtype=np.int8) + 14
        for i in range(3):
            coded += signs_sorted[:, i] << 3 - i

        # one edge fully on the plane
        # note that we are only accepting *one* of the on- edge cases,
        # where the other vertex has a positive dot product (16) instead
        # of both on- edge cases ([6,16])
        # this is so that for regions that are co-planar with the the section plane
        # we don't end up with an invalid boundary
        key = np.zeros(29, dtype=np.bool)
        key[16] = True
        one_edge = key[coded]

        # one vertex on plane, other two on different sides
        key[:] = False
        key[8] = True
        one_vertex = key[coded]

        # one vertex on one side of the plane, two on the other
        key[:] = False
        key[4] = True
        basic1 = key[coded]
        key[:] = False
        key[12] = True
        basic2 = key[coded]

        return basic1, basic2, one_vertex, one_edge

    def handle_on_vertex(signs, faces, vertices):
        # case where one vertex is on plane, two are on different sides
        vertex_plane = faces[signs == 0]
        edge_thru = faces[signs != 0].reshape((-1, 2))
        point_intersect, valid = plane_lines(plane_origin,
                                             plane_normal,
                                             vertices[edge_thru.T],
                                             line_segments=False)
        lines = np.column_stack((vertices[vertex_plane[valid]],
                                 point_intersect)).reshape((-1, 2, 3))
        return lines

    def handle_on_edge(signs, faces, vertices):
        # case where two vertices are on the plane and one is off
        edges = faces[signs == 0].reshape((-1, 2))
        points = vertices[edges]
        return points

    def handle_basic(signs, faces, vertices):
        # case where one vertex is on one side and two are on the other
        unique_element = grouping.unique_value_in_row(signs, unique=[-1, 1])
        edges = np.column_stack((faces[unique_element],
                                 faces[np.roll(unique_element, 1, axis=1)],
                                 faces[unique_element],
                                 faces[np.roll(unique_element, 2, axis=1)])).reshape((-1, 2))
        intersections, valid = plane_lines(plane_origin,
                                           plane_normal,
                                           vertices[edges.T],
                                           line_segments=False)
        # since the data has been pre- culled, any invalid intersections at all
        # means the culling was done incorrectly and thus things are
        # mega-fucked
        assert valid.all()
        return intersections.reshape((-1, 2, 3))

    plane_normal = np.asanyarray(plane_normal)
    plane_origin = np.asanyarray(plane_origin)
    if plane_origin.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError('Plane origin and normal must be (3,)!')

    # dot product of each vertex with the plane normal, indexed by face
    # so for each face the dot product of each vertex is a row
    # shape is the same as mesh.faces (n,3)
    dots = np.dot(plane_normal, (mesh.vertices - plane_origin).T)[mesh.faces]

    # sign of the dot product is -1, 0, or 1
    # shape is the same as mesh.faces (n,3)
    signs = np.zeros(mesh.faces.shape, dtype=np.int8)
    signs[dots < -tol.merge] = -1
    signs[dots > tol.merge] = 1

    # figure out which triangles are in the cross section,
    # and which of the three intersection cases they are in
    cases = triangle_cases(signs)
    # handlers for each case
    handlers = (handle_basic,
                handle_basic,
                handle_on_vertex,
                handle_on_edge)

    signs   = [signs[c]      for c in cases]
    faces   = [mesh.faces[c] for c in cases]
    lines   = [h(s, f, mesh.vertices)
               for h, s, f
               in zip(handlers, signs, faces)]
    dists   = _vertex_distance(signs, faces, lines, mesh.vertices)

    lines   = np.vstack(lines)
    faces   = np.vstack(faces)
    dists   = np.vstack(dists)

    log.debug('mesh_cross_section found %i intersections', len(lines))

    return lines, faces, dists


def _vertex_distance(allSigns, allFaces, allLines, allVertices):
    '''Called by mesh_plane. Calculates the normalised distance of each
    triangle vertex to the intersection line vertices.

    Arguments
    ---------
    allSigns:    List of four (n, 3) arrays, one for each intersection
                 case (see the triangle_cases function inside mesh_plane).
                 Each array contains the signs of the dot product of
                 each triangle vertex with the intersection plane.
    
    allFaces:    List of four (n, 3) arrays, one for each intersection
                 case, with each containing the intersected triangles.
    
    allLines:    List of four (n, 2, 3) ararys, one for each intersection
                 case, containing the interesection line vertices for each
                 triangle.
    
    allVertices: (m, 3) All vertices in the surface

    Returns
    -------
    (n, 2, 3) For each line vertex, the inverse normalised distance of each
              of the intersected triangle vertices to that vertex.
    '''
    
    def distance(v1, v2):
        '''
        Returns the euclidean distance between the two
        sets of vertices.
        '''
        return np.sqrt(np.sum((v1 - v2) ** 2, axis=1))

    # Note: The functions below are very tightly coupled 
    #       to the inner functions of mesh_plane.

    
    def on_vertex_contrib(signs, lines, faces, vertices):
        '''
        Calculate the contribution for the on_vertex intersection case.
        '''
        
        contrib = np.zeros((faces.shape[0], 2, 3), dtype=np.float32)
        
        # The signs for each triangle in the on_vertex
        # case (code 8) are of the form [-1, 0, 1],
        # where the vertex on the plane has sign == 0,
        # and the other vertices are to either side
        # of the plane.
        #
        # The 'pivot' is the vertex which intersects
        # with the plane. 'p1' and 'p2' are the other
        # two vertices - the plane intersects the
        # triangle at the pivot vertex, and on the
        # edge between p1 and p2.
        pivotFaceVert = np.where(signs == 0)[1]
        pFaceVerts    = np.where(signs != 0)[1].reshape(-1, 2)
        p1FaceVert    = pFaceVerts[:, 0]
        p2FaceVert    = pFaceVerts[:, 1]
        faceIdxs      = np.arange(faces.shape[0])

        # For a triangle with code [-1, 0, 1] (i.e.
        # the second vertex intersects the plane),
        # contribution of the triangle vertices
        # to the intersection line vertices are as
        # follows...

        # [0, 1, 0] for the first line vertex
        # (== the pivot vertex)
        contrib[faceIdxs, 0, pivotFaceVert] = 1

        # [1-d, 0, d] for the second line vertex,
        # (the one intersecting the edge between 
        # the other two triangle vertices), where 
        # 'd' is the normalised distance between 
        # the first triangle vertex (p1), and the
        # line vertex.
        p1 = vertices[faces[faceIdxs, p1FaceVert]]
        p2 = vertices[faces[faceIdxs, p2FaceVert]]

        p1p2Dist  = distance(p1, p2)
        p1IntDist = distance(p1, lines[:, 1, :])
        p1Contrib = p1IntDist / p1p2Dist
        
        contrib[faceIdxs, 1, p1FaceVert] =     p1Contrib
        contrib[faceIdxs, 1, p2FaceVert] = 1 - p1Contrib

        return contrib


    def on_edge_contrib(signs, lines, faces, vertices):
        '''
        Calculate the contribution for the on_edge intersection case.
        '''
        
        # The signs for triangles in the on_edge case
        # (code 16) is of the form [0, 0, 1], where the
        # vertices on the plane have sign == 0.
        #
        # The two vertices on the plane are referred
        # to as p1 and p2.
        pFaceVerts = np.where(signs == 0)[1].reshape(-1, 2)
        p1FaceVert = pFaceVerts[:, 0]
        p2FaceVert = pFaceVerts[:, 1]

        # In this case, the two on-plane vertices
        # are identical to the intersection line
        # vertices. So, given an intersection
        # of [0, 0, 1], the contribution for each
        # line vertex would be:
        #
        #   - [1, 0, 0] for the first line vertex
        #     (== the first triangle vertex)
        #   - [0, 1, 0] for the second line vertex
        #     (== the second triangle vertex) 
        contrib   = np.zeros((faces.shape[0], 2, 3), dtype=np.float32)
        faceVerts = np.arange(faces.shape[0])

        contrib[faceVerts, 0, p1FaceVert] = 1
        contrib[faceVerts, 1, p2FaceVert] = 1

        return contrib
        
    
    def basic1_contrib(signs, lines, faces, vertices):
        '''Calculate the contribution for the first basic intersection
        case (intersection code 4).
        '''
        return basic_contrib(signs, lines, faces, vertices, 1)

    
    def basic2_contrib(signs, lines, faces, vertices):
        '''Calculate the contribution for the second basic intersection
        case (intersection code 12). 
        '''
        return basic_contrib(signs, lines, faces, vertices, -1) 

    
    def basic_contrib(signs, lines, faces, vertices, intType):
        '''
        Calculate the contribution for the basic intersection case. This
        function is shared by basic1_contrib and basic2_contrib,

        The intType argument gives the sign value of the vertex which
        is alone on one side of the intersection plane (1 or -1).
        '''
    
        # Signs for the basic intersection case
        # (one vertex on one side of the plane,
        # and the other two on the other side)
        # are of the form [-1, -1, 1] (code 4)
        # or [-1,  1, 1] (code 12). 
        #
        # The 'pivot' is the vertex which is alone
        # on one side of the intersection plane.
        # 'p1' and 'p2' are the remaining two
        # vertices. The plane intersects the
        # triangle on the (pivot, p1) edge and the
        # (pivot, p2) edge.
        #
        # The handle_basic function (inside mesh_plane)
        # uses np.roll to identify which triangle
        # vertices correspond to p1 and p2. Here
        # we do an equivalent thing using modulus.
        pivotFaceVert = np.where(signs == intType)[1]
        p1FaceVert    = (pivotFaceVert + 1) % 3
        p2FaceVert    = (pivotFaceVert + 2) % 3

        faceIdxs = np.arange(faces.shape[0])
        pivot    = vertices[faces[faceIdxs, pivotFaceVert]]
        p1       = vertices[faces[faceIdxs, p1FaceVert]]
        p2       = vertices[faces[faceIdxs, p2FaceVert]]

        # Distances from the pivot
        # vertex to p1 and p2
        pivotp1dist = distance(pivot, p1)
        pivotp2dist = distance(pivot, p2)

        # Distance from the pivot to the
        # intersection on the (pivot, p1)
        # and (pivot, p2) edges.
        pivotInt1dist = distance(pivot, lines[:, 0, :])
        pivotInt2dist = distance(pivot, lines[:, 1, :])

        # For the line vertex which lies on the
        # (pivot, p1) edge, the contribution of
        # the pivot vertex is the normalised
        # distance from the pivot to the
        # intersection. The ccvontribution of p1
        # is 1-minus the pivot contribution.
        #
        # The same logic is applied to the
        # (pivot, p2) intersection.
        pivot1Contrib = pivotInt1dist / pivotp1dist
        pivot2Contrib = pivotInt2dist / pivotp2dist

        # So, given the intersection case [-1, 1, 1],
        # the contribution for each line vertex would be:
        #
        #  - [1 - d01, d01, 0]
        #  - [1 - d02, 0,   d02]
        #
        # where d01 is the normalised distance from
        # the pivot vertex to first line vertex 
        # (the one on the [pivot, p1] edge), and
        # d02 is the normalised distance from the
        # pivot to the second line vertex (on the
        # [pivot, p2] edge)
        contrib = np.zeros((faces.shape[0], 2, 3), dtype=np.float32)
        contrib[faceIdxs, 0, pivotFaceVert] = 1 - pivot1Contrib
        contrib[faceIdxs, 0, p1FaceVert]    =     pivot1Contrib
        contrib[faceIdxs, 1, pivotFaceVert] = 1 - pivot2Contrib
        contrib[faceIdxs, 1, p2FaceVert]    =     pivot2Contrib

        return contrib

    handlers = [basic1_contrib,
                basic2_contrib,
                on_vertex_contrib,
                on_edge_contrib]

    contribs = [h(s, l, f, allVertices)
                for h, s, l, f in zip(handlers, allSigns, allLines, allFaces)]
    
    return contribs


def plane_lines(plane_origin,
                plane_normal,
                endpoints,
                line_segments=True):
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
    endpoints = np.asanyarray(endpoints)
    plane_origin = np.asanyarray(plane_origin).reshape(3)
    line_dir = util.unitize(endpoints[1] - endpoints[0])
    plane_normal = util.unitize(np.asanyarray(plane_normal).reshape(3))

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
    intersection = endpoints[0][valid]
    intersection += np.reshape(d, (-1, 1)) * line_dir[valid]

    return intersection, valid


def planes_lines(plane_origins,
                 plane_normals,
                 line_origins,
                 line_directions):
    '''
    Given one line per plane, find the intersection points

    Arguments
    -----------
    plane_origins:   (n,3) float, plane origins
    plane_normals:   (n,3) float, plane normals
    line_origins:    (n,3) float, line origins
    line_directions: (n,3) float, line directions

    Returns
    ----------
    on_plane: (n,3) float, points on specified planes
    valid:    (n,) bool, did plane intersect line or not
    '''

    plane_origins = np.asanyarray(plane_origins, dtype=np.float64)
    plane_normals = np.asanyarray(plane_normals, dtype=np.float64)
    line_origins = np.asanyarray(line_origins, dtype=np.float64)
    line_directions = np.asanyarray(line_directions, dtype=np.float64)

    origin_vectors = plane_origins - line_origins

    projection_ori = util.diagonal_dot(origin_vectors, plane_normals)
    projection_dir = util.diagonal_dot(line_directions, plane_normals)

    valid = np.abs(projection_dir) > tol.merge

    distance = np.divide(projection_ori[valid],
                         projection_dir[valid])

    on_plane = line_directions[valid] * distance.reshape((-1, 1))
    on_plane += line_origins[valid]

    return on_plane, valid
