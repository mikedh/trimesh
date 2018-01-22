"""
intersections.py
------------------

Primarily mesh-plane intersections (slicing).
"""
import numpy as np

from .constants import log, tol

from . import util
from . import grouping


def mesh_plane(mesh,
               plane_normal,
               plane_origin,
               return_faces=False):
    """
    Find a the intersections between a mesh and a plane,
    returning a set of line segments on that plane.

    Parameters
    ---------
    mesh:          Trimesh object
    plane_normal:  (3,) float, plane normal
    plane_origin:  (3,) float, plane origin
    return_faces:  bool, if True return face index line is from

    Returns
    ----------
    lines:      (m, 2, 3) float, list of 3D line segments
    face_index: (m,) int, index of mesh.faces for each line
    """

    def triangle_cases(signs):
        """
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

        Parameters
        ----------
        signs: (n,3) int, all values are -1,0, or 1
               Each row contains the dot product of all three vertices
               in a face with respect to the plane

        Returns
        ---------
        basic:      (n,) bool, which faces are in the basic intersection case
        one_vertex: (n,) bool, which faces are in the one vertex case
        one_edge:   (n,) bool, which faces are in the one edge case
        """

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
        key[[4, 12]] = True
        basic = key[coded]

        return basic, one_vertex, one_edge

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
        edges = np.column_stack(
            (faces[unique_element], faces[
                np.roll(
                    unique_element, 1, axis=1)], faces[unique_element], faces[
                np.roll(
                    unique_element, 2, axis=1)])).reshape(
                        (-1, 2))
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
                handle_on_vertex,
                handle_on_edge)

    lines = np.vstack([h(signs[c],
                         mesh.faces[c],
                         mesh.vertices) for c, h in zip(cases, handlers)])

    log.debug('mesh_cross_section found %i intersections', len(lines))
    if return_faces:
        face_index = np.hstack([np.nonzero(c)[0] for c in cases])
        return lines, face_index
    return lines


def plane_lines(plane_origin,
                plane_normal,
                endpoints,
                line_segments=True):
    """
    Calculate plane-line intersections

    Parameters
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
    """
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
    """
    Given one line per plane, find the intersection points

    Parameters
    -----------
    plane_origins:   (n,3) float, plane origins
    plane_normals:   (n,3) float, plane normals
    line_origins:    (n,3) float, line origins
    line_directions: (n,3) float, line directions

    Returns
    ----------
    on_plane: (n,3) float, points on specified planes
    valid:    (n,) bool, did plane intersect line or not
    """

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
