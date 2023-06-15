"""
remesh.py
-------------

Deal with re- triangulation of existing meshes.
"""
import numpy as np

from . import util
from . import grouping
from . import graph
from . import geometry

from .geometry import faces_to_edges
from .constants import tol


def subdivide(vertices,
              faces,
              face_index=None,
              vertex_attributes=None,
              return_index=False):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            attr_mid = np.vstack([
                attr_tris[:, g, :].mean(axis=1)
                for g in [[0, 1],
                          [1, 2],
                          [2, 0]]])
            attr_mid = attr_mid[unique]
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        stack = np.arange(
            start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = {k: v for k, v in zip(nonzero, stack)}

        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces


def subdivide_to_size(vertices,
                      faces,
                      max_edge,
                      max_iter=10,
                      return_index=False):
    """
    Subdivide a mesh until every edge is shorter than a
    specified length.

    Will return a triangle soup, not a nicely structured mesh.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles
    max_edge : float
      Maximum length of any edge in the result
    max_iter : int
      The maximum number of times to run subdivision
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    index : (q, 3) int
      Only returned if `return_index`, index of
      original face for each new face.
    """
    # store completed
    done_face = []
    done_vert = []
    done_idx = []

    # copy inputs and make sure dtype is correct
    current_faces = np.array(
        faces, dtype=np.int64, copy=True)
    current_vertices = np.array(
        vertices, dtype=np.float64, copy=True)

    # store a map to the original face index
    current_index = np.arange(len(faces))

    # loop through iteration cap
    for i in range(max_iter + 1):
        # compute the length of every triangle edge
        edge_length = (np.diff(
            current_vertices[current_faces[:, [0, 1, 2, 0]], :3],
            axis=1) ** 2).sum(axis=2) ** 0.5
        # check edge length against maximum
        too_long = (edge_length > max_edge).any(axis=1)
        # faces that are OK
        face_ok = ~too_long

        # clean up the faces a little bit so we don't
        # store a ton of unused vertices
        unique, inverse = grouping.unique_bincount(
            current_faces[face_ok].flatten(),
            return_inverse=True)

        # store vertices and faces meeting criteria
        done_vert.append(current_vertices[unique])
        done_face.append(inverse.reshape((-1, 3)))

        if return_index:
            done_idx.append(current_index[face_ok])
            current_index = np.tile(current_index[too_long],
                                    (4, 1)).T.ravel()

        # met our goals so exit
        if not too_long.any():
            break

        # check max_iter before subdividing again
        if i >= max_iter:
            raise ValueError('max_iter exceeded!')

        # run subdivision again
        (current_vertices,
         current_faces) = subdivide(current_vertices,
                                    current_faces[too_long])

    # stack sequence into nice (n, 3) arrays
    final_vertices, final_faces = util.append_faces(
        done_vert, done_face)

    if return_index:
        final_index = np.concatenate(done_idx)
        assert len(final_index) == len(final_faces)
        return final_vertices, final_faces, final_index

    return final_vertices, final_faces


def subdivide_loop(vertices,
                   faces,
                   iterations=None):
    """
    Subdivide a mesh by dividing each triangle into four triangles
    and approximating their smoothed surface (loop subdivision).
    This function is an array-based implementation of loop subdivision,
    which avoids slow for loop and enables faster calculation.

    Overall process:
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    iterations : int
          Number of iterations to run subdivision
    """
    try:
        from itertools import zip_longest
    except BaseException:
        # python2
        from itertools import izip_longest as zip_longest

    if iterations is None:
        iterations = 1

    def _subdivide(vertices, faces):
        # find the unique edges of our faces
        edges, edges_face = faces_to_edges(
            faces, return_index=True)
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)

        # set interior edges if there are two edges and boundary if there is
        # one.
        edge_inter = np.sort(
            grouping.group_rows(
                edges,
                require_count=2),
            axis=1)
        edge_bound = grouping.group_rows(edges, require_count=1)
        # make sure that one edge is shared by only one or two faces.
        if not len(edge_inter) * 2 + len(edge_bound) == len(edges):
            # we have multiple bodies it's a party!
            # edges shared by 2 faces are "connected"
            # so this connected components operation is
            # essentially identical to `face_adjacency`
            faces_group = graph.connected_components(
                edges_face[edge_inter])

            if len(faces_group) == 1:
                raise ValueError('Some edges are shared by more than 2 faces')

            # collect a subdivided copy of each body
            seq_verts = []
            seq_faces = []
            # keep track of vertex count as we go so
            # we can do a single vstack at the end
            count = 0
            # loop through original face indexes
            for f in faces_group:
                # a lot of the complexity in this operation
                # is computing vertex neighbors so we only
                # want to pass forward the referenced vertices
                # for this particular group of connected faces
                unique, inverse = grouping.unique_bincount(
                    faces[f].reshape(-1), return_inverse=True)

                # subdivide this subset of faces
                cur_verts, cur_faces = _subdivide(
                    vertices=vertices[unique],
                    faces=inverse.reshape((-1, 3)))

                # increment the face references to match
                # the vertices when we stack them later
                cur_faces += count
                # increment the total vertex count
                count += len(cur_verts)
                # append to the sequence
                seq_verts.append(cur_verts)
                seq_faces.append(cur_faces)

            # return results as clean (n, 3) arrays
            return np.vstack(seq_verts), np.vstack(seq_faces)

        # set interior, boundary mask for unique edges
        edge_bound_mask = np.zeros(len(edges), dtype=bool)
        edge_bound_mask[edge_bound] = True
        edge_bound_mask = edge_bound_mask[unique]
        edge_inter_mask = ~edge_bound_mask

        # find the opposite face for each edge
        edge_pair = np.zeros(len(edges)).astype(int)
        edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
        edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
        opposite_face1 = edges_face[unique]
        opposite_face2 = edges_face[edge_pair[unique]]

        # set odd vertices to the middle of each edge (default as boundary
        # case).
        odd = vertices[edges[unique]].mean(axis=1)
        # modify the odd vertices for the interior case
        e = edges[unique[edge_inter_mask]]
        e_v0 = vertices[e][:, 0]
        e_v1 = vertices[e][:, 1]
        e_f0 = faces[opposite_face1[edge_inter_mask]]
        e_f1 = faces[opposite_face2[edge_inter_mask]]
        e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
        e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
        e_v2 = vertices[e_v2_idx]
        e_v3 = vertices[e_v3_idx]

        # simplified from:
        # # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
        odd[edge_inter_mask] = 0.375 * e_v0 + \
            0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

        # find vertex neighbors of each vertex
        neighbors = graph.neighbors(
            edges=edges[unique],
            max_index=len(vertices))
        # convert list type of array into a fixed-shaped numpy array (set -1 to
        # empties)
        neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
        # if the neighbor has -1 index, its point is (0, 0, 0), so that
        # it is not included in the summation of neighbors when calculating the
        # even
        vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
        # number of neighbors
        k = (neighbors + 1).astype(bool).sum(axis=1)

        # calculate even vertices for the interior case
        even = np.zeros_like(vertices)

        # beta = 1 / k * (5 / 8 - (3 / 8 + 1 / 4 * np.cos(2 * np.pi / k)) ** 2)
        # simplified with sympy.parse_expr('...').simplify()
        beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3)**2) / (64 * k)
        even = beta[:, None] * vertices_[neighbors].sum(1) \
            + (1 - k[:, None] * beta[:, None]) * vertices

        # calculate even vertices for the boundary case
        if edge_bound_mask.any():
            # boundary vertices from boundary edges
            vrt_bound_mask = np.zeros(len(vertices), dtype=bool)
            vrt_bound_mask[np.unique(edges[unique][~edge_inter_mask])] = True
            # one boundary vertex has two neighbor boundary vertices (set
            # others as -1)
            boundary_neighbors = neighbors[vrt_bound_mask]
            boundary_neighbors[~vrt_bound_mask[neighbors[vrt_bound_mask]]] = -1

            even[vrt_bound_mask] = (vertices_[boundary_neighbors].sum(axis=1) / 8.0 +
                                    (3.0 / 4.0) * vertices[vrt_bound_mask])

        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack([
            faces[:, 0],
            odd_idx[:, 0],
            odd_idx[:, 2],
            odd_idx[:, 0],
            faces[:, 1],
            odd_idx[:, 1],
            odd_idx[:, 2],
            odd_idx[:, 1],
            faces[:, 2],
            odd_idx[:, 0],
            odd_idx[:, 1],
            odd_idx[:, 2]]).reshape((-1, 3))

        # stack the new even vertices and odd vertices
        new_vertices = np.vstack((even, odd))

        return new_vertices, new_faces

    for _ in range(iterations):
        vertices, faces = _subdivide(vertices, faces)

    if tol.strict or True:
        assert np.isfinite(vertices).all()
        assert np.isfinite(faces).all()
        # should raise if faces are malformed
        assert np.isfinite(vertices[faces]).all()

        # none of the faces returned should be degenerate
        # i.e. every face should have 3 unique vertices
        assert (faces[:, 1:] != faces[:, :1]).all()

    return vertices, faces


def subdivide_plane(
    vertices,
    faces,
    plane_normal,
    plane_origin,
    face_index=None,
    cached_dots=None
):
    """
    Split all edges of a mesh (given as a set of faces and vertices) that intersect a plane,
    returning a new mesh (again as a set of faces and vertices) that has the same shape as the input mesh,
    but no edges intersecting the plane, only vertices.

    Parameters
    ---------
    vertices : (n, 3) float
        Vertices of source mesh to slice
    faces : (n, 3) int
        Faces of source mesh to slice
    plane_normal : (3,) float
        Normal vector of plane to intersect with mesh
    plane_origin :  (3,) float
        Point on plane to intersect with mesh
    face_index : ((m,) int)
        Indexes of faces to slice. When no mask is provided, the
        default is to slice all faces.
    cached_dots : (n, 3) float
        If an external function has stored dot
        products pass them here to avoid recomputing

    Returns
    ----------
    new_vertices : (n, 3) float
        Vertices of sliced mesh
    new_faces : (n, 3) int
        Faces of sliced mesh
    """

    # region Initialize Data
    if len(vertices) == 0:
        return vertices, faces

    plane_origin = np.asarray(plane_origin, dtype=np.float64)
    plane_normal = np.asarray(plane_normal, dtype=np.float64)

    # Construct a mask for the faces to slice.
    if face_index is not None:
        faces = faces[face_index]

    if cached_dots is not None:
        dots = cached_dots
    else:
        # dot product of each vertex with the plane normal indexed by face
        # so for each face the dot product of each vertex is a row
        # shape is the same as faces (n,3)
        dots = np.dot(vertices - plane_origin, plane_normal)

    # endregion

    # region Cluster faces
    # Find vertex orientations w.r.t. faces for all triangles:
    #  -1 -> vertex "above" plane (positive normal direction)
    #   0 -> vertex on plane
    #   1 -> vertex "below" plane (negative normal direction)
    signs = np.zeros(len(vertices), dtype=np.int8)
    signs[dots < -tol.merge] = 1
    signs[dots > tol.merge] = -1
    signs = signs[faces]

    # Find all triangles that intersect this plane BY EDGE.
    # Vertices ON the plane are ignored.
    # beside <- indices of all triangles intersecting the plane
    # onedge <- indices of all triangles with an intersecting edge
    signs_sum = signs.sum(axis=1, dtype=np.int8)
    signs_asum = np.abs(signs).sum(axis=1, dtype=np.int8)

    # Cases:
    # beside:   (1, 1, 1), (-1, -1, -1), (0, 0, 0), (0, 0, 1), (0, 0, -1)
    # onedge:   (-1, 1, 1), (1, -1, -1), (-1, 0, 1)

    beside = np.abs(signs_sum) == signs_asum
    onedge = np.logical_not(beside)
    # endregion

    triangles = vertices[faces]
    cut_triangles = triangles[onedge]

    # region Calculate intersections
    # Extract the intersections of each triangle's edges with the plane
    o = cut_triangles  # origins
    d = np.roll(o, -1, axis=1) - o  # directions
    num = (plane_origin - o).dot(plane_normal)  # compute num/denom
    denom = np.dot(d, plane_normal)
    denom[denom == 0.0] = 1e-12  # prevent division by zero
    dist = np.divide(num, denom)
    # intersection points for each segment
    int_points = np.einsum('ij,ijk->ijk', dist, d) + o
    # endregion

    # region Initialize result arrays
    # Copy non-cut faces
    new_faces = faces[beside]
    # Initialize the array of new vertices with the current vertices
    new_vertices = vertices
    # endregion

    # region Calculate masks for different type of cut faces
    # Positive Tri, Negative Quad
    cut_n_mask = onedge & (signs_sum == 1)
    # Positive Quad, Negative Tri
    cut_p_mask = onedge & (signs_sum == -1)
    # Mask faces with one vertex on the edge and one on each side
    cut_split_faces_mask = onedge & (signs_sum == 0) & (signs_asum == 2)

    cut_faces_p = faces[cut_p_mask]
    cut_signs_p = signs[cut_p_mask]
    cut_faces_n = faces[cut_n_mask]
    cut_signs_n = signs[cut_n_mask]
    # endregion

    # region Handle faces that split into a quad and a tri
    def split_face(cut_signs, int_points_f, cut_faces, sign):
        nonlocal new_vertices, new_faces
        num_faces = len(cut_signs)

        if num_faces > 0:
            # Extract the vertex on the outside of the plane, then get the
            # vertices (in CCW order of the inside vertices)
            int_inds = np.where(cut_signs == sign)[1]
            face_range = np.arange(num_faces)

            quad_int_verts = cut_faces[
                np.stack((face_range, face_range), axis=1),
                np.stack(((int_inds + 1) % 3, (int_inds + 2) % 3), axis=1)]

            tri_int_verts = cut_faces[
                face_range,
                int_inds
            ].reshape(num_faces, 1)

            # Fill out new quad faces with the intersection points as vertices
            vert_offset = len(new_vertices)

            # Extract correct intersection points from int_points and order
            # them in the same way as they were added to faces
            new_face_vertices = int_points_f[
                np.stack((face_range, face_range), axis=1),
                np.stack((((int_inds + 2) % 3).T, int_inds.T), axis=1),
                :
            ].reshape(2 * num_faces, 3)

            # triangulate quads, and add the
            # resulting triangles to the new faces

            new_quad_faces = np.append(
                quad_int_verts,
                np.arange(vert_offset,
                          vert_offset + 2 * num_faces).reshape(num_faces, 2),
                axis=1)

            new_tri_faces_from_quads = geometry.triangulate_quads(new_quad_faces)

            # Fill out new triangles with the intersection points as vertices
            new_tri_faces = np.append(
                tri_int_verts,
                np.arange(vert_offset,
                          vert_offset + 2 * num_faces).reshape(num_faces, 2),
                axis=1)[:, [0, 2, 1]]

            new_vertices = np.row_stack((new_vertices, new_face_vertices))
            new_faces = np.row_stack((
                new_faces,
                new_tri_faces_from_quads, new_tri_faces
            ))

    p_int_points = int_points[cut_p_mask[onedge], :, :]
    n_int_points = int_points[cut_n_mask[onedge], :, :]

    split_face(cut_signs_p, p_int_points, cut_faces_p, 1)
    split_face(cut_signs_n, n_int_points, cut_faces_n, -1)
    # endregion

    # region Handle split faces with one vertex on the plane and one on each side
    num_split_faces = cut_split_faces_mask.sum()

    if num_split_faces > 0:
        # index faces and signs of split faces
        cut_split_faces = faces[cut_split_faces_mask]
        cut_split_signs = signs[cut_split_faces_mask]

        # find intesection points
        cut_int_points = int_points[cut_split_faces_mask[onedge], :, :]

        # find indices of the vertex on the plane
        split_int_inds = np.where(cut_split_signs == 0)[1]

        split_range = np.arange(num_split_faces)

        # Find the indices of the vertices left and right from the plane
        split_pint_verts = cut_split_faces[
            np.stack((split_range, split_range), axis=1),
            np.stack((split_int_inds, (split_int_inds + 1) % 3), axis=1)]

        split_nint_verts = cut_split_faces[
            np.stack((split_range, split_range), axis=1),
            np.stack(((split_int_inds + 2) % 3, split_int_inds), axis=1)]

        vert_offset = len(new_vertices)

        # Indices of the new intersection vertices
        new_split_vert_int = np.arange(
            vert_offset,
            vert_offset +
            num_split_faces)[
            :,
            None]

        # Fill out new triangles with the intersection point as vertices
        new_split_pfaces = np.append(split_pint_verts, new_split_vert_int, axis=1)
        new_split_nfaces = np.append(split_nint_verts, new_split_vert_int, axis=1)

        # Extract correct intersection points from int_points
        split_vertex = (split_int_inds + 1) % 3
        new_split_vertices = cut_int_points[np.arange(num_split_faces), split_vertex]

        new_vertices = np.row_stack((new_vertices, new_split_vertices))
        new_faces = np.row_stack((new_faces, new_split_pfaces, new_split_nfaces))

    # endregion

    # region Cleanup
    # find the unique indices in the new faces
    # using an integer-only unique function

    # merge close vertices
    _, vert_idx, vert_lut = np.unique(np.fix(
        new_vertices / tol.merge) * tol.merge, axis=0, return_index=True, return_inverse=True)
    final_vert = new_vertices[vert_idx]
    final_face = vert_lut[new_faces]
    # endregion

    return final_vert, final_face
