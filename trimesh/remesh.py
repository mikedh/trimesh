"""
remesh.py
-------------

Deal with re- triangulation of existing meshes.
"""

from itertools import zip_longest

import numpy as np

from . import graph, grouping
from .constants import tol
from .geometry import faces_to_edges


def subdivide(
    vertices, faces, face_index=None, vertex_attributes=None, return_index=False
):
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
    f = np.column_stack(
        [
            faces_subset[:, 0],
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
            mid_idx[:, 2],
        ]
    ).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            if len(values) != len(vertices):
                continue
            attr_mid = values[edges[unique]].mean(axis=1)
            new_attributes[key] = np.vstack((values, attr_mid))
        return new_vertices, new_faces, new_attributes

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        stack = np.arange(start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = dict(zip(nonzero, stack))

        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces


def subdivide_to_size(vertices, faces, max_edge, max_iter=10, return_index=False):
    """
    Subdivide a mesh until every edge is shorter than a
    specified length.

    Every edge longer than `max_edge` is bisected at a single shared
    midpoint, so the two faces on either side stay in sync and a
    watertight input stays watertight — no T-junctions (cracks) are
    introduced. Faces already small enough are left untouched.

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
    index : (q,) int
      Only returned if `return_index`, index of
      original face for each new face.
    """
    # copy inputs and make sure dtype is correct
    vertices = np.array(vertices, dtype=np.float64, copy=True)
    faces = np.array(faces, dtype=np.int64, copy=True)
    # map each current face back to its original face index
    index = np.arange(len(faces))

    for i in range(max_iter + 1):
        # the unique edges of the current mesh and the length of each
        edges = np.sort(faces_to_edges(faces), axis=1)
        unique, inverse = grouping.unique_rows(edges)
        edges_unique = edges[unique]
        # length uses xyz only — callers may hstack extra columns (e.g. uv)
        lengths = (
            (vertices[edges_unique[:, 0], :3] - vertices[edges_unique[:, 1], :3]) ** 2
        ).sum(axis=1) ** 0.5
        long_edge = lengths > max_edge

        # every edge is short enough so we're done
        if not long_edge.any():
            break
        # check max_iter before refining again
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")

        # one shared midpoint vertex per long edge, -1 where left intact
        midpoint = np.full(len(unique), -1, dtype=np.int64)
        midpoint[long_edge] = np.arange(int(long_edge.sum())) + len(vertices)
        vertices = np.vstack((vertices, vertices[edges_unique[long_edge]].mean(axis=1)))

        # midpoint id for each face's 3 edges, columns (v0v1, v1v2, v2v0)
        face_mid = midpoint[inverse].reshape((-1, 3))
        split = face_mid >= 0
        # number of edges marked for bisection on each face: 0, 1, 2, or 3
        count = split.sum(axis=1)

        # 0 marked edges — face passes through unchanged
        keep = count == 0
        faces_keep = faces[keep]
        index_keep = index[keep]

        # 1 marked edge — rotate so the split edge is (a, b) so a single
        # template handles all three cases: fan the midpoint to the
        # opposite vertex as [a, p, c], [p, b, c]
        one = count == 1
        j = np.argmax(split[one], axis=1)
        r = np.arange(len(j))
        f1, mid1 = faces[one], face_mid[one]
        a, b, c = f1[r, j], f1[r, (j + 1) % 3], f1[r, (j + 2) % 3]
        p = mid1[r, j]
        faces_one = np.column_stack((a, p, c, p, b, c)).reshape((-1, 3))
        index_one = np.repeat(index[one], 2)

        # 2 marked edges — rotate so the unsplit edge is (c, a), again so
        # one template covers all three cases: a corner triangle [p, b, q]
        # plus the quad (a, p, q, c) cut along its shorter diagonal
        two = count == 2
        j = (np.argmin(split[two], axis=1) + 1) % 3
        r = np.arange(len(j))
        f2, mid2 = faces[two], face_mid[two]
        a, b, c = f2[r, j], f2[r, (j + 1) % 3], f2[r, (j + 2) % 3]
        p, q = mid2[r, j], mid2[r, (j + 1) % 3]
        # the shorter of the two quad diagonals: a-q versus p-c (xyz only)
        use_aq = (
            ((vertices[a, :3] - vertices[q, :3]) ** 2).sum(axis=1)
            <= ((vertices[p, :3] - vertices[c, :3]) ** 2).sum(axis=1)
        )[:, None]
        corner = np.column_stack((p, b, q))
        quad_a = np.where(use_aq, np.column_stack((a, p, q)), np.column_stack((a, p, c)))
        quad_b = np.where(use_aq, np.column_stack((a, q, c)), np.column_stack((p, q, c)))
        faces_two = np.vstack((corner, quad_a, quad_b))
        index_two = np.tile(index[two], 3)

        # 3 marked edges — the regular 1 -> 4 split, same winding as subdivide
        three = count == 3
        f3, mid3 = faces[three], face_mid[three]
        v0, v1, v2 = f3.T
        m0, m1, m2 = mid3.T
        faces_three = np.column_stack(
            (v0, m0, m2, m0, v1, m1, m2, m1, v2, m0, m1, m2)
        ).reshape((-1, 3))
        index_three = np.repeat(index[three], 4)

        faces = np.vstack((faces_keep, faces_one, faces_two, faces_three)).astype(
            np.int64
        )
        index = np.concatenate((index_keep, index_one, index_two, index_three))

    if return_index:
        assert len(index) == len(faces)
        return vertices, faces, index

    return vertices, faces


def subdivide_loop(vertices, faces, iterations=None):
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
    if iterations is None:
        iterations = 1

    def _subdivide(vertices, faces):
        # find the unique edges of our faces
        edges, edges_face = faces_to_edges(faces, return_index=True)
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)

        # set interior edges if there are two edges and boundary if there is
        # one.
        edge_inter = np.sort(grouping.group_rows(edges, require_count=2), axis=1)
        edge_bound = grouping.group_rows(edges, require_count=1)
        # make sure that one edge is shared by only one or two faces.
        if not len(edge_inter) * 2 + len(edge_bound) == len(edges):
            # we have multiple bodies it's a party!
            # edges shared by 2 faces are "connected"
            # so this connected components operation is
            # essentially identical to `face_adjacency`
            faces_group = graph.connected_components(edges_face[edge_inter])

            if len(faces_group) == 1:
                raise ValueError("Some edges are shared by more than 2 faces")

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
                    faces[f].reshape(-1), return_inverse=True
                )

                # subdivide this subset of faces
                cur_verts, cur_faces = _subdivide(
                    vertices=vertices[unique], faces=inverse.reshape((-1, 3))
                )

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
        odd[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

        # find vertex neighbors of each vertex
        neighbors = graph.neighbors(edges=edges[unique], max_index=len(vertices))
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
        beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
        even = (
            beta[:, None] * vertices_[neighbors].sum(1)
            + (1 - k[:, None] * beta[:, None]) * vertices
        )

        # calculate even vertices for the boundary case
        if edge_bound_mask.any():
            # boundary vertices from boundary edges
            vrt_bound_mask = np.zeros(len(vertices), dtype=bool)
            vrt_bound_mask[np.unique(edges[unique][~edge_inter_mask])] = True
            # one boundary vertex has two neighbor boundary vertices (set
            # others as -1)
            boundary_neighbors = neighbors[vrt_bound_mask]
            boundary_neighbors[~vrt_bound_mask[neighbors[vrt_bound_mask]]] = -1

            even[vrt_bound_mask] = (
                vertices_[boundary_neighbors].sum(axis=1) / 8.0
                + (3.0 / 4.0) * vertices[vrt_bound_mask]
            )

        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack(
            [
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
                odd_idx[:, 2],
            ]
        ).reshape((-1, 3))

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
