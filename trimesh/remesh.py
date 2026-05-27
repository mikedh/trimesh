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


def _subdivide_to_size_pass(vertices, faces, index, max_edge):
    """
    Run a single crack-free refinement pass.

    Every edge longer than `max_edge` is bisected at a *single* midpoint vertex
    that is shared by both faces adjacent to the edge, so neighboring faces stay
    in sync and no T-junctions (cracks) are introduced. Each face is then split
    with a template chosen by how many of its three edges are being bisected
    (0, 1, 2, or 3); when two edges are split the residual quad is divided along
    its shorter diagonal.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles
    index : (m,) int
      Original face index carried by each current face
    max_edge : float
      Maximum allowed edge length

    Returns
    ------------
    vertices : (n + k, 3) float
      Vertices with one shared midpoint appended per bisected edge
    faces : (p, 3) int
      Refined faces
    index : (p,) int
      Original face index for each refined face
    changed : bool
      False if no edge exceeded `max_edge` (nothing to do)
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)

    # the unique edges of the mesh and the length of each
    edges = np.sort(faces_to_edges(faces), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    unique_edges = edges[unique]
    lengths = ((vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]]) ** 2).sum(
        axis=1
    ) ** 0.5
    long_edge = lengths > max_edge

    # every edge is already short enough: done
    if not long_edge.any():
        return vertices, faces, index, False

    # assign one shared midpoint vertex to each long edge (-1 for edges left intact)
    midpoint_id = np.full(len(unique), -1, dtype=np.int64)
    midpoint_id[long_edge] = np.arange(int(long_edge.sum())) + len(vertices)
    midpoints = vertices[unique_edges[long_edge]].mean(axis=1)
    new_vertices = np.vstack((vertices, midpoints))

    # midpoint id for each of the 3 edges of every face, columns (v0v1, v1v2, v2v0)
    face_mid = midpoint_id[inverse].reshape((-1, 3))
    v0, v1, v2 = faces.T
    m0, m1, m2 = face_mid.T
    split = face_mid >= 0
    n_split = split.sum(axis=1)

    new_faces = []
    new_index = []

    def emit(mask, *triangles):
        # `triangles` is a flat sequence of masked column arrays, 3 columns per
        # output face; reshape interleaves a face's children, so index uses repeat
        rows = np.nonzero(mask)[0]
        if len(rows) == 0:
            return
        block = np.column_stack(triangles).reshape((-1, 3))
        new_faces.append(block)
        new_index.append(np.repeat(index[rows], len(triangles) // 3))

    def quad(mask, p, q, r, s):
        # split quad (p, q, r, s) (wound CCW) into two triangles along its shorter
        # diagonal; the two triangle blocks are stacked, so index uses tile
        rows = np.nonzero(mask)[0]
        if len(rows) == 0:
            return
        shorter_pr = (
            ((new_vertices[p] - new_vertices[r]) ** 2).sum(axis=1)
            <= ((new_vertices[q] - new_vertices[s]) ** 2).sum(axis=1)
        )[:, None]
        t1 = np.where(shorter_pr, np.column_stack((p, q, r)), np.column_stack((p, q, s)))
        t2 = np.where(shorter_pr, np.column_stack((p, r, s)), np.column_stack((q, r, s)))
        new_faces.append(np.vstack((t1, t2)))
        new_index.append(np.tile(index[rows], 2))

    # 0 marked edges: the face is already fine, keep it unchanged
    keep = n_split == 0
    emit(keep, v0[keep], v1[keep], v2[keep])

    # 3 marked edges: regular 1 -> 4 split
    m = n_split == 3
    emit(
        m,
        v0[m],
        m0[m],
        m2[m],
        m0[m],
        v1[m],
        m1[m],
        m2[m],
        m1[m],
        v2[m],
        m0[m],
        m1[m],
        m2[m],
    )

    # 1 marked edge: bisect from the edge midpoint to the opposite vertex
    m = (n_split == 1) & split[:, 0]
    emit(m, v0[m], m0[m], v2[m], m0[m], v1[m], v2[m])
    m = (n_split == 1) & split[:, 1]
    emit(m, v0[m], v1[m], m1[m], v0[m], m1[m], v2[m])
    m = (n_split == 1) & split[:, 2]
    emit(m, v0[m], v1[m], m2[m], m2[m], v1[m], v2[m])

    # 2 marked edges: a corner triangle plus a quad split along its shorter diagonal
    m = (n_split == 2) & split[:, 0] & split[:, 1]  # edges v0v1, v1v2
    emit(m, m0[m], v1[m], m1[m])
    quad(m, v0[m], m0[m], m1[m], v2[m])

    m = (n_split == 2) & split[:, 1] & split[:, 2]  # edges v1v2, v2v0
    emit(m, m2[m], m1[m], v2[m])
    quad(m, v0[m], v1[m], m1[m], m2[m])

    m = (n_split == 2) & split[:, 0] & split[:, 2]  # edges v0v1, v2v0
    emit(m, v0[m], m0[m], m2[m])
    quad(m, m0[m], v1[m], v2[m], m2[m])

    new_faces = np.vstack(new_faces).astype(np.int64)
    new_index = np.concatenate(new_index)
    return new_vertices, new_faces, new_index, True


def subdivide_to_size(vertices, faces, max_edge, max_iter=10, return_index=False):
    """
    Subdivide a mesh until every edge is shorter than a
    specified length.

    Unlike calling `subdivide` with a subset of faces, this splits edges shared
    between a refined and an unrefined face on *both* sides, so a watertight input
    stays watertight (no T-junctions / cracks are introduced). Only edges longer
    than `max_edge` are bisected, so faces that are already small are left intact.

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
    current_vertices = np.array(vertices, dtype=np.float64, copy=True)
    current_faces = np.array(faces, dtype=np.int64, copy=True)

    # map each current face back to its original face index
    current_index = np.arange(len(faces))

    # loop through iteration cap, bisecting all long edges conformally each pass
    for i in range(max_iter + 1):
        current_vertices, current_faces, current_index, changed = _subdivide_to_size_pass(
            current_vertices, current_faces, current_index, max_edge
        )
        # every edge met the target so we're done
        if not changed:
            break
        # check max_iter before refining again
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")

    if return_index:
        assert len(current_index) == len(current_faces)
        return current_vertices, current_faces, current_index

    return current_vertices, current_faces


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
