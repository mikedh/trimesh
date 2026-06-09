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

    By default every face is split into four. If `face_index` is passed
    only those faces are split, and their neighbors are split along the
    shared edge too so a watertight mesh stays watertight (no cracks).

    Parameters
    ------------
    vertices : (n, d) float
      Vertices in space, xyz in the first three columns
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : (j,) int, (len(faces), 3) bool, or None
      If None all faces are subdivided. Integer indices / a face mask
      split those faces; a (len(faces), 3) bool array bisects only the
      marked edges (columns v0v1, v1v2, v2v0).
    vertex_attributes : dict or None
      Per-vertex (n, k) arrays, interpolated onto new midpoints
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (p, d) float
      Vertices, with one shared midpoint per split edge appended
    new_faces : (q, 3) int
      Subdivided faces
    new_attributes : dict
      Only if `vertex_attributes` was passed, interpolated attributes
    index : (q,) int
      Only if `return_index`, original face index of each new face
    """
    vertices = np.array(vertices, dtype=np.float64, copy=True)
    faces = np.asanyarray(faces, dtype=np.int64)
    count = len(vertices)

    if face_index is None:
        # fast path for the common `mesh.subdivide()` case: every face is a
        # clean 1 -> 4 split, so skip the crack-free bookkeeping below
        # (which only earns its keep when a subset of edges is bisected).
        edges = np.sort(faces_to_edges(faces), axis=1)
        unique, inverse = grouping.unique_rows(edges)
        mid = inverse.reshape((-1, 3)) + count
        new_vertices = np.vstack((vertices, vertices[edges[unique]].mean(axis=1)))
        new_faces = np.column_stack(
            (
                faces[:, 0], mid[:, 0], mid[:, 2],
                mid[:, 0], faces[:, 1], mid[:, 1],
                mid[:, 2], mid[:, 1], faces[:, 2],
                mid[:, 0], mid[:, 1], mid[:, 2],
            )
        ).reshape((-1, 3))
        if vertex_attributes is not None:
            new_attributes = {
                key: np.vstack((value, value[edges[unique]].mean(axis=1)))
                for key, value in vertex_attributes.items()
                if len(value) == count
            }
            return new_vertices, new_faces, new_attributes
        if return_index:
            return new_vertices, new_faces, np.repeat(np.arange(len(faces)), 4)
        return new_vertices, new_faces

    # subset: bisect only the marked edges, splitting their neighbors along
    # the shared edge so the mesh stays watertight. `face_index` is faces to
    # split (all three of their edges) or a (len(faces), 3) bool edge mask.
    face_index = np.asanyarray(face_index)
    if face_index.dtype == bool and face_index.shape == (len(faces), 3):
        # a per-face edge mask, e.g. the long edges from subdivide_to_size
        mark = face_index
    else:
        # face indices or a 1-D face mask: bisect all three of their edges
        mark = np.zeros((len(faces), 3), dtype=bool)
        mark[face_index] = True

    # unique edges, bisected if marked on either adjacent face so the two
    # faces sharing an edge stay in sync (no T-junctions / cracks)
    edges = np.sort(faces_to_edges(faces), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    bisect = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(bisect, inverse, mark.reshape(-1))

    # one shared midpoint vertex per bisected edge, -1 where left intact
    midpoint = np.full(len(unique), -1, dtype=np.int64)
    midpoint[bisect] = np.arange(int(bisect.sum())) + count
    ends = edges[unique][bisect]
    new_vertices = np.vstack((vertices, vertices[ends].mean(axis=1)))

    # the midpoint id (or -1) on each face's 3 edges, ordered
    # (corner0-corner1, corner1-corner2, corner2-corner0)
    edge_mid = midpoint[inverse].reshape((-1, 3))
    split = edge_mid >= 0
    n_split = split.sum(axis=1)
    source = np.arange(len(faces))

    # Split each face by how many of its edges were bisected (0/1/2/3). For
    # the 1- and 2-edge cases we first `roll` the face columns so the
    # relevant edge sits in a known position, letting one triangle template
    # cover all three rotations. In each block `corner[:, i]` are the rolled
    # face corners and `mid[:, i]` the midpoint on edge i; the new triangles
    # are interleaved per face, so each source face id repeats once per
    # triangle. Collected as (triangles, source) and stacked at the end.
    blocks = [(faces[n_split == 0], source[n_split == 0])]

    # 1 edge split: roll the split edge to the front, then fan its midpoint
    # to the opposite corner (corner2).
    one = n_split == 1
    roll = (np.arange(3) + np.argmax(split[one], axis=1)[:, None]) % 3
    corner = np.take_along_axis(faces[one], roll, axis=1)
    mid = np.take_along_axis(edge_mid[one], roll, axis=1)
    triangles = (
        np.column_stack((corner[:, 0], mid[:, 0], corner[:, 2])),
        np.column_stack((mid[:, 0], corner[:, 1], corner[:, 2])),
    )
    blocks.append(
        (np.stack(triangles, axis=1).reshape((-1, 3)), np.repeat(source[one], 2))
    )

    # 2 edges split: roll the unsplit edge to the back, leaving a corner
    # triangle at corner1 and a quad (corner0, mid0, mid1, corner2) that we
    # cut along its shorter diagonal — corner0-mid1 or mid0-corner2.
    two = n_split == 2
    roll = (np.arange(3) + np.argmin(split[two], axis=1)[:, None] + 1) % 3
    corner = np.take_along_axis(faces[two], roll, axis=1)
    mid = np.take_along_axis(edge_mid[two], roll, axis=1)
    xyz = new_vertices[:, :3]
    diagonal_corner0 = ((xyz[corner[:, 0]] - xyz[mid[:, 1]]) ** 2).sum(axis=1)
    diagonal_mid0 = ((xyz[mid[:, 0]] - xyz[corner[:, 2]]) ** 2).sum(axis=1)
    use_corner0 = (diagonal_corner0 <= diagonal_mid0)[:, None]
    triangles = (
        np.column_stack((mid[:, 0], corner[:, 1], mid[:, 1])),
        np.where(
            use_corner0,
            np.column_stack((corner[:, 0], mid[:, 0], mid[:, 1])),
            np.column_stack((corner[:, 0], mid[:, 0], corner[:, 2])),
        ),
        np.where(
            use_corner0,
            np.column_stack((corner[:, 0], mid[:, 1], corner[:, 2])),
            np.column_stack((mid[:, 0], mid[:, 1], corner[:, 2])),
        ),
    )
    blocks.append(
        (np.stack(triangles, axis=1).reshape((-1, 3)), np.repeat(source[two], 3))
    )

    # 3 edges split: the regular 1 -> 4 split, three corner triangles and a
    # central one made of the three midpoints.
    full = n_split == 3
    corner, mid = faces[full], edge_mid[full]
    triangles = (
        np.column_stack((corner[:, 0], mid[:, 0], mid[:, 2])),
        np.column_stack((mid[:, 0], corner[:, 1], mid[:, 1])),
        np.column_stack((mid[:, 2], mid[:, 1], corner[:, 2])),
        np.column_stack((mid[:, 0], mid[:, 1], mid[:, 2])),
    )
    blocks.append(
        (np.stack(triangles, axis=1).reshape((-1, 3)), np.repeat(source[full], 4))
    )

    new_faces = np.vstack([tri for tri, _ in blocks]).astype(np.int64)
    source = np.concatenate([src for _, src in blocks])

    if vertex_attributes is not None:
        new_attributes = {
            key: np.vstack((value, value[ends].mean(axis=1)))
            for key, value in vertex_attributes.items()
            if len(value) == count
        }
        return new_vertices, new_faces, new_attributes
    if return_index:
        return new_vertices, new_faces, source
    return new_vertices, new_faces


def subdivide_to_size(vertices, faces, max_edge, max_iter=10, return_index=False):
    """
    Subdivide a mesh until every edge is shorter than a specified length.

    Each pass bisects every edge longer than `max_edge` with `subdivide`,
    which keeps the mesh watertight (no cracks). Only the long edges are
    split, so long thin faces aren't needlessly shattered.

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
      Only returned if `return_index`, original face for each new face.
    """
    vertices = np.array(vertices, dtype=np.float64, copy=True)
    faces = np.array(faces, dtype=np.int64, copy=True)
    index = np.arange(len(faces))

    for i in range(max_iter + 1):
        # which of each face's edges (v0v1, v1v2, v2v0) exceed max_edge
        # (xyz only); bisect just those, not the whole face
        edge = np.diff(vertices[faces][:, [0, 1, 2, 0], :3], axis=1)
        mark = (edge**2).sum(axis=2) ** 0.5 > max_edge
        if not mark.any():
            break
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")
        # bisect the long edges (crack-free) and carry each original index
        vertices, faces, source = subdivide(
            vertices, faces, face_index=mark, return_index=True
        )
        index = index[source]

    if return_index:
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
