"""
remesh.py
-------------

Deal with re- triangulation of existing meshes.
"""

import numpy as np

import collections

from . import util
from . import grouping


def subdivide(vertices, faces, face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Parameters
    ----------
    vertices: (n,3) float, vertices
    faces:    (n,3) int,   indexes of vertices which make up triangular faces
    face_index: faces to subdivide.
                if None: all faces of mesh will be subdivided
                if (n,) int array of indices: only specified faces will be
                   subdivided. Note that in this case the mesh will generally
                   no longer be manifold, as the additional vertex on the midpoint
                   will not be used by the adjacent faces to the faces specified,
                   and an additional postprocessing step will be required to
                   make resulting mesh watertight

    Returns
    ----------
    new_vertices: (n,3) float, vertices
    new_faces:    (n,3) int,   remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]
    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                               [1, 2],
                                                               [2, 0]]])
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    # for adjacent faces we are going to be generating the same midpoint
    # twice, so we handle it here by finding the unique vertices
    unique, inverse = grouping.unique_rows(mid)

    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)
    # the new faces, with correct winding
    f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                         mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                         mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                         mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    return new_vertices, new_faces


def subdivide_to_size(vertices, faces, max_edge, max_iter=10):
    """
    Subdivide a mesh until every edge is shorter than a specified length.

    Will return a triangle soup, not a nicely structured mesh.

    Parameters
    ------------
    vertices: (n,3) float, vertices in space
    faces:    (m,3) int,   indices of vertices which make up triangles
    max_edge: float,       maximum length of any edge in the result
    max_iter: int,         the maximum number of times to run subdivisions

    Returns
    ------------
    vertices: (j,3) float, vertices in space
    faces:    (q,3) int,   indices of vertices
    """
    done_face = collections.deque()
    done_vert = collections.deque()

    current_faces = faces
    current_vertices = vertices

    for i in range(max_iter + 1):
        triangles = current_vertices[current_faces]

        # compute the length of every triangle edge
        edge_lengths = (
            np.diff(triangles[:, [0, 1, 2, 0]], axis=1)**2).sum(axis=2) ** .5

        too_long = (edge_lengths > max_edge).any(axis=1)

        # clean up the faces a little bit so we don't carry a ton of unused
        # vertices
        unique, inverse = np.unique(current_faces[np.logical_not(too_long)],
                                    return_inverse=True)

        done_vert.append(current_vertices[unique])
        done_face.append(inverse.reshape((-1, 3)))

        if not too_long.any():
            break

        (current_vertices,
         current_faces) = subdivide(current_vertices,
                                    current_faces[too_long])

    vertices, faces = util.append_faces(done_vert, done_face)
    return vertices, faces
