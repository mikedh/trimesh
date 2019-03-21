"""
remesh.py
-------------

Deal with re- triangulation of existing meshes.
"""

import numpy as np

from . import util
from . import grouping


def subdivide(vertices,
              faces,
              face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those faces will
    be subdivided and their neighbors won't be modified making
    the mesh no longer "watertight."

    Parameters
    ----------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (n, 3) float
      Vertices in space
    new_faces : (n, 3) int
      Remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]
    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    # the 3 midpoints of each triangle edge
    # stacked to a (3 * c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1)
                     for g in [[0, 1],
                               [1, 2],
                               [2, 0]]])

    # for adjacent faces we are going to be generating
    # the same midpoint twice so merge them here
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = grouping.unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)

    # the new faces with correct winding
    f = np.column_stack([faces[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    return new_vertices, new_faces


def subdivide_to_size(vertices,
                      faces,
                      max_edge,
                      max_iter=10):
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

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    """
    # store completed
    done_face = []
    done_vert = []

    # copy inputs and make sure dtype is correct
    current_faces = np.array(faces,
                             dtype=np.int64,
                             copy=True)
    current_vertices = np.array(vertices,
                                dtype=np.float64,
                                copy=True)

    # loop through iteration cap
    for i in range(max_iter + 1):
        # (n, 3, 3) float triangle soup
        triangles = current_vertices[current_faces]

        # compute the length of every triangle edge
        edge_lengths = (np.diff(triangles[:, [0, 1, 2, 0]],
                                axis=1) ** 2).sum(axis=2) ** .5
        too_long = (edge_lengths > max_edge).any(axis=1)

        # clean up the faces a little bit so we don't
        # store a ton of unused vertices
        unique, inverse = np.unique(
            current_faces[np.logical_not(too_long)],
            return_inverse=True)

        # store vertices and faces meeting criteria
        done_vert.append(current_vertices[unique])
        done_face.append(inverse.reshape((-1, 3)))

        # met our goals so abort
        if not too_long.any():
            break

        # run subdivision again
        (current_vertices,
         current_faces) = subdivide(current_vertices,
                                    current_faces[too_long])

    # stack sequence into nice (n, 3) arrays
    vertices, faces = util.append_faces(done_vert,
                                        done_face)

    return vertices, faces
