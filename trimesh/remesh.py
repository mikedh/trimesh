import numpy as np

from .grouping import unique_rows


def subdivide(mesh, face_index=None):
    '''
    Subdivide a mesh into smaller triangles.

    Parameters
    ----------
    mesh: Trimesh object
    face_index: faces to subdivide.
                if None: all faces of mesh will be subdivided
                if (n,) int array of indices: only specified faces will be
                   subdivided. Note that in this case the mesh will generally
                   no longer be manifold, as the additional vertex on the midpoint
                   will not be used by the adjacent faces to the faces specified,
                   and an additional postprocessing step will be required to
                   make resulting mesh watertight


    '''
    if face_index is None:
        face_index = np.arange(len(mesh.faces))
    else:
        face_index = np.asanyarray(face_index, dtype=np.int64)

    # the (c,3) int set of vertex indices
    faces = mesh.faces[face_index]
    # the (c, 3, 3) float set of points in the triangles
    triangles = mesh.triangles[face_index]
    # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                               [1, 2],
                                                               [2, 0]]])
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    # for adjacent faces we are going to be generating the same midpoint
    # twice, so we handle it here by finding the unique vertices
    unique, inverse = unique_rows(mid)

    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(mesh.vertices)
    # the new faces, with correct winding
    f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                         mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                         mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                         mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((mesh.faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    mesh.vertices = np.vstack((mesh.vertices, mid))
    mesh.faces = new_faces
