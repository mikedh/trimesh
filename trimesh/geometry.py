import numpy as np

from .transformations import rotation_matrix
from .constants import tol, log

from . import util

try:
    from scipy.sparse import coo_matrix
except ImportError:
    log.warning('scipy.sparse.coo_matrix unavailable')


def plane_transform(origin, normal):
    """
    Given the origin and normal of a plane find the transform
    that will move that plane to be coplanar with the XY plane.

    Parameters
    ----------
    origin : (3,) float
        Point that lies on the plane
    normal : (3,) float
        Vector that points along normal of plane

    Returns
    ---------
    transform: (4,4) float
        Transformation matrix to move points onto XY plane
    """
    transform = align_vectors(normal, [0, 0, 1])
    transform[0:3, 3] = -np.dot(transform,
                                np.append(origin, 1))[0:3]
    return transform


def align_vectors(vector_start, vector_end, return_angle=False):
    """
    Returns the 4x4 transformation matrix which will rotate from
    vector_start to vector_end, eg:

    vector_end == np.dot(T, np.append(vector_start, 1))[0:3]


    Parameters
    -----------
    vector_start: (3,) float, vector in space
    vector_end:   (3,) float, vector in space
    return_angle: bool, return angle between vectors or not

    Returns
    -----------
    transform: (4,4) float, transformation matrix
    angle:     float, angle in radians (only returned if flag set)

    """
    # convert start and end to (3, ) float unit vectors
    start = np.asanyarray(vector_start,
                          dtype=np.float64).reshape(3)
    start /= np.linalg.norm(start)
    end = np.asanyarray(vector_end,
                        dtype=np.float64).reshape(3)
    end /= np.linalg.norm(end)

    # get a unit vector perpendicular to both vectors
    # this will be the axis we are rotating around
    cross = np.cross(start, end)
    # we clip the norm to 1, as otherwise floating point bs
    # can cause the arcsin to error
    norm = np.linalg.norm(cross)
    norm = np.clip(norm, -1.0, 1.0)
    direction = np.sign(np.dot(start, end))

    if norm < tol.zero:
        # if the norm is zero, the vectors are the same
        # and no rotation is needed
        T = np.eye(4)
        T[0:3] *= direction
        angle = 0.0
    else:
        angle = np.arcsin(norm)
        if direction < 0:
            angle = np.pi - angle
        T = rotation_matrix(angle, cross)

    check = np.abs(np.dot(T[:3, :3], start) - end)
    if not (check < 1e-5).all():
        raise ValueError('aligning vectors failed!')

    if return_angle:
        return T, angle
    return T


def faces_to_edges(faces, return_index=False):
    """
    Given a list of faces (n,3), return a list of edges (n*3,2)

    Parameters
    -----------
    faces: (n,3) int, vertex indices representing faces

    Returns
    -----------
    edges: (n*3, 2) int, vertex indices representing edges
    """
    faces = np.asanyarray(faces)

    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    if return_index:
        # edges are in order of faces due to reshape
        face_index = np.tile(np.arange(len(faces)),
                             (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges


def vector_angle(pairs):
    """
    Find the angles between pairs of unit vectors.

    Parameters
    ----------
    pairs : (n,2,3) float
      Unit vector pairs

    Returns
    ----------
    angles : (n,) float
      Angles between vectors in radians
    """
    pairs = np.asanyarray(pairs, dtype=np.float64)
    if len(pairs) == 0:
        return np.array([])
    elif not util.is_shape(pairs, (-1, 2, (2, 3))):
        raise ValueError('pairs must be (n,2,(2|3))!')

    # do the dot product between vectors
    dots = util.diagonal_dot(pairs[:, 0], pairs[:, 1])
    # clip for floating point error
    dots = np.clip(dots, -1.0, 1.0)
    # do cos and remove arbitrary sign
    angles = np.abs(np.arccos(dots))

    return angles


def triangulate_quads(quads):
    """
    Given a set of quad faces, return them as triangle faces.

    Parameters
    -----------
    quads: (n,4) int, vertex indices of quad faces

    Returns
    -----------
    faces: (m,3) int, vertex indices of triangular faces
    """
    if len(quads) == 0:
        return quads
    quads = np.asanyarray(quads)
    faces = np.vstack((quads[:, [0, 1, 2]],
                       quads[:, [2, 3, 0]]))
    return faces


def mean_vertex_normals(vertex_count,
                        faces,
                        face_normals,
                        **kwargs):
    """
    Find vertex normals from the mean of the faces that contain that vertex.

    Parameters
    -----------
    vertex_count: int, the number of vertices faces refer to
    faces:        (n,3) int, list of vertex indices
    face_normals: (n,3) float, normal vector for each face

    Returns
    -----------
    vertex_normals: (vertex_count, 3) float normals for every vertex
                    Uncontained vertices will be zero.
    """
    def summed_sparse():
        # use a sparse matrix of which face contains each vertex to
        # figure out the summed normal at each vertex
        # allow cached sparse matrix to be passed
        if 'sparse' in kwargs:
            sparse = kwargs['sparse']
        else:
            sparse = index_sparse(vertex_count, faces)
        summed = sparse.dot(face_normals)
        return summed

    def summed_loop():
        # loop through every face, in tests was ~50x slower than
        # doing this with a sparse matrix
        summed = np.zeros((vertex_count, 3))
        for face, normal in zip(faces, face_normals):
            summed[face] += normal
        return summed

    try:
        summed = summed_sparse()
    except BaseException:
        log.warning(
            'unable to generate sparse matrix! Falling back!',
            exc_info=True)
        summed = summed_loop()

    # invalid normals will be returned as zero
    vertex_normals = util.unitize(summed)

    return vertex_normals


def index_sparse(column_count, indices):
    """
    Return a sparse matrix for which vertices are contained in which faces.

    Returns
    ---------
    sparse: scipy.sparse.coo_matrix of shape (column_count, len(faces))
            dtype is boolean

    Examples
     ----------
    In [1]: sparse = faces_sparse(len(mesh.vertices), mesh.faces)

    In [2]: sparse.shape
    Out[2]: (12, 20)

    In [3]: mesh.faces.shape
    Out[3]: (20, 3)

    In [4]: mesh.vertices.shape
    Out[4]: (12, 3)

    In [5]: dense = sparse.toarray().astype(int)

    In [6]: dense
    Out[6]:
    array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]])

    In [7]: dense.sum(axis=0)
    Out[7]: array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    """
    indices = np.asanyarray(indices)
    column_count = int(column_count)

    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape(
        (-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (column_count, len(indices))
    data = np.ones(len(col), dtype=np.bool)
    sparse = coo_matrix((data, (row, col)),
                        shape=shape,
                        dtype=np.bool)
    return sparse
