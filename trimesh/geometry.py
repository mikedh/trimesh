import numpy as np

from .transformations import rotation_matrix
from .constants import tol, log

from . import util

try:
    from scipy.sparse import coo_matrix
except ImportError:
    log.warning('scipy.sparse.coo_matrix unavailable')


def plane_transform(origin, normal):
    '''
    Given the origin and normal of a plane, find the transform that will move
    that plane to be coplanar with the XY plane

    Parameters
    ----------
    origin: (3,) float, point in space
    normal: (3,) float, plane normal vector

    Returns
    ---------
    transform: (4,4) float, transformation matrix
    '''
    transform = align_vectors(normal, [0, 0, 1])
    transform[0:3, 3] = -np.dot(transform, np.append(origin, 1))[0:3]
    return transform


def transform_around(matrix, point):
    '''
    Given a transformation matrix, apply its rotation component around a
    point in space.

    Parameters
    ----------
    matrix: (4,4) float, transformation matrix
    point:  (3,)  float, point in space

    Returns
    ---------
    result: (4,4) transformation matrix
    '''
    point = np.array(point)
    translate = np.eye(4)
    translate[0:3, 3] = -point
    result = np.dot(matrix, translate)
    translate[0:3, 3] = point
    result = np.dot(translate, result)
    return result


def align_vectors(vector_start, vector_end, return_angle=False):
    '''
    Returns the 4x4 transformation matrix which will rotate from
    vector_start (3,) to vector_end (3,), ex:

    vector_end == np.dot(T, np.append(vector_start, 1))[0:3]
    '''
    vector_start = util.unitize(vector_start)
    vector_end = util.unitize(vector_end)
    cross = np.cross(vector_start, vector_end)
    # we clip the norm to 1, as otherwise floating point bs
    # can cause the arcsin to error
    norm = np.linalg.norm(cross)
    norm = np.clip(norm, -1.0, 1.0)
    direction = np.sign(np.dot(vector_start, vector_end))

    if norm < tol.zero:
        # if the norm is zero, the vectors are the same
        # and no rotation is needed
        T = np.eye(4)
        T[0:3] *= direction
    else:
        angle = np.arcsin(norm)
        if direction < 0:
            angle = np.pi - angle
        T = rotation_matrix(angle, cross)

    check = np.abs(np.dot(T[:3, :3], vector_start) - vector_end)
    if (check > 1e-5).any():
        raise ValueError('Vectors unaligned!')

    if return_angle:
        return T, angle
    return T


def faces_to_edges(faces, return_index=False):
    '''
    Given a list of faces (n,3), return a list of edges (n*3,2)
    '''
    faces = np.asanyarray(faces)
    edges = np.column_stack((faces[:, (0, 1)],
                             faces[:, (1, 2)],
                             faces[:, (2, 0)])).reshape(-1, 2)
    if return_index:
        face_index = np.tile(np.arange(len(faces)), (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges


def vector_angle(pairs):
    '''
    Find the angles between vector pairs

    Parameters
    ----------
    pairs: (n,2,3) set of vector pairs

    Returns
    ----------
    angles: (n,) float, angles between vectors

    Example
    ----------
    angles = mesh.face_normals[mesh.face_adjacency]
    '''
    pairs = np.asanyarray(pairs)
    if not util.is_shape(pairs, (-1, 2, 3)):
        raise ValueError('pairs must be (n,2,3)!')
    dots = util.diagonal_dot(pairs[:, 0], pairs[:, 1])
    # clip for floating point error
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.abs(np.arccos(dots))
    return angles


def triangulate_quads(quads):
    '''
    Given a set of quad faces, return them as triangle faces.
    '''
    quads = np.array(quads)
    faces = np.vstack((quads[:, [0, 1, 2]],
                       quads[:, [2, 3, 0]]))
    return faces


def mean_vertex_normals(vertex_count, faces, face_normals, **kwargs):
    '''
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
    '''
    def summed_sparse():
        # use a sparse matrix of which face contains each vertex to
        # figure out the summed normal at each vertex
        # allow cached sparse matrix to be passed
        if 'sparse' in kwargs:
            sparse = kwargs['sparse']
        else:
            sparse = index_sparse(vertex_count, faces)
        summed = sparse.dot(face_normals)
        log.debug('Generated vertex normals using sparse matrix')
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
    except:
        log.warning('Unable to generate sparse matrix! Falling back!',
                    exc_info=True)
        summed = summed_loop()
    unit_normals, valid = util.unitize(summed, check_valid=True)
    vertex_normals = np.zeros((vertex_count, 3), dtype=np.float64)
    vertex_normals[valid] = unit_normals

    return vertex_normals


def index_sparse(column_count, indices):
    '''
    Return a sparse matrix for which vertices are contained in which faces.

    Returns
    ---------
    sparse: scipy.sparse.coo_matrix of shape (column_count, len(faces))
            dtype is boolean

    Example
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
    '''
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


def medial_axis(samples, contains):
    '''
    Given a set of samples on a boundary, find the approximate medial axis based
    on a voronoi diagram and a containment function which can assess whether
    a point is inside or outside of the closed geometry.

    Parameters
    ----------
    samples:    (n,d) set of points on the boundary of the geometry
    contains:   function which takes (m,d) points and returns an (m) bool array

    Returns
    ----------
    lines:     (n,2,2) set of line segments
    '''

    from scipy.spatial import Voronoi
    from .path.io.load import load_path

    # create the voronoi diagram, after vertically stacking the points
    # deque from a sequnce into a clean (m,2) array
    voronoi = Voronoi(samples)
    # which voronoi vertices are contained inside the original polygon
    contained = contains(voronoi.vertices)
    # ridge vertices of -1 are outside, make sure they are False
    contained = np.append(contained, False)
    inside = [i for i in voronoi.ridge_vertices if contained[i].all()]
    line_indices = np.vstack([util.stack_lines(i)
                              for i in inside if len(i) >= 2])
    lines = voronoi.vertices[line_indices]
    return load_path(lines)
