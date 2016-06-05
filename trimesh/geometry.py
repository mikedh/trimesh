import numpy as np

from .transformations import rotation_matrix
from .constants       import tol, log
from .util            import unitize, stack_lines

try: 
    from scipy.sparse import coo_matrix
except ImportError: 
    log.warning('scipy.sparse.coo_matrix unavailable')

def plane_transform(origin, normal):
    '''
    Given the origin and normal of a plane, find the transform that will move 
    that plane to be coplanar with the XY plane

    Arguments
    ----------
    origin: (3,) float, point in space
    normal: (3,) float, plane normal vector

    Returns
    ---------
    transform: (4,4) float, transformation matrix
    '''
    transform        =  align_vectors(normal, [0,0,1])
    transform[0:3,3] = -np.dot(transform, np.append(origin, 1))[0:3]
    return transform
    
def transform_around(matrix, point):
    '''
    Given a transformation matrix, apply its rotation component around a 
    point in space. 

    Arguments
    ----------
    matrix: (4,4) float, transformation matrix
    point:  (3,)  float, point in space

    Returns
    ---------
    result: (4,4) transformation matrix
    '''
    point = np.array(point)
    translate = np.eye(4)
    translate[0:3,3] = -point
    result = np.dot(matrix, translate)
    translate[0:3,3] = point
    result = np.dot(translate, result)
    return result

def nondegenerate_faces(faces):
    '''
    Find all faces which have three unique vertex indices.

    Arguments
    ----------
    faces: (n, 3) int array of vertex indices

    Returns
    ----------
    nondegenerate: (n,) bool array of faces that have 3 unique indices
    '''
    diffed = np.diff(np.sort(faces, axis=1), axis=1)
    nondegenerate = np.all(diffed != 0, axis=1)
    return nondegenerate
    

def align_vectors(vector_start, vector_end, return_angle=False):
    '''
    Returns the 4x4 transformation matrix which will rotate from 
    vector_start (3,) to vector_end (3,), ex:
    
    vector_end == np.dot(T, np.append(vector_start, 1))[0:3]
    '''
    
    vector_start = unitize(vector_start)
    vector_end   = unitize(vector_end)
    cross        = np.cross(vector_start, vector_end)
    # we clip the norm to 1, as otherwise floating point bs
    # can cause the arcsin to error
    norm         = np.clip(np.linalg.norm(cross), -1.0, 1.0)
    direction    = np.sign(np.dot(vector_start, vector_end))
  
    if norm < tol.zero:
        # if the norm is zero, the vectors are the same
        # and no rotation is needed
        T       = np.eye(4)
        T[0:3] *= direction
    else:  
        angle = np.arcsin(norm) 
        if direction < 0:
            angle = np.pi - angle
        T = rotation_matrix(angle, cross)
    if return_angle:
        return T, angle
    return T
    
def faces_to_edges(faces, return_index=False):
    '''
    Given a list of faces (n,3), return a list of edges (n*3,2)
    '''
    faces = np.asanyarray(faces)
    edges = np.column_stack((faces[:,(0,1)],
                             faces[:,(1,2)],
                             faces[:,(2,0)])).reshape(-1,2)
    if return_index:
        face_index = np.tile(np.arange(len(faces)), (3,1)).T.reshape(-1)
        return edges, face_index
    return edges

def triangulate_quads(quads):
    '''
    Given a set of quad faces, return them as triangle faces.
    '''
    quads = np.array(quads)
    faces = np.vstack((quads[:,[0,1,2]],
                       quads[:,[2,3,0]]))
    return faces
    
def mean_vertex_normals(vertex_count, faces, face_normals, **kwargs):
    '''
    Find vertex normals from the mean of the faces that contain that vertex.

    Arguments
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
            sparse = vertices_faces_sparse(vertex_count, faces)
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
                                      exc_info = True)
        summed = summed_loop()
    unit_normals, valid = unitize(summed, check_valid=True)
    vertex_normals = np.zeros((vertex_count, 3), dtype=np.float64)
    vertex_normals[valid] = unit_normals

    return vertex_normals

def vertices_faces_sparse(vertex_count, faces):
    '''
    Return a sparse matrix for which vertices are contained in which faces.

    Returns
    ---------
    sparse: scipy.sparse.coo_matrix of shape (len(m.vertices), len(m.faces))
            dtype is boolean

    Example
     ----------
    In [1]: sparse = vertices_faces_sparse(len(mesh.vertices), mesh.faces)

    In [2]: sparse.shape
    Out[2]: (12, 20)

    In [3]: m.faces.shape
    Out[3]: (20, 3)

    In [4]: m.vertices.shape
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
    row  = faces.reshape(-1)
    col  = np.tile(np.arange(len(faces)).reshape((-1,1)), (1,3)).reshape(-1)

    shape  = (vertex_count, len(faces))
    data   = np.ones(len(col), dtype=np.bool)
    sparse = coo_matrix((data, (row,col)), 
                        shape = shape, 
                        dtype = np.bool)
    return sparse

def medial_axis(samples, contains):
    '''
    Given a set of samples on a boundary, find the approximate medial axis based
    on a voronoi diagram and a containment function which can assess whether
    a point is inside or outside of the closed geometry. 

    Arguments
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
    line_indices = np.vstack([stack_lines(i) for i in inside if len(i) >=2])
    lines = voronoi.vertices[line_indices]    
    return load_path(lines)

def rotation_2D_to_3D(matrix_2D):
    '''
    Given a 2D homogenous rotation matrix convert it to a 3D rotation
    matrix that is rotating around the Z axis

    Arguments
    ----------
    matrix_2D: (3,3) float, homogenous 2D rotation matrix
    
    Returns
    ----------
    matrix_3D: (4,4) float, homogenous 3D rotation matrix
    '''

    matrix_2D = np.asanyarray(matrix_2D)
    if matrix_2D.shape != (3,3):
        raise ValueError('Homogenous 2D transformation matrix required!')

    matrix_3D = np.eye(4)    
    # translation
    matrix_3D[0:2, 3]   = matrix_2D[0:2,2] 
    # rotation from 2D to around Z
    matrix_3D[0:2, 0:2] = matrix_2D[0:2,0:2]

    return matrix_3D
