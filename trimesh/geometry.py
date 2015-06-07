import numpy as np
from .transformations import rotation_matrix
from .constants       import *
from .points          import project_to_plane, unitize

def plane_transform(origin, normal):
    '''
    Given the origin and normal of a plane, find the transform that will move 
    that plane to be coplanar with the XY plane                                                                                                                                 
    '''
    transform        =  align_vectors(normal, [0,0,1])
    transform[0:3,3] = -np.dot(transform, np.append(origin, 1))[0:3]
    return transform

def planar_hull(mesh, 
                normal           = [0,0,1], 
                origin           = [0,0,0], 
                return_transform = False):
    from scipy.spatial import ConvexHull
    planar , T = project_to_plane(mesh.vertices,
                                  plane_normal     = normal,
                                  plane_origin     = origin,
                                  return_transform = True)
    hull_edges = ConvexHull(planar).simplices
    if return_transform:
        return planar[hull_edges], T
    return planar[hull_edges]
    
def counterclockwise_angles(vector, vectors):
    dots    = np.dot(vector, np.array(vectors).T)
    dets    = np.cross(vector, vectors)
    angles  = np.arctan2(dets, dots)
    angles += (angles < 0.0)*np.pi*2
    return angles

def transform_points(points, matrix):
    '''
    Returns points, rotated by transformation matrix 
    If points is (n,2), matrix must be (3,3)
    if points is (n,3), matrix must be (4,4)
    '''
    dimension   = np.shape(points)[1]
    stacked     = np.column_stack((points, np.ones(len(points))))
    transformed = np.dot(matrix, stacked.T).T[:,0:dimension]
    return transformed

def align_vectors(vector_start, vector_end):
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
  
    if norm < TOL_ZERO:
        # if the norm is zero, the vectors are the same
        # and no rotation is needed
        T       = np.eye(4)
        T[0:3] *= direction
    else:  
        angle = np.arcsin(norm) 
        if direction < 0:
            angle = np.pi - angle
        T = rotation_matrix(angle, cross)
    return T
    
def faces_to_edges(faces, sort=True, return_index=False):
    '''                                                                                 
    Given a list of faces (n,3), return a list of edges (n*3,2)
    '''
    edges = np.column_stack((faces[:,(0,1)],
                             faces[:,(1,2)],
                             faces[:,(2,0)])).reshape(-1,2)
    if sort: edges.sort(axis=1)
    if return_index:
        face_index = np.tile(np.arange(len(faces)), (3,1)).T.reshape(-1)
        return edges, face_index
    return edges

def nondegenerate_faces(faces):
    '''
    Returns a 1D boolean array where non-degenerate faces are 'True'                        
    Faces should be (n, m) where for Trimeshes m=3. Returns (n) array                       
    '''
    nondegenerate = np.all(np.diff(np.sort(faces, axis=1), axis=1) != 0, axis=1)
    return nondegenerate
