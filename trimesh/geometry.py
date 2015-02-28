import numpy as np
from . import transformations as tr
from .constants import *
    
def point_plane_distance(points, plane_normal, plane_origin=[0,0,0]):
    w         = np.array(points) - plane_origin
    distances = np.abs(np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal))
    return distances

def unitize(points, check_valid=False):
    '''
    Flexibly turn a list of vectors into a list of unit vectors.
    
    Arguments
    ---------
    points:       (n,m) or (j) input array of vectors. 
                  For 1D arrays, points is treated as a single vector
                  For 2D arrays, each row is treated as a vector
    check_valid:  boolean, if True enables valid output and checking

    Returns
    ---------
    unit_vectors: (n,m) or (j) length array of unit vectors

    valid:        (n) boolean array, output only if check_valid.
                   True for all valid (nonzero length) vectors, thus m=sum(valid)
    '''
    points       = np.array(points)
    axis         = len(points.shape) - 1
    length       = np.sum(points ** 2, axis=axis) ** .5
    if check_valid:
        valid = np.greater(length, TOL_ZERO)
        if axis == 1: 
            unit_vectors = (points[valid].T / length[valid]).T
        elif len(points.shape) == 1 and valid: 
            unit_vectors = points / length
        else:         
            unit_vectors = []
        return unit_vectors, valid        
    else: 
        unit_vectors = (points.T / length).T
    return unit_vectors
    
def major_axis(points):
    '''
    Returns an approximate vector representing the major axis of points
    '''
    #sq = np.dot(points, np.transpose(points))
    #d, v = np.linalg.eig(sq)
    u,s,v = np.linalg.svd(points)
    return v[0]
        
def surface_normal(points):
    '''
    Returns a normal estimate using SVD http://www.lsr.ei.tum.de/fileadmin/publications/KlasingAlthoff-ComparisonOfSurfaceNormalEstimationMethodsForRangeSensingApplications_ICRA09.pdf

    points: (n,m) set of points

    '''
    return np.linalg.svd(points)[2][-1]

def plane_fit(points):
    '''
    Given a set of points, find an origin and normal using least squares

    Arguments
    ---------
    points: (n,3) 

    Returns
    ---------
    C: (3) point on the plane
    N: (3) normal vector
    '''
    C = points.mean(axis=0)
    x = points - C
    M = np.dot(x.T, x)
    N = np.linalg.svd(M)[0][:,-1]
    return C, N

def radial_sort(points, 
                origin = None, 
                normal = None):
    '''
    Sorts a set of points radially (by angle) around an origin/normal.
    If origin/normal aren't specified, it sorts around centroid
    and the approximate plane the points lie in. 

    points: (n,3) set of points
    '''
    #if origin and normal aren't specified, generate one at the centroid
    if origin==None: origin = np.average(points, axis=0)
    if normal==None: normal = surface_normal(points)
    
    #create two axis perpendicular to each other and the normal, and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)

    #calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)

    #return the points sorted by angle
    return points[[np.argsort(angles)]]
               

def plane_transform(origin, normal):
    '''
    Given the origin and normal of a plane, find the transform that will move 
    that plane to be coplanar with the XY plane                                                                                                                                 
    '''
    transform        =  align_vectors(normal, [0,0,1])
    transform[0:3,3] = -np.dot(transform, np.append(origin, 1))[0:3]
    return transform

def project_to_plane(points, 
                     plane_normal     = [0,0,1], 
                     plane_origin     = [0,0,0],
                     transform        = None,
                     return_transform = False,
                     return_planar    = True):
    '''
    Projects a set of (n,3) points onto a plane.

    Arguments
    ---------
    points:           (n,3) array of points
    plane_normal:     (3) normal vector of plane
    plane_origin:     (3) point on plane
    transform:        None or (4,4) matrix. If specified, normal/origin are ignored
    return_transform: bool, if true returns the (4,4) matrix used to project points 
                      onto a plane
    return_planar:    bool, if True, returns (n,2) points. If False, returns
                      (n,3), where the Z column consists of zeros
    '''

    if np.all(np.abs(plane_normal) < TOL_ZERO):
        raise NameError('Normal must be nonzero!')

    if transform is None:
        transform = plane_transform(plane_origin, plane_normal)
        
    transformed = transform_points(points, transform)[:,0:(3-int(return_planar))]

    if return_transform: 
        polygon_to_3D = np.linalg.inv(transform)
        return transformed, polygon_to_3D
    return transformed

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
        T = tr.rotation_matrix(angle, cross)
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

def absolute_orientation(points_A, points_B, return_error=False):
    '''
    Calculates the transform that best aligns points_A with points_B
    Uses Horn's method for the absolute orientation problem, in 3D with no scaling.

    Arguments
    ---------
    points_A:     (n,3) list of points
    points_B:     (n,3) list of points, T*points_A
    return_error: boolean, if True returns (n) list of euclidean distances
                  representing the distance from  T*points_A[i] to points_B[i]


    Returns
    ---------
    M:    (4,4) transformation matrix for the transform that best aligns points_A to 
           points_B
    error: (n) list of euclidean distances
    '''

    def transform_error():
        '''
        Returns the squared euclidean distance per point
        '''
        dim = np.shape(points_A)
        AR  = np.hstack((points_A, np.ones((dim[0],1)))).T
        return (np.sum(((np.dot(M,AR)[0:3,:]-np.array(points_B).T)**2), axis=0))

    dim = np.shape(points_A)
    if ((np.shape(points_B) != dim) or (dim[1] != 3)): return False
    lc = np.average(points_A, axis=0)
    rc = np.average(points_B, axis=0)
    left  = points_A - lc
    right = points_B - rc

    M = np.dot(left.T, right)

    [[Sxx, Sxy, Sxz], 
     [Syx, Syy, Syz], 
     [Szx, Szy, Szz]] = M

    N=[[(Sxx+Syy+Szz), (Syz-Szy), (Szx-Sxz), (Sxy-Syx)],
       [(Syz-Szy), (Sxx-Syy-Szz), (Sxy+Syx), (Szx+Sxz)],
       [(Szx-Sxz), (Sxy+Syx), (-Sxx+Syy-Szz), (Syz+Szy)],
       [(Sxy-Syx), (Szx+Sxz), (Syz+Szy),(-Sxx-Syy+Szz)]]

    (w,v) = np.linalg.eig(N)

    q = v[:,np.argmax(w)]
    q = q/np.linalg.norm(q)

    M1 = [[q[0], -q[1], -q[2], -q[3]], 
          [q[1],  q[0],  q[3], -q[2]], 
          [q[2], -q[3],  q[0],  q[1]],
          [q[3],  q[2], -q[1],  q[0]]]

    M2 = [[q[0], -q[1], -q[2], -q[3]], 
          [q[1],  q[0], -q[3],  q[2]], 
          [q[2],  q[3],  q[0], -q[1]],
          [q[3], -q[2],  q[1],  q[0]]]

    R = np.dot(np.transpose(M1),M2)[1:4,1:4]
    T = rc - np.dot(R, lc)

    M          = np.eye(4) 
    M[0:3,0:3] = R
    M[0:3,3]   = T

    if return_error: 
        return M, transform_error()
    return M
