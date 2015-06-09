'''
Functions dealing with (n,d) points
'''
import numpy as np
from scipy.spatial import cKDTree as KDTree

from .constants import TOL_ZERO
from .geometry import plane_transform

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

def point_plane_distance(points, plane_normal, plane_origin=[0,0,0]):
    w         = np.array(points) - plane_origin
    distances = np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal)
    return distances


def major_axis(points):
    '''
    Returns an approximate vector representing the major axis of points
    '''
    u,s,v = np.linalg.svd(points)
    axis = v[0]
    return axis 

def surface_normal(points):
    '''
    Returns a normal estimate of a group of points using SVD

    Arguments
    ---------
    points: (n,d) set of points

    Returns
    ---------
    normal: (d) vector
    '''
    normal = np.linalg.svd(points)[2][-1]
    return normal

def plane_fit(points, tolerance=TOL_ZERO):
    '''                                   
    Given a set of points, find an origin and normal using least squares
    Arguments
    ---------
    points: (n,3)
    tolerance: how non-planar the result can be without raising an error
    
    Returns
    ---------
    C: (3) point on the plane
    N: (3) normal vector
    '''

    C = points[0]
    x = points - C
    M = np.dot(x.T, x)
    N = np.linalg.svd(M)[0][:,-1]

    if not tolerance is None:
        normal_bad  = np.ptp(np.dot(N, points.T)) > tolerance
        if normal_bad: 
            raise ValueError('Plane outside tolerance!')
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
    
    # create two axis perpendicular to each other and the normal, 
    # and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)

    #calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)

    #return the points sorted by angle
    return points[[np.argsort(angles)]]
               
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

def remove_close(points, radius):
    '''
    Given an (n, m) set of points where n=(2|3) return a list of points
    where no point is closer than radius
    '''
    tree     = KDTree(points)
    consumed = np.zeros(len(points), dtype=np.bool)
    unique   = np.zeros(len(points), dtype=np.bool)
    for i in xrange(len(points)):
        if consumed[i]: continue
        neighbors = tree.query_ball_point(points[i], r=radius)
        consumed[neighbors] = True
        unique[i]           = True
    return points[unique]

def remove_close_set(points_fixed, points_reduce, radius):
    '''
    Given two sets of points and a radius, return a set of points
    that is the subset of points_reduce where no point is within 
    radius of any point in points_fixed
    '''
    tree_fixed  = KDTree(points_fixed)
    tree_reduce = KDTree(points_reduce)
    reduce_duplicates = tree_fixed.query_ball_tree(tree_reduce, r = radius)
    reduce_duplicates = np.unique(np.hstack(reduce_duplicates).astype(int))
    reduce_mask = np.ones(len(points_reduce), dtype=np.bool)
    reduce_mask[reduce_duplicates] = False
    points_clean = points_reduce[reduce_mask]
    return points_clean

def plot_points(points, show=True):    
    import matplotlib.pyplot as plt

    dimension = np.shape(points)[2]    
    if dimension == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.array(points).T)
    elif dimension == 2:
        plt.scatter(*np.array(points).T)
    else:
        raise ValueError('Points must be 2D or 3D, not %dD', dimension)

    if show: plt.show()
    
