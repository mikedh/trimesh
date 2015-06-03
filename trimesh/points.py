'''
Functions dealing with point clouds
'''

import numpy as np
from scipy.spatial import cKDTree as KDTree

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

def plot_points(points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.array(points).T)
    plt.show()
