import numpy as np
from .geometry import faces_to_edges
from .grouping import group_rows

def transformation_2D(offset=[0.0,0.0], theta=0.0):
    '''
    2D homogeonous transformation matrix
    '''
    T = np.eye(3)
    s = np.sin(theta)
    c = np.cos(theta)

    T[0,0:2] = [ c, s]
    T[1,0:2] = [-s, c]
    T[0:2,2] = offset
    return T

def euclidean(a, b):
    return np.sum((np.array(a) - b)**2) ** .5

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))
    
def is_ccw(points):
    '''
    Given an (n,2) set of points, return True if they are counterclockwise
    '''
    xd = np.diff(points[:,0])
    yd = np.sum(np.column_stack((points[:,1], 
                                 points[:,1])).reshape(-1)[1:-1].reshape((-1,2)), axis=1)
    area = np.sum(xd*yd)*.5
    return area < 0

def three_dimensionalize(points, return_2D=True):
    points = np.array(points)
    shape = points.shape
 
    if len(shape) != 2:
        raise ValueError('Points must be 2D array!')

    if shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
        is_2D = True
    elif shape[1] == 3:
        is_2D = False
    else:
        raise ValueError('Points must be (n,2) or (n,3)!')

    if return_2D: 
        return is_2D, points
    return points

def grid_arange_2D(bounds, step):
    x_grid = np.arange(*bounds[:,0], step = step)
    y_grid = np.arange(*bounds[:,1], step = step)
    grid   = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1,2))
    return grid

def grid_linspace_2D(bounds, count):
    x_grid = np.linspace(*bounds[:,0], count = count)
    y_grid = np.linspace(*bounds[:,1], count = count)
    grid   = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1,2))
    return grid
