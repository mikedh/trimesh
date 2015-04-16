import numpy as np
from ..geometry import faces_to_edges
from ..grouping import group_rows


def transformation_2D(offset=[0.0,0.0], theta=0.0):
    '''
    2D homogeonous transformation matrix
    '''
    t = np.eye(3)
    s = np.sin(theta); c = np.cos(theta)
    t[0,0:2] = [ c, s]
    t[1,0:2] = [-s, c]
    t[0:2,2] = offset
    return t

def three_dimensionalize(points):
    if np.shape(points)[1] == 2:
        return True, np.column_stack((points, np.zeros(len(points))))
    else: return False, points

def euclidean(a, b):
    return np.sum((np.array(a) - b)**2) ** .5

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))
