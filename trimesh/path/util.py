import numpy as np


def is_ccw(points):
    '''
    Given an (n,2) set of points, return True if they are counterclockwise
    '''
    xd = np.diff(points[:, 0])
    yd = np.sum(np.column_stack(
        (points[:, 1], points[:, 1])).reshape(-1)[1:-1].reshape((-1, 2)), axis=1)
    area = np.sum(xd * yd) * .5
    ccw = area < 0

    return ccw
