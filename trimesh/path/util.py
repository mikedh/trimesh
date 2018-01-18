import numpy as np


def is_ccw(points):
    '''
    Check if connected planar points are counterclockwise.

    Parameters
    -----------
    points: (n,2) float, connected points on a plane

    Returns
    ----------
    ccw: bool, True if points are counterclockwise
    '''
    points = np.asanyarray(points)

    if (len(points.shape) != 2 or
            points.shape[1] != 2):
        raise ValueError('CCW is only defined for 2D')
    xd = np.diff(points[:, 0])
    yd = np.column_stack((
        points[:, 1],
        points[:, 1])).reshape(-1)[1:-1].reshape((-1, 2)).sum(axis=1)
    area = np.sum(xd * yd) * .5
    ccw = area < 0

    return ccw
