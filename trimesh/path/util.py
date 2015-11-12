import numpy as np

def angles_to_threepoint(angles, center, radius, normal=[0,0,1]):
    if angles[1] < angles[0]: 
        angles[1] += np.pi*2
    angles = [angles[0], np.mean(angles), angles[1]]
    planar = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    points = planar + center
    return points

def is_ccw(points):
    ''' 
    Given an (n,2) set of points, return True if they are counterclockwise
    '''
    xd = np.diff(points[:,0])
    yd = np.sum(np.column_stack((points[:,1],
                                 points[:,1])).reshape(-1)[1:-1].reshape((-1,2)), axis=1)
    area = np.sum(xd*yd)*.5
    ccw =  area < 0

    return ccw
