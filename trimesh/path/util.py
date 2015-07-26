import numpy as np

def angles_to_threepoint(angles, center, radius, normal=[0,0,1]):
    if angles[1] < angles[0]: 
        angles[1] += np.pi*2
    angles = [angles[0], np.mean(angles), angles[1]]
    planar = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    points = planar + center
    return points
