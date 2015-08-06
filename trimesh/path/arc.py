import numpy as np

from ..util         import three_dimensionalize, euclidean
from ..points       import unitize
from .intersections import line_line
from .constants     import *

try: 
    from scipy.optimize import leastsq
except ImportError: 
    log.warning('No scipy.optimize for arc fitting!')

def arc_center(points):
    '''
    Given three points of an arc, find the center, radius, normal, and angle.

    This uses the fact that the intersection of the perpendicular
    bisectors of the segments between the control points is the center of the arc.

    Arguments
    ---------
    points: (3,d) list of points where (d in [2,3])
    
    Returns
    ---------
    center:       (d) point of the center of the arc
    radius:       float, radius of the arc
    plane_normal: (3) vector of the plane normal. 
    angle:        float, angle swept by the arc
    '''
    #it's a lot easier to treat 2D as 3D with a zero Z value
    is_2D, points = three_dimensionalize(points, return_2D = True)

    #find the two edge vectors of the triangle
    edge_direction = np.diff(points, axis=0)
    edge_midpoints = (edge_direction*.5) + points[0:2]

    #three points define a plane, so we find its normal vector
    plane_normal         = unitize(np.cross(*edge_direction[::-1]))
    vector_edge          = unitize(edge_direction)
    vector_perpendicular = unitize(np.cross(vector_edge, plane_normal))

    intersects, center   = line_line(edge_midpoints, vector_perpendicular)

    if not intersects:
        raise NameError('Segments do not intersect!')

    radius        = euclidean(points[0], center)
    vector_center = unitize(points[[0,2]] - center)
    angle         = np.arccos(np.clip(np.dot(*vector_center), -1.0, 1.0))
    large_arc     = np.dot(*edge_direction)
    if (abs(angle) > TOL_ZERO) and (large_arc < 0.0): 
        angle = (np.pi*2) - angle

    return center[:(3-is_2D)], radius, plane_normal, angle

def discretize_arc(points, close = False):
    '''
    Returns a version of a three point arc consisting of line segments

    Arguments
    ---------
    points: (n, d) points on the arc where d in [2,3]
    close:  boolean, if True close the arc (circle)

    Returns
    ---------
    discrete: (m, d)  
    points: either (3,3) or (3,2) of points for arc going from 
            points[0] to points[2], going through control point points[1]
    '''
    two_dimensional, points = three_dimensionalize(points, return_2D = True)
    center, R, N, angle     = arc_center(points)
    if close: angle = np.pi * 2
    
    #the number of facets, based on the angle critera
    facets_a = int(np.ceil(angle     / RES_ANGLE))
    #the number of facets, based on the facet length critera
    facets_d = int(np.ceil((R*angle) / RES_LENGTH))
    #we use the larger number so both RES_ANGLE and RES_LENGTH are satisfied
    count = np.max([facets_a, facets_d])
    
    V1 = unitize(points[0] - center)
    V2 = unitize(np.cross(-N, V1))
    t  = np.linspace(0, angle, count)

    discrete  = np.tile(center, (count, 1))
    discrete += R * np.cos(t).reshape((-1,1))*np.tile(V1, (count, 1)) 
    discrete += R * np.sin(t).reshape((-1,1))*np.tile(V2, (count, 1))

    if not close:
        arc_ok = np.linalg.norm(points[[0,-1]]-discrete[[0,-1]], axis=1) 
        assert (arc_ok < TOL_MERGE).all()

    discrete  = discrete[:,0:(3-two_dimensional)]

    return discrete

def arc_tangents(points):
    '''
    returns tangent vectors for points
    '''
    two_dimensional, points = three_dimensionalize(points, return_2D=True)
    center, R, N, angle     = arc_center(points)
    vectors  = points - center
    tangents = unitize(np.cross(vectors, N))
    return tangents[:,0:(3-two_dimensional)]

def arc_offset(points, distance):
    two_dimensional, points = three_dimensionalize(points)
    center, R, N, angle     = arc_center(points)
    vectors    = unitize(points - center)
    new_points = center + vectors*distance
    return new_points[:,0:(3-two_dimensional)]

def angles_to_threepoint(angles, center, radius):
    if angles[1] < angles[0]: angles[1] += np.pi*2
    angles = [angles[0], np.mean(angles), angles[1]]
    planar = np.column_stack((np.cos(angles), np.sin(angles)))*radius
    return planar + center

def fit_circle(points):
    def circle_residuals(points, center):
        Ri = np.sqrt(np.sum((points - center)**2, axis=1))
        return Ri - np.mean(Ri)

    def current_residuals(center):
        return circle_residuals(points, center)

    center_estimate    = np.mean(points, axis=0)
    center_result, ier = leastsq(current_residuals, center_estimate)
    if not (ier in [1,2,3,4]):
        raise NameError('Least square fit failed!')
    Ri = np.sqrt(np.sum((points - center_result)**2, axis=1))
    R  = Ri.mean()
    E  = np.max(np.abs(Ri - R))
    return center_result, R, E
