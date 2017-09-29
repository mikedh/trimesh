import numpy as np


from .. import util
from .. import bounds
from .. import grouping
from .. import constants

def contains_points(intersector, points, check_direction=None):
    '''
    Check if a mesh contains a set of points, using ray tests.

    If the point is on the surface of the mesh, behavior is undefined.

    Parameters
    ---------
    mesh: Trimesh object
    points: (n,3) points in space

    Returns
    ---------
    contains: (n) boolean array, whether point is inside mesh or not
    '''

    
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1,3)):
        raise ValueError('points must be (n,3)')
    
    # default ray direction is a random number, but we are not generating it
    # to be unique each time so the behavior of the function is easier to debug
    default_direction = np.array([ 0.4395064455,  0.617598629942,  0.652231566745])
    
    if check_direction is None:
        ray_directions = np.tile(default_direction, (len(points), 1))
    else:
        ray_directions = np.tile(np.array(check_direction).reshape(3),
                                (len(points), 1))

    # cast a ray both forwards and backwards
    index_ray = intersector.intersects_location(np.vstack((points, 
                                                           points)),
                                                np.vstack((ray_directions, 
                                                          -ray_directions)))[1]
                                                          
    # placeholder result we'll fill in later    
    contains = np.zeros(len(points), dtype=np.bool)      
    # if we hit nothing in either direction, just return with no hits
    if len(index_ray) == 0:
        return contains                                                       
                                                          
    # reshape so bi_hits[0] is the result in the forward direction and
    #            bi_hits[1] is the result in the backwards directions
    bi_hits = np.bincount(index_ray, 
                          minlength=len(ray_directions)*2).reshape((2,-1))
    bi_contains = np.mod(bi_hits,  2) == 1

    # if the mod of the hit count is the same in both
    # directions, we can save that result and move on
    agree = np.equal(*bi_contains)
    contains[agree] = bi_contains[0][agree]
    
    # if one of the rays in either direction hit nothing
    # it is a very solid indicator we are in free space
    # as the edge cases we are working around tend to 
    # add hits rather than miss hits
    one_freespace = (bi_hits == 0).any(axis=0)
    
    # rays where they don't agree and one isn't in free space
    # are deemed to be broken
    broken = np.logical_and(np.logical_not(agree), 
                            np.logical_not(one_freespace))
                            
    # try to run again with a new random vector
    # only do it if check_direction isn't specified 
    # to avoid infinite recursion
    if broken.any() and check_direction is None:
        new_direction = util.unitize(np.random.random(3))
        contains[broken] = contains_points(intersector, 
                                           points[broken],
                                           check_direction = new_direction)           
        constants.log.debug('detected %d broken contains test, attempting to fix',
                            broken.sum())
        
    return contains
