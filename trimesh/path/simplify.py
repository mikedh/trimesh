import numpy as np

from collections import deque

from .arc       import fit_circle, angles_to_threepoint
from .entities  import Arc
from ..constants import log
from ..constants import tol_path as tol

def simplify(path):
    simplify_circles(path)
    
def simplify_circles(path):
    '''
    Turn closed paths represented with lines into 
    closed arc entities (also known as circles).
    '''
    # which entities are lines
    lines = np.array([i.__class__.__name__ == 'Line' for i in path.entities])
    # if all entities are lines, we don't need to check the individual path
    check = not lines.all()
    
    new_vertices = deque()
    new_entities = deque()
    old_entities = deque()    

    for path_index, entities in enumerate(path.paths):
        if check and not lines[entities].all(): continue
        points = np.array(path.polygons_closed[path_index].exterior.coords)
        
        # check aspect ratio as an early exit if the path is not a circle
        aspect = np.divide(*points.ptp(axis=0))
        if np.abs(aspect - 1.0) > tol.aspect_frac: continue

        # make sure the facets all meet the length tolerances specified
        facet_len = np.sum(np.diff(points, axis=0)**2, axis=1)
        facet_bad = facet_len > (tol.seg_frac * path.scale)
        if facet_bad.any(): continue

        # fit a circle using least squares
        C, R, E = fit_circle(points)
        # check to make sure the radius tolerance is met
        if (E/R) > tol.radius_frac: continue

        # we've passed all the tests/exits, so convert the group of lines
        # to a single circle entity
        new_entities.append(Arc(points = (np.arange(3) + 
                                          len(path.vertices) +
                                          len(new_vertices)),
                                closed = True))
        new_vertices.extend(angles_to_threepoint([0,np.pi], C, R))        
        old_entities.extend(entities)

    if len(new_vertices) > 0:
        path.vertices = np.vstack((path.vertices, new_vertices))
    if len(new_entities) > 0:
        path.entities = np.append(path.entities,  new_entities)
        path.remove_entities(old_entities)
  
def merge_colinear(points, scale=1.0):
    '''
    Given a set of points representing a path in space,
    merge points which are colinear.

    Arguments
    ----------
    points: (n, d) set of points (where d is dimension)
    scale:  float, scale of drawing
    Returns
    ----------
    merged: (j, d) set of points with colinear and duplicate 
             points merged, where (j < n)
    '''
    points         = np.array(points)

    # the vector from one point to the next
    direction      = np.diff(points, axis=0)
    # the length of the direction vector
    direction_norm = np.linalg.norm(direction, axis=1)
    # make sure points don't have zero length
    direction_ok   = direction_norm > tol.merge

    # remove duplicate points 
    points = np.vstack((points[0], points[1:][direction_ok]))
    direction = direction[direction_ok]
    direction_norm = direction_norm[direction_ok]
    
    # change nonzero direction vectors to unit vectors
    direction /= direction_norm.reshape((-1,1))
    # find the difference between subsequent direction vectors
    direction_diff = np.linalg.norm(np.diff(direction, axis=0), axis=1)

    # magnitude of direction difference between vectors times direction length
    colinear = (direction_diff * direction_norm[1:]) < (tol.merge * scale)
    colinear_index = np.nonzero(colinear)[0]

    mask = np.ones(len(points), dtype=np.bool)
    # since we took diff, we need to offset by one
    mask[colinear_index + 1] = False
    merged = points[mask]
    return merged
