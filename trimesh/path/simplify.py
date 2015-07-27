import numpy as np

from collections import deque

from .arc       import fit_circle, angles_to_threepoint
from .entities  import Arc
from .constants import *

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
        if np.abs(aspect - 1.0) > TOL_ASPECT: continue

        # make sure the facets all meet the length tolerances specified
        facet_len = np.sum(np.diff(points, axis=0)**2, axis=1)
        facet_bad = facet_len > TOL_FACET**2
        if facet_bad.any(): continue

        # fit a circle using least squares
        C, R, E = fit_circle(points)
        # check to make sure the radius tolerance is met
        if E > TOL_RADIUS: continue

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
  
