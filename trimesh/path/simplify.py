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
    lines = np.array([i.__class__.__name__ == 'Line' for i in path.entities])
    check = not lines.all()
    
    old_entities = deque()    
    new_vertices = deque()
    new_entities = deque()

    for path_index, entities in enumerate(path.paths):
        if check and not lines[entities].all(): continue
        points = np.array(path.polygons[path_index].exterior.coords)
        aspect = np.divide(*points.ptp(axis=0))
        if np.abs(aspect - 1.0) > TOL_ASPECT: continue
        facet_len = np.sum(np.diff(points, axis=0)**2, axis=1)
        facet_bad = facet_len > TOL_FACET**2
        if facet_bad.any(): continue
        C, R, E = fit_circle(points)
        if E > TOL_RADIUS: continue
        new_entities.append(Arc(points = (np.arange(3) + 
                                          len(path.vertices) +
                                          len(new_vertices)),
                                closed = True))
        new_vertices.extend(angles_to_threepoint([0,np.pi],
                                                 C, R))        
        old_entities.extend(entities)
    path.entities = np.append(path.entities,  new_entities)
    path.vertices = np.vstack((path.vertices, new_vertices))
    path.remove_entities(old_entities)
  
