import numpy as np

from ..entities  import Line, Arc

from ...geometry import faces_to_edges
from ...grouping import group_rows

from collections import deque

def dict_to_path(drawing_obj):
    loaders      = {'Arc': Arc, 'Line': Line}
    vertices     = np.array(drawing_obj['vertices'])
    entities     = [None] * len(drawing_obj['entities'])
    for entity_index, entity in enumerate(drawing_obj['entities']):
        entities[entity_index] = loaders[entity['type']](points = entity['points'],
                                                         closed = entity['closed'])
    return {'entities' : entities,
            'vertices' : vertices}

def lines_to_path(lines):
    '''
    Given a set of line segments (n, 2, [2|3]), populate a path
    '''
    shape = np.shape(lines)
    dimension = shape[-1]
    if len(shape) == 2:
        lines     = np.column_stack((lines[:-1], lines[1:])).reshape((-1,2,dimension))
        shape     = np.shape(lines)

    if ((len(shape) != 3) or 
        (shape[1] != 2) or 
        (not (shape[2] in [2,3]))):
        raise ValueError('Lines MUST be (n, 2, [2|3])')
    entities = deque()
    for i in range(0, (len(lines) * 2) - 1, 2):
        entities.append(Line([i, i+1]))
    vertices = lines.reshape((-1,dimension))
    return {'entities' : entities,
            'vertices' : vertices}

def polygon_to_lines(polygon):
    '''
    Given a shapely.geometry.Polygon, convert it to a set
    of (n,2,2) line segments.
    '''
    def append_boundary(boundary):
        vertices = np.array(boundary.coords)
        lines.append(np.column_stack((vertices[:-1],
                                      vertices[1:])).reshape((-1,2,2)))
    lines = deque()
    append_boundary(polygon.exterior)
    for interior in polygon.interiors:
        append_boundary(interior)
    return np.vstack(lines)

def faces_to_path(mesh, face_ids=None):
    '''
    Given a mesh and face indices, find the outline edges and
    turn them into a Path3D.

    Arguments
    ---------
    mesh:  Trimesh object
    facet: (n) list of indices of mesh.faces

    Returns
    ---------
    dict
    '''
    if face_ids is None: faces = mesh.faces
    else:                faces = mesh.faces[[face_ids]]

    edges        = faces_to_edges(faces)
    unique_edges = group_rows(edges, require_count=1)
    segments     = mesh.vertices[edges[unique_edges]]        
    return lines_to_path(segments)
