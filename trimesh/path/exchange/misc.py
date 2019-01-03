import numpy as np

from ... import util
from ... import graph
from ... import grouping

from ..entities import Line, Arc

from collections import deque


def dict_to_path(drawing_obj):
    """
    Turn a pure dict into a dict containing entity objects that
    can be sent directly to a Path constructor.

    Parameters
    -----------
    as_dict: dict, with keys ['vertices', 'entities']

    Returns
    ------------
    kwargs: dict, with keys ['vertices', 'entities']
    """
    loaders = {'Arc': Arc, 'Line': Line}
    vertices = np.array(drawing_obj['vertices'])
    entities = [None] * len(drawing_obj['entities'])
    for entity_index, entity in enumerate(drawing_obj['entities']):
        entities[entity_index] = loaders[entity['type']](
            points=entity['points'], closed=entity['closed'])
    return {'entities': entities,
            'vertices': vertices}


def lines_to_path(lines):
    """
    Turn line segments into a Path2D or Path3D object.

    Parameters
    ------------
    lines: (n, 2, 2) float, Path2D object from line segments
           (n, 2, 3) float, Path3D object from line segments
           (n, 2) float: Path2D object, assumes vertices are connected
           (n, 3) float: Path3D object, assumes vertices are connected

    Returns
    -----------
    kwargs : dict
        kwargs for Path constructor
    """
    lines = np.asanyarray(lines, dtype=np.float64)

    if util.is_shape(lines, (-1, (2, 3))):
        # the case where we have a list of points
        # we are going to assume they are connected
        result = {'entities': np.array([Line(np.arange(len(lines)))]),
                  'vertices': lines}
        return result
    elif util.is_shape(lines, (-1, 2, (2, 3))):
        # case where we have line segments in 2D or 3D
        dimension = lines.shape[-1]
        # convert lines to even number of (n, dimension) points
        lines = lines.reshape((-1, dimension))
        # merge duplicate vertices
        unique, inverse = grouping.unique_rows(lines)
        # use scipy edges_to_path to skip creating
        # a bajillion individual line entities which
        # will be super slow vs. fewer polyline entities
        return edges_to_path(edges=inverse.reshape((-1, 2)),
                             vertices=lines[unique])
    else:
        raise ValueError('Lines must be (n,(2|3)) or (n,2,(2|3))')
    return result


def polygon_to_path(polygon):
    """
    Load shapely Polygon objects into a trimesh.path.Path2D object

    Parameters
    -------------
    polygon: shapely.geometry.Polygon object

    Returns
    -------------
    kwargs: dict, keyword arguments for Path2D constructor
    """
    entities = deque([Line(points=np.arange(len(polygon.exterior.coords)))])
    vertices = deque(np.array(polygon.exterior.coords))

    # append interiors as single Line objects
    for boundary in polygon.interiors:
        entities.append(Line(np.arange(len(boundary.coords)) +
                             len(vertices)))
        vertices.extend(boundary.coords)

    return {'entities': np.array(entities),
            'vertices': np.array(vertices)}


def linestrings_to_path(multi):
    """
    Load shapely LineString objects into a trimesh.path.Path2D object

    Parameters
    -------------
    multi: LineString or MultiLineString

    Returns
    -------------
    kwargs: dict, keyword arguments for Path2D constructor
    """
    entities = deque()
    vertices = deque()

    if not util.is_sequence(multi):
        multi = [multi]

    for line in multi:
        if hasattr(line, 'coords'):
            coords = np.array(line.coords)
            entities.append(Line(np.arange(len(coords)) +
                                 len(vertices)))
            vertices.extend(coords)

    return {'entities': np.array(entities),
            'vertices': np.array(vertices)}


def faces_to_path(mesh, face_ids=None, **kwargs):
    """
    Given a mesh and face indices find the outline edges and
    turn them into a Path3D.

    Parameters
    ---------
    mesh:      Trimesh object
    face_ids: (n) list of indices of mesh.faces

    Returns
    ---------
    kwargs: dict, kwargs for Path3D constructor
    """
    if face_ids is None:
        edges = mesh.edges_sorted
    else:
        # take advantage of edge ordering to index as single row
        edges = mesh.edges_sorted.reshape(
            (-1, 6))[face_ids].reshape((-1, 2))

    # an edge which occurs onely once is on the boundary
    unique_edges = grouping.group_rows(edges,
                                       require_count=1)

    kwargs.update(edges_to_path(edges=edges[unique_edges],
                                vertices=mesh.vertices))

    return kwargs


def edges_to_path(edges, vertices, **kwargs):
    """
    Given an edge list of indices and associated vertices
    representing lines, generate kwargs for a Path object.

    Parameters
    -----------
    edges:    (n,2)       int, vertex index of lines
    vertices: (m,(2,3)) float, vertex positions

    Returns
    ----------
    kwargs: dict, kwargs for Path constructor
    """
    # sequence of ordered traversals
    dfs = graph.traversals(edges, mode='dfs')
    # make sure every consecutive index in DFS
    # traversal is an edge in the source edge list
    dfs_connected = graph.fill_traversals(dfs, edges=edges)
    # kwargs for Path constructor
    # turn traversals into Line objects
    kwargs.update({'entities': [Line(d) for d in dfs_connected],
                   'vertices': vertices})
    return kwargs
