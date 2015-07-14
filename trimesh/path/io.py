import numpy as np
import json
import os 
import sys

from collections import deque

from .constants import *
from .path      import Path2D, Path3D
from .entities  import Line, Arc, Bezier, arc_center
from .dxf       import dxf_to_vector

from ..geometry import faces_to_edges
from ..grouping import group_rows
from ..util     import is_sequence

PY3 = sys.version_info.major >= 3
if PY3: 
    from io import StringIO
    basestring = str 
else:
    from cStringIO import StringIO

def available_formats():
    return _LOADERS.keys()

def load_path(obj, file_type=None):
    '''
    Utility function which can be passed a filename, file object, or list of lines
    '''
    if hasattr(obj, 'read'):
        loaded = _LOADERS[file_type](obj)
        obj.close()
    elif isinstance(obj, basestring):
        file_obj  = open(obj, 'rb')
        file_type = os.path.splitext(obj)[-1][1:].lower()
        loaded = _LOADERS[file_type](file_obj)
        file_obj.close()
    elif obj.__class__.__name__ == 'Polygon':
        lines  = polygon_to_lines(obj)
        loaded = lines_to_path(lines)
    elif is_sequence(obj):
        loaded = lines_to_path(obj)
    else:
        raise NameError('Not a supported object type!')
    return loaded

def dxf_to_path(file_obj, type=None):
    '''
    Load a dxf file into a path container
    '''
    
    entities, vertices = dxf_to_vector(file_obj)
    vector = Path2D(entities, vertices)
    vector.metadata['is_planar'] = True
    return vector

def dict_to_path(drawing_obj):
    loaders      = {'Arc': Arc, 'Line': Line}
    vertices     = np.array(drawing_obj['vertices'])
    entities     = [None] * len(drawing_obj['entities'])

    for entity_index, entity in enumerate(drawing_obj['entities']):
        entities[entity_index] = loaders[entity['type']](points = entity['points'],
                                                         closed = entity['closed'])

    drawing_type = [Path2D, Path3D][vertices.shape[1] - 2]
    return drawing_type(entities=entities, vertices=vertices)
    
def lines_to_path(lines):
    '''
    Given a set of line segments (n, 2, [2|3]), populate a path
    '''
    shape = np.shape(lines)
    if len(shape) == 2:
        dimension = shape[1]
        lines     = np.column_stack((lines[:-1], lines[1:])).reshape((-1,2,dimension))
        shape     = np.shape(lines)

    if ((len(shape) != 3) or 
        (shape[1] != 2) or 
        (not (shape[2] in [2,3]))):
        raise NameError('Lines MUST be (n, 2, [2|3])')
    entities = deque()
    for i in range(0, (len(lines) * 2) - 1, 2):
        entities.append(Line([i, i+1]))
    vector_type = [Path2D, Path3D][shape[2]-2]
    vector      = vector_type(entities = np.array(entities),
                              vertices = lines.reshape((-1, shape[2])))
    return vector

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

def svg_to_path(file_obj, file_type=None):
    def complex_to_float(values):
        return np.array([[i.real, i.imag] for i in values])

    def load_line(svg_line):
        points = complex_to_float([svg_line.start, 
                                   svg_line.end])
        entities.append(Line(np.arange(2)+len(vertices)))
        vertices.extend(points)

    def load_arc(svg_arc):
        points = complex_to_float([svg_arc.start, 
                                   svg_arc.point(.5), 
                                   svg_arc.end])
        entities.append(Arc(np.arange(3))+len(vertices))
        vertices.extend(points)

    def load_cubic(svg_cubic):
        points = complex_to_float([svg_cubic.start, 
                                   svg_cubic.control1, 
                                   svg_cubic.control2, 
                                   svg_cubic.end])
        entities.append(Bezier(np.arange(len(points))+len(vertices)))
        vertices.extend(points)

    def load_quadratic(svg_quadratic):
        points = complex_to_float([svg_quadratic.start, 
                                   svg_quadratic.control, 
                                   svg_quadratic.end])
        entities.append(Bezier(np.arange(len(points))+len(vertices)))
        vertices.extend(points)

    from svg.path        import parse_path
    from xml.dom.minidom import parseString as parse_xml

    # first, we grab all of the path strings from the xml file
    xml   = parse_xml(file_obj.read())
    paths = [p.attributes['d'].value for p in xml.getElementsByTagName('path')]
    loaders = {'Arc'             : load_arc,
               'Line'            : load_line,
               'CubicBezier'     : load_cubic,
               'QuadraticBezier' : load_quadratic}
               
    entities = deque()
    vertices = deque()
    
    for svg_string in paths:
        for svg_entity in parse_path(svg_string):
            loaders[svg_entity.__class__.__name__](svg_entity)
    vector = Path2D(entities = np.array(entities), 
                    vertices = np.array(vertices))
    return vector
 
def path_to_svg(drawing):
    '''
    Will turn a path drawing into an SVG path string. 

    'holes' will be in reverse order, so they can be rendered as holes by the 
    '''
    def circle_to_svgpath(center, radius, reverse):
        radius_str = format(radius, EXPORT_PRECISION)
        path_str  = '  M' + format(center[0]-radius, EXPORT_PRECISION) + ',' 
        path_str += format(center[1], EXPORT_PRECISION)       
        path_str += 'a' + radius_str + ',' + radius_str  
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(2*radius, EXPORT_PRECISION) +  ',0'
        path_str += 'a' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(-2*radius, EXPORT_PRECISION) + ',0Z  '
        return path_str

    def svg_arc(arc, reverse):
        #A: elliptical arc	(rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
        #large-arc-flag: greater than 180 degrees
        #sweep flag: direction (cw/ccw)

        vertices = drawing.vertices[arc.points[::((reverse*-2) + 1)]]        
        vertex_start, vertex_mid, vertex_end = vertices
        C, R, N, angle = arc_center(vertices)
        if arc.closed: return circle_to_svgpath(C, R, reverse)
        large_flag = str(int(angle > np.pi))
        sweep_flag = str(int(np.cross(vertex_mid-vertex_start, 
                                      vertex_end-vertex_start) > 0))
        R_ex = format(R, EXPORT_PRECISION)
        x_ex = format(vertex_end[0],EXPORT_PRECISION)
        y_ex = format(vertex_end [1],EXPORT_PRECISION)
        arc_str  = 'A' + R_ex + ',' + R_ex + ' 0 ' 
        arc_str += large_flag + ',' + sweep_flag + ' '
        arc_str += x_ex + ',' + y_ex
        return arc_str

    def svg_line(line, reverse):
        vertex_end = drawing.vertices[line.points[-(not reverse)]]
        x_ex = format(vertex_end[0], EXPORT_PRECISION) 
        y_ex = format(vertex_end[1], EXPORT_PRECISION) 
        line_str = 'L' + x_ex + ',' + y_ex
        return line_str

    def svg_moveto(vertex_id):
        x_ex = format(drawing.vertices[vertex_id][0], EXPORT_PRECISION) 
        y_ex = format(drawing.vertices[vertex_id][1], EXPORT_PRECISION) 
        move_str = 'M' + x_ex + ',' + y_ex
        return move_str

    converters = {'Line'  : svg_line,
                  'Arc'   : svg_arc}

    def convert_path(path, reverse=False):
        path     = path[::(reverse*-2) + 1]
        path_str = svg_moveto(drawing.entities[path[0]].end_points()[-reverse])
        for i, entity_id in enumerate(path):
            entity    = drawing.entities[entity_id]
            path_str += converters[entity.__class__.__name__](entity, reverse)
        path_str += 'Z'
        return path_str
        
    path_str = ''
    for path_index, path in enumerate(drawing.paths):
        reverse   = not (path_index in drawing.root_paths)
        path_str += convert_path(path, reverse)
    return path_str

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
    path: Path3D of the outline of the facet
    '''
    if face_ids is None: faces = mesh.faces
    else:                faces = mesh.faces[[face_ids]]

    edges        = faces_to_edges(faces)
    unique_edges = group_rows(edges, require_count=1)
    segments     = mesh.vertices[edges[unique_edges]]        
    path         = lines_to_path(segments)
    return path

_LOADERS = {'dxf': dxf_to_path,
            'svg': svg_to_path}
