import numpy as np
import json
import os 
import sys

from ..constants import *
from ..entities  import Line, Arc, Bezier, arc_center

from ...geometry import faces_to_edges
from ...grouping import group_rows
from ...util     import is_sequence

from collections import deque
from xml.dom.minidom import parseString as parse_xml

try:     
    from svg.path import parse_path
except:
    log.warn('SVG path loading unavailable!')

if sys.version_info.major >= 3:
    # python 3
    from io import StringIO
    basestring = str 
else:
    from cStringIO import StringIO

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
        entities.append(Arc(np.arange(3)+len(vertices)))
        vertices.extend(points)
    def load_quadratic(svg_quadratic):
        points = complex_to_float([svg_quadratic.start, 
                                   svg_quadratic.control, 
                                   svg_quadratic.end])
        entities.append(Bezier(np.arange(3)+len(vertices)))
        vertices.extend(points)
    def load_cubic(svg_cubic):
        points = complex_to_float([svg_cubic.start, 
                                   svg_cubic.control1, 
                                   svg_cubic.control2, 
                                   svg_cubic.end])
        entities.append(Bezier(np.arange(4)+len(vertices)))
        vertices.extend(points)
    
    # first, we grab all of the path strings from the xml file
    xml   = parse_xml(file_obj.read())
    paths = [p.attributes['d'].value for p in xml.getElementsByTagName('path')]

    entities = deque()
    vertices = deque()  
    loaders  = {'Arc'             : load_arc,
                'Line'            : load_line,
                'CubicBezier'     : load_cubic,
                'QuadraticBezier' : load_quadratic}

    for svg_string in paths:
        for svg_entity in parse_path(svg_string):
            loaders[svg_entity.__class__.__name__](svg_entity)
    return {'entities' : np.array(entities),
            'vertices' : np.array(vertices)}
 
def path_to_svg(drawing):
    '''
    Will turn a path drawing into an SVG path string. 

    'holes' will be in reverse order, so they can be rendered as holes by
    rendering libraries
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
        '''
        arc string: (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
        large-arc-flag: greater than 180 degrees
        sweep flag: direction (cw/ccw)
        '''
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
    def convert_path(path, reverse=False):
        path     = path[::(reverse*-2) + 1]
        path_str = svg_moveto(drawing.entities[path[0]].end_points()[-reverse])
        for i, entity_id in enumerate(path):
            entity    = drawing.entities[entity_id]
            path_str += converters[entity.__class__.__name__](entity, reverse)
        path_str += 'Z'
        return path_str
    converters = {'Line'  : svg_line,
                  'Arc'   : svg_arc}
    path_str = ''
    for path_index, path in enumerate(drawing.paths):
        reverse   = not (path_index in drawing.root_paths)
        path_str += convert_path(path, reverse)
    return path_str
