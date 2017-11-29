import numpy as np

from string import Template
from collections import deque
from xml.dom.minidom import parseString as parse_xml

from ... import util
from .. import entities as entities_mod

from ..arc import arc_center
from ...resources import get_resource
from ...constants import log
from ...constants import tol_path as tol
from ...constants import res_path as res

_template_svg = Template(get_resource('svg.xml.template'))

try:
    from svg.path import parse_path
except BaseException:
    log.warning('SVG path loading unavailable!')


def svg_to_path(file_obj, file_type=None):
    '''
    Load an SVG file into a Path2D object.

    Parameters
    -----------
    file_obj: open file object
    file_type: unused

    Returns
    -----------
    loaded: dict with kwargs for Path2D constructor
    '''
    def complex_to_float(values):
        return np.array([[i.real, i.imag] for i in values])

    def load_line(svg_line):
        points = complex_to_float([svg_line.point(0.0),
                                   svg_line.point(1.0)])
        if not starting:
            points[0] = vertices[-1]
        entities.append(entities_mod.Line(np.arange(2) + len(vertices)))
        vertices.extend(points)

    def load_arc(svg_arc):
        points = complex_to_float([svg_arc.start,
                                   svg_arc.point(.5),
                                   svg_arc.end])
        if not starting:
            points[0] = vertices[-1]
        entities.append(entities_mod.Arc(np.arange(3) + len(vertices)))
        vertices.extend(points)

    def load_quadratic(svg_quadratic):
        points = complex_to_float([svg_quadratic.start,
                                   svg_quadratic.control,
                                   svg_quadratic.end])
        if not starting:
            points[0] = vertices[-1]
        entities.append(entities_mod.Bezier(np.arange(3) + len(vertices)))
        vertices.extend(points)

    def load_cubic(svg_cubic):
        points = complex_to_float([svg_cubic.start,
                                   svg_cubic.control1,
                                   svg_cubic.control2,
                                   svg_cubic.end])
        if not starting:
            points[0] = vertices[-1]
        entities.append(entities_mod.Bezier(np.arange(4) + len(vertices)))
        vertices.extend(points)

    # first, we grab all of the path strings from the xml file
    xml = parse_xml(file_obj.read())
    paths = [p.attributes['d'].value for p in xml.getElementsByTagName('path')]

    entities = deque()
    vertices = deque()
    loaders = {'Arc': load_arc,
               'Line': load_line,
               'CubicBezier': load_cubic,
               'QuadraticBezier': load_quadratic}

    for svg_string in paths:
        starting = True
        for svg_entity in parse_path(svg_string):
            loaders[svg_entity.__class__.__name__](svg_entity)
            
    loaded = {'entities': np.array(entities),
              'vertices': np.array(vertices)}
    return loaded


def export_svg(drawing, **kwargs):
    '''
    Export a Path2D object into an SVG file.

    Parameters
    -----------
    drawing: Path2D object

    Returns
    -----------
    as_svg: str, XML formatted as SVG

    '''
    if not util.is_instance_named(drawing, 'Path2D'):
        raise ValueError('drawing must be Path2D object!')
    
    points = drawing.vertices.view(np.ndarray)
    # svg origin is in top left
    points[:,1] *= -1.0
    
    def circle_to_svgpath(center, radius, reverse):
        radius_str = format(radius, res.export)
        path_str = ' M' + format(center[0] - radius, res.export) + ','
        path_str += format(center[1], res.export)
        path_str += 'a' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(2 * radius, res.export) + ',0'
        path_str += 'a' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(-2 * radius, res.export) + ',0Z'
        return path_str

    def svg_arc(arc, reverse):
        '''
        arc string: (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
        large-arc-flag: greater than 180 degrees
        sweep flag: direction (cw/ccw)
        '''
        arc_idx = arc.points[::((reverse * -2) + 1)]
        vertices = points[arc_idx]
        vertex_start, vertex_mid, vertex_end = vertices
        center_info = arc_center(vertices)
        C, R, N, angle = (center_info['center'],
                          center_info['radius'],
                          center_info['normal'],
                          center_info['span'])
        if arc.closed:
            return circle_to_svgpath(C, R, reverse)

        large_flag = str(int(angle > np.pi))
        sweep_flag = str(int(np.cross(vertex_mid - vertex_start,
                                      vertex_end - vertex_start) > 0.0))
        R_ex = format(R, res.export)
        x_ex = format(vertex_end[0], res.export)
        y_ex = format(vertex_end[1], res.export)
        
        arc_str = move_to(arc_idx[0])
        arc_str += 'A{},{} 0 {},{} {},{}'.format(R_ex,
                                                 R_ex,
                                                 large_flag,
                                                 sweep_flag,
                                                 x_ex,
                                                 y_ex)
        return arc_str

    def svg_line(line, reverse):
        index = line.points
        if reverse:
            index = index[::-1]
        current = move_to(index[0])
        for index_end in index[1:]:
            vertex_end = points[index_end]
            x_ex = format(vertex_end[0], res.export)
            y_ex = format(vertex_end[1], res.export)
            current += 'L' + x_ex + ',' + y_ex
        return current

    def move_to(vertex_id):
        x_ex = format(points[vertex_id][0], res.export)
        y_ex = format(points[vertex_id][1], res.export)
        move_str = 'M' + x_ex + ',' + y_ex
        return move_str

    def convert_path(path, reverse=False, close=True):
        path = path[::(reverse * -2) + 1]
        converted = ''
        for i, entity_id in enumerate(path):
            entity = drawing.entities[entity_id]
            e_type = entity.__class__.__name__
            try:
                converted += converters[e_type](entity, reverse)
            except KeyError:
                log.debug('%s entity not available for export!', e_type)
        # remove leading and trailing whitespace
        converted = converted.strip()
        return converted

    converters = {'Line': svg_line,
                  'Arc': svg_arc}
    path_str = ''
    for path_index, path in enumerate(drawing.paths):
        reverse = not (path_index in drawing.root)
        path_str += convert_path(path,
                                 reverse=reverse,
                                 close=True)

    # entities which haven't been included in a closed path
    path_str += convert_path(drawing.dangling,
                             reverse=False,
                             close=False)
        
    if 'stroke_width' in kwargs:
        stroke_width = float(kwargs['stroke_width'])
    else:
        stroke_width = drawing.extents.max() / 800.0

        
    subs = {'PATH_STRING': path_str,
            'MIN_X': points[:,0].min(),
            'MIN_Y': points[:,1].min(),
            'WIDTH': drawing.extents[0],
            'HEIGHT': drawing.extents[1],
            'STROKE': stroke_width}

    result = _template_svg.substitute(subs)

    return result
