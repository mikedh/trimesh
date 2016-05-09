import numpy as np

from ...constants import log
from ..entities  import Line, Arc, Bezier

from collections import deque
from xml.dom.minidom import parseString as parse_xml

try:     
    from svg.path import parse_path
except:
    log.warning('SVG path loading unavailable!')

def svg_to_path(file_obj, file_type=None):
    def complex_to_float(values):
        return np.array([[i.real, i.imag] for i in values])

    def load_line(svg_line):
        points = complex_to_float([svg_line.point(0.0),
                                   svg_line.point(1.0)])
        if not starting: points[0] = vertices[-1]
        entities.append(Line(np.arange(2)+len(vertices)))
        vertices.extend(points)

    def load_arc(svg_arc):
        points = complex_to_float([svg_arc.start, 
                                   svg_arc.point(.5), 
                                   svg_arc.end])
        if not starting: points[0] = vertices[-1]
        entities.append(Arc(np.arange(3)+len(vertices)))
        vertices.extend(points)
    def load_quadratic(svg_quadratic):
        points = complex_to_float([svg_quadratic.start, 
                                   svg_quadratic.control, 
                                   svg_quadratic.end])
        if not starting: points[0] = vertices[-1]
        entities.append(Bezier(np.arange(3)+len(vertices)))
        vertices.extend(points)
    def load_cubic(svg_cubic):
        points = complex_to_float([svg_cubic.start, 
                                   svg_cubic.control1, 
                                   svg_cubic.control2, 
                                   svg_cubic.end])
        if not starting: points[0] = vertices[-1]
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
        starting = True
        for svg_entity in parse_path(svg_string):
            loaders[svg_entity.__class__.__name__](svg_entity)

    return {'entities' : np.array(entities),
            'vertices' : np.array(vertices)}
