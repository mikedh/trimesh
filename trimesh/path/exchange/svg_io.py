import numpy as np

from string import Template


from ..arc import arc_center
from ..entities import Line, Arc, Bezier

from ...constants import log, tol
from ...constants import res_path as res

from ... import util
from ... import grouping
from ... import resources
from ... import exceptions

from ... transformations import transform_points, planar_matrix

try:
    # pip install svg.path
    from svg.path import parse_path
except BaseException as E:
    # will re-raise the import exception when
    # someone tries to call `parse_path`
    parse_path = exceptions.closure(E)

try:
    from lxml import etree
except BaseException as E:
    # will re-raise the import exception when
    # someone actually tries to use the module
    etree = exceptions.ExceptionModule(E)


def svg_to_path(file_obj, file_type=None):
    """
    Load an SVG file into a Path2D object.

    Parameters
    -----------
    file_obj : open file object
      Contains SVG data
    file_type: None
      Not used

    Returns
    -----------
    loaded : dict
      With kwargs for Path2D constructor
    """

    def element_transform(e, max_depth=100):
        """
        Find a transformation matrix for an XML element.
        """
        matrices = []
        current = e
        for i in range(max_depth):
            if 'transform' in current.attrib:
                matrices.extend(transform_to_matrices(
                    current.attrib['transform']))
            current = current.getparent()
            if current is None:
                break

        if len(matrices) == 0:
            return np.eye(3)
        elif len(matrices) == 1:
            return matrices[0]
        else:
            return util.multi_dot(matrices[::-1])

    # first parse the XML
    xml = etree.fromstring(file_obj.read())

    # store paths and transforms as
    # (path string, 3x3 matrix)
    paths = []

    # store every path element
    for element in xml.iter('{*}path'):
        paths.append((element.attrib['d'],
                      element_transform(element)))

    return _svg_path_convert(paths)


def transform_to_matrices(transform):
    """
    Convert an SVG transform string to an array of matrices.

    i.e. "rotate(-10 50 100)
          translate(-36 45.5)
          skewX(40)
          scale(1 0.5)"

    Parameters
    -----------
    transform : str
      Contains transformation information in SVG form

    Returns
    -----------
    matrices : (n, 3, 3) float
      Multiple transformation matrices from input transform string
    """
    # split the transform string in to components of:
    # (operation, args) i.e. (translate, '-1.0, 2.0')
    components = [
        [j.strip() for j in i.strip().split('(') if len(j) > 0]
        for i in transform.lower().split(')') if len(i) > 0]
    # store each matrix without dotting
    matrices = []
    for line in components:
        if len(line) == 0:
            continue
        elif len(line) != 2:
            raise ValueError('should always have two components!')
        key, args = line
        # convert string args to array of floats
        # support either comma or space delimiter
        values = np.array([float(i) for i in
                           args.replace(',', ' ').split()])
        if key == 'translate':
            # convert translation to a (3, 3) homogeneous matrix
            matrices.append(np.eye(3))
            matrices[-1][:2, 2] = values
        elif key == 'matrix':
            # [a b c d e f] ->
            # [[a c e],
            #  [b d f],
            #  [0 0 1]]
            matrices.append(np.vstack((
                values.reshape((3, 2)).T, [0, 0, 1])))
        elif key == 'rotate':
            # SVG rotations are in degrees
            angle = np.degrees(values[0])
            # if there are three values rotate around point
            if len(values) == 3:
                point = values[1:]
            else:
                point = None
            matrices.append(planar_matrix(theta=angle,
                                          point=point))
        elif key == 'scale':
            # supports (x_scale, y_scale) or (scale)
            mat = np.eye(3)
            mat[:2, :2] *= values
            matrices.append(mat)
        else:
            log.warning('unknown SVG transform: {}'.format(key))

    return matrices


def _svg_path_convert(paths):
    """
    Convert an SVG path string into a Path2D object

    Parameters
    -------------
    paths: list of tuples
      Containing (path string, (3, 3) matrix)

    Returns
    -------------
    drawing : dict
      Kwargs for Path2D constructor
    """
    def complex_to_float(values):
        return np.array([[i.real, i.imag] for i in values])

    def load_multi(multi):
        # load a previously parsed multiline
        return Line(np.arange(len(multi.points)) + count), multi.points

    def load_arc(svg_arc):
        # load an SVG arc into a trimesh arc
        points = complex_to_float([svg_arc.start,
                                   svg_arc.point(.5),
                                   svg_arc.end])
        return Arc(np.arange(3) + count), points

    def load_quadratic(svg_quadratic):
        # load a quadratic bezier spline
        points = complex_to_float([svg_quadratic.start,
                                   svg_quadratic.control,
                                   svg_quadratic.end])
        return Bezier(np.arange(3) + count), points

    def load_cubic(svg_cubic):
        # load a cubic bezier spline
        points = complex_to_float([svg_cubic.start,
                                   svg_cubic.control1,
                                   svg_cubic.control2,
                                   svg_cubic.end])
        return Bezier(np.arange(4) + count), points

    # store loaded values here
    entities = []
    vertices = []
    # how many vertices have we loaded
    count = 0
    # load functions for each entity
    loaders = {'Arc': load_arc,
               'MultiLine': load_multi,
               'CubicBezier': load_cubic,
               'QuadraticBezier': load_quadratic}

    class MultiLine(object):
        # An object to hold one or multiple Line entities.
        def __init__(self, lines):
            if tol.strict:
                # in unit tests make sure we only have lines
                assert all(type(L).__name__ == 'Line'
                           for L in lines)
            # get the starting point of every line
            points = [L.start for L in lines]
            # append the endpoint
            points.append(lines[-1].end)
            # convert to (n, 2) float points
            self.points = np.array([[i.real, i.imag]
                                    for i in points],
                                   dtype=np.float64)

    for path_string, matrix in paths:
        # get parsed entities from svg.path
        raw = np.array(list(parse_path(path_string)))
        # check to see if each entity is a Line
        is_line = np.array([type(i).__name__ == 'Line'
                            for i in raw])
        # find groups of consecutive lines so we can combine them
        blocks = grouping.blocks(
            is_line, min_len=1, only_nonzero=False)
        if tol.strict:
            # in unit tests make sure we didn't lose any entities
            assert np.allclose(np.hstack(blocks),
                               np.arange(len(raw)))

        # Combine consecutive lines into a single MultiLine
        parsed = []
        for b in blocks:
            if type(raw[b[0]]).__name__ == 'Line':
                # if entity consists of lines add a multiline
                parsed.append(MultiLine(raw[b]))
            else:
                # otherwise just add the entities
                parsed.extend(raw[b])
        # loop through parsed entity objects
        for svg_entity in parsed:
            # keyed by entity class name
            type_name = type(svg_entity).__name__
            if type_name in loaders:
                # get new entities and vertices
                e, v = loaders[type_name](svg_entity)
                # append them to the result
                entities.append(e)
                # create a sequence of vertex arrays
                vertices.append(transform_points(v, matrix))
                count += len(vertices[-1])

    # store results as kwargs and stack vertices
    loaded = {'entities': np.array(entities),
              'vertices': np.vstack(vertices)}
    return loaded


def export_svg(drawing,
               return_path=False,
               layers=None,
               **kwargs):
    """
    Export a Path2D object into an SVG file.

    Parameters
    -----------
    drawing : Path2D
     Source geometry
    return_path : bool
      If True return only path string not wrapped in XML
    layers : None, or [str]
      Only export specified layers

    Returns
    -----------
    as_svg : str
      XML formatted SVG, or path string
    """
    if not util.is_instance_named(drawing, 'Path2D'):
        raise ValueError('drawing must be Path2D object!')

    # copy the points and make sure they're not a TrackedArray
    points = drawing.vertices.view(np.ndarray).copy()

    # fetch the export template for SVG files
    template_svg = Template(resources.get('svg.template.xml'))

    def circle_to_svgpath(center, radius, reverse):
        radius_str = format(radius, res.export)
        path_str = ' M ' + format(center[0] - radius, res.export) + ','
        path_str += format(center[1], res.export)
        path_str += ' a ' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(2 * radius, res.export) + ',0'
        path_str += ' a ' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(-2 * radius, res.export) + ',0 Z'
        return path_str

    def svg_arc(arc, reverse):
        """
        arc string: (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
        large-arc-flag: greater than 180 degrees
        sweep flag: direction (cw/ccw)
        """
        arc_idx = arc.points[::((reverse * -2) + 1)]
        vertices = points[arc_idx]
        vertex_start, vertex_mid, vertex_end = vertices
        center_info = arc_center(vertices)
        C, R, angle = (center_info['center'],
                       center_info['radius'],
                       center_info['span'])
        if arc.closed:
            return circle_to_svgpath(C, R, reverse)

        large_flag = str(int(angle > np.pi))
        sweep_flag = str(int(np.cross(vertex_mid - vertex_start,
                                      vertex_end - vertex_start) > 0.0))

        arc_str = move_to(arc_idx[0])
        arc_str += 'A {},{} 0 {}, {} {},{}'.format(R,
                                                   R,
                                                   large_flag,
                                                   sweep_flag,
                                                   vertex_end[0],
                                                   vertex_end[1])
        return arc_str

    def move_to(vertex_id):
        x_ex = format(points[vertex_id][0], res.export)
        y_ex = format(points[vertex_id][1], res.export)
        move_str = ' M ' + x_ex + ',' + y_ex
        return move_str

    def svg_discrete(entity, reverse):
        """
        Use an entities discrete representation to export a
        curve as a polyline
        """
        discrete = entity.discrete(points)
        # if entity contains no geometry return
        if len(discrete) == 0:
            return ''
        # are we reversing the entity
        if reverse:
            discrete = discrete[::-1]
        # the format string for the SVG path
        template = ' M {},{} ' + (' L {},{}' * (len(discrete) - 1))
        # apply the data from the discrete curve
        result = template.format(*discrete.reshape(-1))
        return result

    def convert_entity(entity, reverse=False):
        if layers is not None and entity.layer not in layers:
            return ''
        # the class name of the entity
        etype = entity.__class__.__name__
        if etype == 'Arc':
            # export the exact version of the entity
            return svg_arc(entity, reverse=False)
        else:
            # just export the polyline version of the entity
            return svg_discrete(entity, reverse=False)

    # convert each entity to an SVG entity
    converted = [convert_entity(e) for e in drawing.entities]

    # append list of converted into a string
    path_str = ''.join(converted).strip()

    # return path string without XML wrapping
    if return_path:
        return path_str

    # format as XML
    if 'stroke_width' in kwargs:
        stroke_width = float(kwargs['stroke_width'])
    else:
        stroke_width = drawing.extents.max() / 800.0
    subs = {'PATH_STRING': path_str,
            'MIN_X': points[:, 0].min(),
            'MIN_Y': points[:, 1].min(),
            'WIDTH': drawing.extents[0],
            'HEIGHT': drawing.extents[1],
            'STROKE': stroke_width}
    result = template_svg.substitute(subs)
    return result
