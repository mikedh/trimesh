import json
import base64
import collections

import numpy as np

from copy import deepcopy

from ..arc import arc_center
from ..entities import Line, Arc, Bezier

from ...constants import log, tol
from ...constants import res_path as res

from ... import util
from ... import grouping
from ... import resources
from ... import exceptions

from ...util import jsonify
from ...transformations import transform_points, planar_matrix

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

# store any additional properties using a trimesh namespace
_ns_name = 'trimesh'
_ns_url = 'https://github.com/mikedh/trimesh'
_ns = '{{{}}}'.format(_ns_url)


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

    def element_transform(e, max_depth=10):
        """
        Find a transformation matrix for an XML element.

        Parameters
        --------------
        e : lxml.etree.Element
          Element to search upwards from.
        max_depth : int
          Maximum depth to search for transforms.
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
    tree = etree.fromstring(file_obj.read())
    # store paths and transforms as
    # (path string, 3x3 matrix)
    paths = []
    for element in tree.iter('{*}path'):
        # store every path element attributes and transform
        paths.append((element.attrib,
                      element_transform(element)))

    try:
        # see if the SVG should be reproduced as a scene
        force = tree.attrib[_ns + 'class']
    except BaseException:
        force = None

    result = _svg_path_convert(paths=paths, force=force)
    try:
        # get overall metadata from JSON string if it exists
        result['metadata'] = _decode(
            tree.attrib[_ns + 'metadata'])
    except KeyError:
        # not in the trimesh ns
        pass
    except BaseException:
        # no metadata stored with trimesh ns
        log.warning('failed metadata', exc_info=True)

    # if the result is a scene try to get the metadata
    # for each subgeometry here
    if 'geometry' in result:
        try:
            # get per-geometry metadata if available
            bag = _decode(
                tree.attrib[_ns + 'metadata_geometry'])
            for name, meta in bag.items():
                if name in result['geometry']:
                    # assign this metadata to the geometry
                    result['geometry'][name]['metadata'] = meta
        except KeyError:
            # no stored geometry metadata so ignore
            pass
        except BaseException:
            # failed to load existing metadata
            log.warning('failed metadata', exc_info=True)

    return result


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


def _svg_path_convert(paths, force=None):
    """
    Convert an SVG path string into a Path2D object

    Parameters
    -------------
    paths: list of tuples
      Containing (path string, (3, 3) matrix, metadata)

    Returns
    -------------
    drawing : dict
      Kwargs for Path2D constructor
    """
    def complex_to_float(values):
        return np.array([[i.real, i.imag] for i in values],
                        dtype=np.float64)

    def load_multi(multi):
        # load a previously parsed multiline
        return (Line(points=np.arange(len(multi.points)) + counts[name]),
                multi.points)

    def load_arc(svg_arc):
        # load an SVG arc into a trimesh arc
        points = complex_to_float([svg_arc.start,
                                   svg_arc.point(0.5),
                                   svg_arc.end])
        return Arc(points=np.arange(3) + counts[name]), points

    def load_quadratic(svg_quadratic):
        # load a quadratic bezier spline
        points = complex_to_float([svg_quadratic.start,
                                   svg_quadratic.control,
                                   svg_quadratic.end])
        return Bezier(points=np.arange(3) + counts[name]), points

    def load_cubic(svg_cubic):
        # load a cubic bezier spline
        points = complex_to_float([svg_cubic.start,
                                   svg_cubic.control1,
                                   svg_cubic.control2,
                                   svg_cubic.end])
        return Bezier(np.arange(4) + counts[name]), points

    class MultiLine(object):
        # An object to hold one or multiple Line entities.
        def __init__(self, lines):
            if tol.strict:
                # in unit tests make sure we only have lines
                assert all(type(L).__name__ in ('Line', 'Close')
                           for L in lines)
            # get the starting point of every line
            points = [L.start for L in lines]
            # append the endpoint
            points.append(lines[-1].end)
            # convert to (n, 2) float points
            self.points = np.array([[i.real, i.imag]
                                    for i in points],
                                   dtype=np.float64)

    # load functions for each entity
    loaders = {'Arc': load_arc,
               'MultiLine': load_multi,
               'CubicBezier': load_cubic,
               'QuadraticBezier': load_quadratic}

    entities = collections.defaultdict(list)
    vertices = collections.defaultdict(list)
    counts = collections.defaultdict(lambda: 0)

    for attrib, matrix in paths:
        # the path string is stored under `d`
        path_string = attrib['d']

        # get the name of the geometry if trimesh specified it
        # note that the get will by default return `None`
        name = _decode(attrib.get(_ns + 'name'))
        # get parsed entities from svg.path
        raw = np.array(list(parse_path(path_string)))
        # if there is no path string exit
        if len(raw) == 0:
            continue
        # check to see if each entity is "line-like"
        is_line = np.array([type(i).__name__ in
                            ('Line', 'Close')
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
            if type(raw[b[0]]).__name__ in ('Line', 'Close'):
                # if entity consists of lines add a multiline
                parsed.append(MultiLine(raw[b]))
            else:
                # otherwise just add the entities
                parsed.extend(raw[b])
        try:
            # try to retrieve any trimesh attributes as metadata
            entity_meta = {
                k.lstrip(_ns): _decode(v)
                for k, v in attrib.items()
                if k[1:].startswith(_ns_url)}
        except BaseException:
            entity_meta = {}

        # loop through parsed entity objects
        for svg_entity in parsed:
            # keyed by entity class name
            type_name = type(svg_entity).__name__
            if type_name in loaders:
                # get new entities and vertices
                e, v = loaders[type_name](svg_entity)
                e.metadata.update(entity_meta)
                # append them to the result
                entities[name].append(e)
                # transform the vertices by the matrix and append
                vertices[name].append(transform_points(v, matrix))
                counts[name] += len(v)

    geoms = {name: {'vertices': np.vstack(v),
                    'entities': entities[name]}
             for name, v in vertices.items()}
    if len(geoms) > 1 or force == 'Scene':
        kwargs = {'geometry': geoms}
    else:
        # return a Path2D
        kwargs = next(iter(geoms.values()))

    return kwargs


def _entities_to_str(entities,
                     vertices,
                     name=None,
                     only_layers=None):
    """
    Convert the entities of a path to path strings.

    Parameters
    ------------
    entities : (n,) list
      Entity objects
    vertices : (m, 2) float
      Vertices entities reference
    name : any
      Trimesh namespace name to assign to entity
    only_layers : set
      Only export these layers if passed
    """

    points = vertices.copy()

    def circle_to_svgpath(center, radius, reverse):
        c = 'M {x},{y} a {r},{r},0,1,0,{d},0 a {r},{r},0,1,{rev},-{d},0 Z'
        fmt = '{{:{}}}'.format(res.export)
        return c.format(x=fmt.format(center[0] - radius),
                        y=fmt.format(center[1]),
                        r=fmt.format(radius),
                        d=fmt.format(2.0 * radius),
                        rev=int(bool(reverse)))

    def svg_arc(arc, reverse=False):
        """
        arc string: (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
        large-arc-flag: greater than 180 degrees
        sweep flag: direction (cw/ccw)
        """
        arc_idx = arc.points[::((reverse * -2) + 1)]
        vertices = points[arc_idx]
        vertex_start, vertex_mid, vertex_end = vertices
        center_info = arc_center(
            vertices, return_normal=False, return_angle=True)
        C, R, angle = (center_info['center'],
                       center_info['radius'],
                       center_info['span'])
        if arc.closed:
            return circle_to_svgpath(C, R, reverse)
        large_flag = str(int(angle > np.pi))
        sweep_flag = str(int(np.cross(
            vertex_mid - vertex_start,
            vertex_end - vertex_start) > 0.0))
        return (move_to(arc_idx[0]) +
                'A {R},{R} 0 {}, {} {},{}'.format(
                    large_flag,
                    sweep_flag,
                    vertex_end[0],
                    vertex_end[1],
                    R=R))

    def move_to(vertex_id):
        x_ex = format(points[vertex_id][0], res.export)
        y_ex = format(points[vertex_id][1], res.export)
        move_str = 'M ' + x_ex + ',' + y_ex
        return move_str

    def svg_discrete(entity, reverse=False):
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
        result = ('M {:0.5f},{:0.5f} ' + (
            ' L {:0.5f},{:0.5f}' * (len(discrete) - 1))).format(
                *discrete.reshape(-1))
        return result

    # tuples of (metadata, path string)
    pairs = []

    for entity in entities:
        if only_layers is not None and entity.layer not in only_layers:
            continue
        # the class name of the entity
        etype = entity.__class__.__name__
        if etype == 'Arc':
            # export the exact version of the entity
            path_string = svg_arc(entity)
        else:
            # just export the polyline version of the entity
            path_string = svg_discrete(entity)
        meta = deepcopy(entity.metadata)
        if name is not None:
            meta['name'] = name
        pairs.append((meta, path_string))
    return pairs


def export_svg(drawing,
               return_path=False,
               only_layers=None,
               **kwargs):
    """
    Export a Path2D object into an SVG file.

    Parameters
    -----------
    drawing : Path2D
     Source geometry
    return_path : bool
      If True return only path string not wrapped in XML

    Returns
    -----------
    as_svg : str
      XML formatted SVG, or path string
    """
    # collect custom attributes for the overall export
    attribs = {'class': type(drawing).__name__}

    if util.is_instance_named(drawing, 'Scene'):
        pairs = []

        geom_meta = {}
        for name, geom in drawing.geometry.items():
            if not util.is_instance_named(geom, 'Path2D'):
                continue
            geom_meta[name] = geom.metadata
            # a pair of (metadata, path string)
            pairs.extend(_entities_to_str(
                entities=geom.entities,
                vertices=geom.vertices,
                name=name,
                only_layers=only_layers))
        if len(geom_meta) > 0:
            # encode the whole metadata bundle here to avoid
            # polluting the file with a ton of loose attribs
            attribs['metadata_geometry'] = _encode(geom_meta)
    elif util.is_instance_named(drawing, 'Path2D'):
        pairs = _entities_to_str(
            entities=drawing.entities,
            vertices=drawing.vertices,
            only_layers=only_layers)

    else:
        raise ValueError('drawing must be Scene or Path2D object!')

    # return path string without XML wrapping
    if return_path:
        return ' '.join(v[1] for v in pairs)

    # fetch the export template for the base SVG file
    template_svg = resources.get('templates/base.svg')

    elements = []
    for meta, path_string in pairs:
        # create a simple path element
        elements.append('<path d="{d}" {attr}/>'.format(
            d=path_string,
            attr=_format_attrib(meta)))

    # format as XML
    if 'stroke_width' in kwargs:
        stroke_width = float(kwargs['stroke_width'])
    else:
        # set stroke to something OK looking
        stroke_width = drawing.extents.max() / 800.0
    try:
        # store metadata in XML as JSON -_-
        attribs['metadata'] = _encode(drawing.metadata)
    except BaseException:
        # log failed metadata encoding
        log.warning('failed to encode', exc_info=True)

    subs = {'elements': '\n'.join(elements),
            'min_x': drawing.bounds[0][0],
            'min_y': drawing.bounds[0][1],
            'width': drawing.extents[0],
            'height': drawing.extents[1],
            'stroke_width': stroke_width,
            'attribs': _format_attrib(attribs)}
    return template_svg.format(**subs)


def _format_attrib(attrib):
    """
    Format attribs into the trimesh namespace.

    Parameters
    -----------
    attrib : dict
      Bag of keys and values.
    """
    bag = {k: _encode(v) for k, v in attrib.items()}
    return '\n'.join('{ns}:{key}="{value}"'.format(
        ns=_ns_name, key=k, value=v)
        for k, v in bag.items()
        if len(k) > 0 and v is not None
        and len(v) > 0)


def _encode(stuff):
    """
    Wangle things into a string.

    Parameters
    -----------
    stuff : dict, str
      Thing to pack

    Returns
    ------------
    encoded : str
      Packaged into url-safe b64 string
    """
    if util.is_string(stuff) and '"' not in stuff:
        return stuff
    pack = base64.urlsafe_b64encode(jsonify(
        stuff, separators=(',', ':')).encode('utf-8'))
    result = 'base64,' + util.decode_text(pack)
    if tol.strict:
        # make sure we haven't broken the things
        _deep_same(stuff, _decode(result))

    return result


def _deep_same(original, other):
    """
    Do a recursive comparison of two items to check
    our encoding scheme in unit tests.

    Parameters
    -----------
    original : str, bytes, list, dict
      Original item
    other : str, bytes, list, dict
      Item that should be identical

    Raises
    ------------
    AssertionError
      If items are not the same.
    """
    # ndarrays will be converted to lists
    # but otherwise types should be identical
    if isinstance(original, np.ndarray):
        assert isinstance(other, (list, np.ndarray))
    elif util.is_string(original):
        # handle python 2+3 unicode vs str
        assert util.is_string(other)
    else:
        # otherwise they should be the same type
        assert isinstance(original, type(other))

    if isinstance(original, (str, bytes)):
        # string and bytes should just be identical
        assert original == other
        return
    elif isinstance(original, (float, int, np.ndarray)):
        # for numeric classes use numpy magic comparison
        # which includes an epsilon for floating point
        assert np.allclose(original, other)
        return
    elif isinstance(original, list):
        # lengths should match
        assert len(original) == len(other)
        # every element should be identical
        for a, b in zip(original, other):
            _deep_same(a, b)
        return

    # we should have special-cased everything else by here
    assert isinstance(original, dict)

    # all keys should match
    assert set(original.keys()) == set(other.keys())
    # do a recursive comparison of the values
    for k in original.keys():
        _deep_same(original[k], other[k])


def _decode(bag):
    """
    Decode a base64 bag of stuff.

    Parameters
    ------------
    bag : str
      Starts with `base64,`

    Returns
    -------------
    loaded : dict
      Loaded bag of stuff
    """
    if bag is None:
        return
    text = util.decode_text(bag)
    if text.startswith('base64,'):
        return json.loads(base64.urlsafe_b64decode(
            text[7:].encode('utf-8')).decode('utf-8'))
    return text
