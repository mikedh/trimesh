import numpy as np
import collections

import numpy as np
import json

from string import Template
from ...resources import get_resource
from ...util import three_dimensionalize

from ...constants import log
from ...constants import tol_path as tol
from ..entities import Line, Arc, BSpline
from ..util import angles_to_threepoint, is_ccw
from ...util import is_binary_file, multi_dict, make_sequence


_templates_dxf = {k: Template(v) for k, v in json.loads(
    get_resource('dxf.json.template')).items()}


# unit codes
_DXF_UNITS = {1: 'inches',
              2: 'feet',
              3: 'miles',
              4: 'millimeters',
              5: 'centimeters',
              6: 'meters',
              7: 'kilometers',
              8: 'microinches',
              9: 'mils',
              10: 'yards',
              11: 'angstroms',
              12: 'nanometers',
              13: 'microns',
              14: 'decimeters',
              15: 'decameters',
              16: 'hectometers',
              17: 'gigameters',
              18: 'AU',
              19: 'light years',
              20: 'parsecs'}


def get_key(blob, field, code):
    try:
        line = blob[np.nonzero(blob[:, 1] == field)[0][0] + 1]
    except IndexError:
        return None
    if line[0] == code:
        return int(line[1])
    else:
        return None


def load_dxf(file_obj):
    '''
    Load a DXF file to a dictionary containing vertices and entities.

    Parameters
    ----------
    file_obj: file or file- like object (has object.read method)

    Returns
    ----------
    result: dict, keys are  entities, vertices and metadata
    '''
    def update_metadata(e_data):
        '''
        Pull metadata based on group code
        '''
        # which keys should we extract from the entity data
        # DXF group code : our metadata key
        candidates = {'8': 'layers'}
        for k, v in candidates.items():
            # dict.get will return None if key is not present,
            # maintaining correct length of deque for values
            # so the indexes of a metadata entry correspond with
            # an entity object.
            e_value = make_sequence(e_data.get(k))
            entity_metadata[v].extend(e_value)

    def convert_line(e):
        entities.append(Line(len(vertices) + np.arange(2)))
        vertices.extend(np.array([[e['10'], e['20']],
                                  [e['11'], e['21']]], dtype=np.float64))

    def convert_circle(e):
        R = float(e['40'])
        C = np.array([e['10'],
                      e['20']]).astype(np.float64)
        points = angles_to_threepoint([0, np.pi], C[0:2], R)
        entities.append(Arc(points=(len(vertices) + np.arange(3)),
                            closed=True))
        vertices.extend(points)

    def convert_arc(e):
        R = float(e['40'])
        C = np.array([e['10'],
                      e['20']], dtype=np.float64)
        A = np.radians(np.array([e['50'],
                                 e['51']], dtype=np.float64))
        points = angles_to_threepoint(A, C[0:2], R)
        entities.append(Arc(len(vertices) + np.arange(3),
                            closed=False))
        vertices.extend(points)

    def convert_polyline(e):
        lines = np.column_stack((e['10'], e['20'])).astype(np.float64)

        # 70 is the closed flag for polylines
        # if the closed flag is set, make sure we connect the end to the
        # beginning
        if ('70' in e and
                int(e['70'][0]) == 1):
            lines = np.vstack((lines, lines[:1]))

        # 42 is the bulge flag for polylines
        # "bulge" is autocad for "add a stupid arc using implicit flags
        # in my otherwise normal polygon"
        if '42' in e:
            log.warning('polyline with bulge %s detected, ignoring!',
                        e['42'])

        entities.append(Line(np.arange(len(lines)) + len(vertices)))
        vertices.extend(lines)

    def convert_bspline(e):
        # in the DXF there are n points and n ordered fields
        # with the same group code
        points = np.column_stack((e['10'], e['20'])).astype(np.float64)
        knots = np.array(e['40']).astype(np.float64)
        # check euclidean distance to see if closed
        closed = np.linalg.norm(points[0] - points[-1]) < tol.merge
        # if it is closed, make sure it is CCW for later polygon happiness
        if closed and (not is_ccw(np.vstack((points, points[0])))):
            points = points[::-1]
        entities.append(BSpline(points=np.arange(len(points)) + len(vertices),
                                knots=knots,
                                closed=closed))
        vertices.extend(points)

    if is_binary_file(file_obj):
        raise ValueError("Binary DXF is unsupported!")

    # in a DXF file, lines come in pairs,
    # a group code then the next line is the value
    # we are removing all whitespace then splitting with the
    # splitlines function which uses the universal newline method
    raw = file_obj.read()
    if hasattr(raw, 'decode'):
        raw = raw.decode('utf-8', errors='ignore')
    raw = raw.upper().replace(' ', '')
    # if this reshape fails, it means the DXF is malformed
    blob = np.array(str.splitlines(raw)).reshape((-1, 2))

    # get the section which contains the header in the DXF file
    endsec = np.nonzero(blob[:, 1] == 'ENDSEC')[0]
    header_start = np.nonzero(blob[:, 1] == 'HEADER')[0][0]
    header_end = endsec[np.searchsorted(endsec, header_start)]
    header_blob = blob[header_start:header_end]

    # get the section which contains entities in the DXF file
    entity_start = np.nonzero(blob[:, 1] == 'ENTITIES')[0][0]
    entity_end = endsec[np.searchsorted(endsec, entity_start)]
    entity_blob = blob[entity_start:entity_end]

    # store metadata pulled from the header of the DXF
    metadata = dict()
    units = get_key(header_blob, '$INSUNITS', '70')
    if units in _DXF_UNITS:
        metadata['units'] = _DXF_UNITS[units]
    else:
        log.warning('DXF doesn\'t have units specified!')

    # find the start points of entities
    # group_check = np.logical_or(entity_blob[:,0] == '0',
    #                            entity_blob[:,0] == '5')
    group_check = entity_blob[:, 0] == '0'
    inflection = np.nonzero(group_check)[0]

    # inflection = np.nonzero(np.logical_and(group_check[:-1],
    # group_check[:-1] == group_check[1:]))[0]
    loaders = {'LINE': (dict, convert_line),
               'LWPOLYLINE': (multi_dict, convert_polyline),
               'ARC': (dict, convert_arc),
               'CIRCLE': (dict, convert_circle),
               'SPLINE': (multi_dict, convert_bspline)}

    vertices = collections.deque()
    entities = collections.deque()
    entity_metadata = collections.defaultdict(collections.deque)

    for chunk in np.array_split(entity_blob, inflection):
        if len(chunk) > 2:
            entity_type = chunk[0][1]
            if entity_type in loaders:
                chunker, loader = loaders[entity_type]
                entity_data = chunker(chunk)
                loader(entity_data)
                update_metadata(entity_data)
            else:
                log.debug('Entity type %s not supported', entity_type)
    metadata.update({k: np.array(v) for k, v in entity_metadata.items()})

    result = {'vertices': np.vstack(vertices).astype(np.float64),
              'entities': np.array(entities),
              'metadata': metadata}

    return result


def export_dxf(path):
    '''
    Export a 2D path object to a DXF file

    Parameters
    ----------
    path: trimesh.path.path.Path2D

    Returns
    ----------
    export: str, path formatted as a DXF file
    '''

    def format_points(points, increment=True):
        '''
        Format points into DXF- style point string.

        Parameters
        -----------
        points: (n,2) or (n,3) float, points in space
        increment: bool, if True increment group code per point
                   Example:
                       [[X0, Y0, Z0], [X1, Y1, Z1]]
                   Result, new lines replaced with spaces:
                     True  -> 10 X0 20 Y0 30 Z0 11 X1 21 Y1 31 Z1
                     False -> 10 X0 20 Y0 30 Z0 10 X1 20 Y1 30 Z1

        Returns
        -----------
        packed: str, points formatted with group code
        '''
        points = np.asanyarray(points)
        three = three_dimensionalize(points, return_2D=False)
        if increment:
            group = np.tile(np.arange(len(three)).reshape((-1, 1)), (1, 3))
        else:
            group = np.zeros((len(three), 3), dtype=np.int)
        group += [10, 20, 30]
        interleaved = np.dstack((group.astype(str),
                                 three.astype(str))).reshape(-1)
        packed = '\n'.join(interleaved)

        return packed

    def entity_info(entity):
        '''
        Pull layer, color, and name information about an entity

        Parameters
        -----------
        entity: entity object

        Returns
        ----------
        name:  str, name of entity
        color: int, color code
        layer: int, layer code
        '''
        color, layer = 0, 0
        if hasattr(entity, 'color'):
            color = int(entity.color)
        if hasattr(entity, 'layer'):
            layer = int(entity.layer)
        name = str(id(entity))[:16]
        return name, color, layer

    def convert_line(line, vertices):
        points = line.discrete(vertices)
        name, color, layer = entity_info(line)
        is_poly = len(points) > 2
        line_type = ['LINE', 'LWPOLYLINE'][int(is_poly)]
        result = templates['line'].substitute({
            'TYPE': line_type,
            'POINTS': format_points(points, increment=not is_poly),
            'NAME': name,
            'LAYER_NUMBER': layer,
            'COLOR_NUMBER': color})
        return result

    def convert_arc(arc, vertices):
        info = arc.center(vertices)
        name, color, layer = entity_info(arc)
        angles = np.degrees(info['angles'])
        arc_type = ['ARC', 'CIRCLE'][int(arc.closed)]
        result = templates['arc'].substitute({
            'TYPE': arc_type,
            'CENTER_POINT': format_points([info['center']]),
            'ANGLE_MIN': angles[0],
            'ANGLE_MAX': angles[1],
            'RADIUS': info['radius'],
            'LAYER_NUMBER': layer,
            'COLOR_NUMBER': color})
        return result

    def convert_bspline(spline, vertices):
        # points formatted with group code
        points = format_points(vertices[spline.points],
                               increment=False)

        # (n,) float knots, formatted with group code
        knots = '40\n' + '\n40\n'.join(spline.knots.reshape(-1).astype(str))

        # pull entity info
        name, color, layer = entity_info(spline)

        # format into string template
        result = templates['bspline'].substitute({
            'TYPE': 'SPLINE',
            'POINTS': points,
            'KNOTS': knots,
            'NAME': name,
            'LAYER_NUMBER': layer,
            'COLOR_NUMBER': color})

        return result

    def convert_generic(entity, vertices):
        '''
        For entities we don't know how to handle, return their
        discrete form as a polyline
        '''
        return convert_line(entity, vertices)

    templates = _templates_dxf
    np.set_printoptions(precision=12)
    conversions = {'Line': convert_line,
                   'Arc': convert_arc,
                   'Bezier': convert_generic,
                   'BSpline': convert_bspline}
    entities_str = ''
    for e in path.entities:
        name = type(e).__name__
        if name in conversions:
            entities_str += conversions[name](e, path.vertices)
        else:
            log.debug('Entity type %s not exported!', name)

    header = templates['header'].substitute(
        {'BOUNDS_MIN': format_points([path.bounds[0]]),
         'BOUNDS_MAX': format_points([path.bounds[1]]),
         'UNITS_CODE': '1'})
    entities = templates['entities'].substitute({'ENTITIES': entities_str})
    footer = templates['footer'].substitute()
    export = '\n'.join([header, entities, footer])
    return export
