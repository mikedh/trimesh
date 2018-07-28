import json
import numpy as np

from string import Template

from ..arc import to_threepoint
from ..entities import Line, Arc, BSpline
from ..util import is_ccw

from ...constants import log
from ...constants import tol_path as tol
from ...resources import get_resource
from ... import util

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
# backwards, for reference
_UNITS_TO_DXF = {v: k for k, v in _DXF_UNITS.items()}

# save metadata to a DXF Xrecord
# Valid values are 1-369 (except 5 and 105)
XRECORD_METADATA = 134

# get the templates for exporting DXF files
_TEMPLATES = {k: Template(v) for k, v in json.loads(
    get_resource('dxf.json.template')).items()}

def get_key(blob, field, code):
    try:
        line = blob[np.nonzero(blob[:, 1] == field)[0][0] + 1]
    except IndexError:
        return None
    if line[0] == code:
        try:
            return int(line[1])
        except ValueError:
            return line[1]
    else:
        return None


def load_dxf(file_obj):
    """
    Load a DXF file to a dictionary containing vertices and entities.

    Parameters
    ----------
    file_obj: file or file- like object (has object.read method)

    Returns
    ----------
    result: dict, keys are  entities, vertices and metadata
    """

    def get_metadata():
        """
        Get metadata from DXF objects section of the file.

        Returns
        ----------
        metadata : dict
           Any available metadata stored in XRecord
        """
        metadata = {}
        # save file version info
        metadata['ACADVER'] = get_key(header_blob, '$ACADVER', '1')

        # get the section which contains objects in the DXF file
        obj_start = np.nonzero(blob[:, 1] == 'OBJECTS')[0]

        if len(obj_start) == 0:
            return metadata
        obj_start = obj_start[0]

        obj_end = endsec[np.searchsorted(endsec, obj_start)]
        obj_blob = blob[obj_start:obj_end]

        # the index of xrecords are one past the group code key
        xrecords = np.nonzero((
            obj_blob == ['100', 'ACDBXRECORD']).all(axis=1))[0] + 1

        # if there are no XRecords return
        if len(xrecords) == 0:
            return metadata

        # resplit the file data without upper() to preserve case
        blob_lower = np.array(str.splitlines(raw)).reshape((-1, 2))
        # newlines and split should never be effected by upper()
        assert len(blob_lower) == len(blob)

        # the likely exceptions are related to JSON decoding
        try:
            # loop through xrecords by group code
            for code, data in blob_lower[obj_start:obj_end][xrecords]:
                if code == str(XRECORD_METADATA):
                    metadata.update(json.loads(data))
                # we could store xrecords in the else here
                # but they have a lot of garbage so don't
                # metadata['XRECORD_' + code] = data
        except BaseException:
            log.error('failed to load metadata!', exc_info=True)

        return metadata

    def info(e):
        """
        Pull metadata based on group code
        """
        # which keys should we extract from the entity data
        # DXF group code : our metadata key
        get = {'8': 'layer'}

        # replace group codes with names and only
        # take info from the entity dict if it is in cand
        renamed = {get[k]: util.make_sequence(v)[0] for k,
                   v in e.items() if k in get}

        return renamed

    def convert_line(e):
        entities.append(Line(points=len(vertices) + np.arange(2),
                             **info(e)))
        vertices.extend(np.array([[e['10'], e['20']],
                                  [e['11'], e['21']]],
                                 dtype=np.float64))

    def convert_circle(e):
        R = float(e['40'])
        C = np.array([e['10'],
                      e['20']]).astype(np.float64)
        points = to_threepoint(center=C[0:2],
                               radius=R)
        entities.append(Arc(points=(len(vertices) + np.arange(3)),
                            closed=True,
                            **info(e)))
        vertices.extend(points)

    def convert_arc(e):
        R = float(e['40'])
        C = np.array([e['10'],
                      e['20']], dtype=np.float64)
        A = np.radians(np.array([e['50'],
                                 e['51']], dtype=np.float64))
        points = to_threepoint(center=C[0:2],
                               radius=R,
                               angles=A)
        entities.append(Arc(points=len(vertices) + np.arange(3),
                            closed=False,
                            **info(e)))
        vertices.extend(points)

    def convert_polyline(e):
        lines = np.column_stack((e['10'], e['20'])).astype(np.float64)

        # 70 is the closed flag for polylines
        # if the closed flag is set make sure to close
        if ('70' in e and int(e['70'][0]) == 1):
            lines = np.vstack((lines, lines[:1]))

        # 42 is the bulge flag for polylines
        # "bulge" is autocad for "add a stupid arc using implicit flags
        # in my otherwise normal polygon"
        if '42' in e:
            log.warning('polyline with bulge %s detected, ignoring!',
                        e['42'])

        entities.append(Line(points=np.arange(len(lines)) + len(vertices),
                             **info(e)))
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
                                closed=closed,
                                **info(e)))
        vertices.extend(points)

    # in a DXF file, lines come in pairs,
    # a group code then the next line is the value
    # we are removing all whitespace then splitting with the
    # splitlines function which uses the universal newline method
    raw = file_obj.read()
    # if we've been passed bytes
    if hasattr(raw, 'decode'):
        # search for the sentinal string indicating binary DXF
        # do it by encoding sentinel to bytes and subset searching
        if raw[:100].find('AutoCAD Binary DXF'.encode('utf-8')) != -1:
            raise ValueError('binary DXF not supported!')
        raw = raw.decode('utf-8', errors='ignore')

    raw = str(raw).replace(' ', '').strip()
    raw_upper = raw.upper()
    # if this reshape fails, it means the DXF is malformed
    blob = np.array(str.splitlines(raw_upper)).reshape((-1, 2))

    # get the section which contains the header in the DXF file
    endsec = np.nonzero(blob[:, 1] == 'ENDSEC')[0]
    header_start = np.nonzero(blob[:, 1] == 'HEADER')[0][0]
    header_end = endsec[np.searchsorted(endsec, header_start)]
    header_blob = blob[header_start:header_end]

    # get the section which contains entities in the DXF file
    entity_start = np.nonzero(blob[:, 1] == 'ENTITIES')[0][0]
    entity_end = endsec[np.searchsorted(endsec, entity_start)]
    entity_blob = blob[entity_start:entity_end]

    # try to load path metadata from xrecords stored in DXF
    try:
        metadata = get_metadata()
    except BaseException:
        log.error('failed to extract metadata!',
                  exc_info=True)
        metadata = {}

    # store unit data pulled from the header of the DXF
    # prefer LUNITS over INSUNITS
    # I couldn't find a table for LUNITS values but they
    # look like they are 0- indexed versions of
    # the INSUNITS keys, so for now offset the key value
    for offset, key in [(0, '$INSUNITS'),
                        (-1, '$LUNITS')]:
        # get the key from the header blob
        units = get_key(header_blob, key, '70')
        # if it exists add the offset
        if units is not None:
            units += offset
        # if the key is in our list of units store it
        if units in _DXF_UNITS:
            metadata['units'] = _DXF_UNITS[units]
    # warn on drawings with no units
    if 'units' not in metadata:
        log.warning('DXF doesn\'t have units specified!')

    # find the start points of entities
    group_check = entity_blob[:, 0] == '0'
    inflection = np.nonzero(group_check)[0]

    # DXF object to trimesh object converters
    loaders = {'LINE': (dict, convert_line),
               'LWPOLYLINE': (util.multi_dict, convert_polyline),
               'ARC': (dict, convert_arc),
               'CIRCLE': (dict, convert_circle),
               'SPLINE': (util.multi_dict, convert_bspline)}

    # store loaded objects
    vertices = []
    entities = []

    # loop through chunks of entity information
    for chunk in np.array_split(entity_blob, inflection):
        if len(chunk) > 2:
            entity_type = chunk[0][1]
            if entity_type in loaders:
                chunker, loader = loaders[entity_type]
                entity_data = chunker(chunk)
                loader(entity_data)
            else:
                log.debug('Entity type %s not supported', entity_type)

    # return result as kwargs for trimesh.path.Path2D constructor
    result = {'vertices': util.vstack_empty(vertices).astype(np.float64),
              'entities': np.array(entities),
              'metadata': metadata}

    return result


def export_dxf(path):
    """
    Export a 2D path object to a DXF file

    Parameters
    ----------
    path: trimesh.path.path.Path2D

    Returns
    ----------
    export: str, path formatted as a DXF file
    """

    def format_points(points,
                      as_2D=False,
                      increment=True):
        """
        Format points into DXF- style point string.

        Parameters
        -----------
        points:    (n,2) or (n,3) float, points in space
        as_2D:     bool, if True only output 2 points per vertex
        increment: bool, if True increment group code per point
                   Example:
                       [[X0, Y0, Z0], [X1, Y1, Z1]]
                   Result, new lines replaced with spaces:
                     True  -> 10 X0 20 Y0 30 Z0 11 X1 21 Y1 31 Z1
                     False -> 10 X0 20 Y0 30 Z0 10 X1 20 Y1 30 Z1

        Returns
        -----------
        packed: str, points formatted with group code
        """
        points = np.asanyarray(points, dtype=np.float64)
        three = util.three_dimensionalize(points, return_2D=False)
        if increment:
            group = np.tile(np.arange(len(three),
                                      dtype=np.int).reshape((-1, 1)),
                            (1, 3))
        else:
            group = np.zeros((len(three), 3), dtype=np.int)
        group += [10, 20, 30]

        if as_2D:
            group = group[:, :2]
            three = three[:, :2]

        packed = '\n'.join('{:d}\n{:.12f}'.format(g, v)
                           for g, v in zip(group.reshape(-1),
                                           three.reshape(-1)))

        return packed

    def entity_info(entity):
        """
        Pull layer, color, and name information about an entity

        Parameters
        -----------
        entity: entity object

        Returns
        ----------
        subs: dict, with keys 'COLOR', 'LAYER', 'NAME'
        """
        subs = {'COLOR': 255,  # default is ByLayer
                'LAYER': 0,
                'NAME': str(id(entity))[:16]}

        if hasattr(entity, 'color'):
            # all colors must be integers between 0-255
            color = str(entity.color)
            if str.isnumeric(color):
                subs['COLOR'] = int(color) % 256

        if hasattr(entity, 'layer'):
            subs['LAYER'] = str(entity.layer)

        return subs

    def convert_line(line, vertices):
        points = line.discrete(vertices)

        subs = entity_info(line)
        subs['POINTS'] = format_points(points,
                                       as_2D=True,
                                       increment=False)
        subs['TYPE'] = 'LWPOLYLINE'
        subs['VCOUNT'] = len(points)
        # 1 is closed
        # 0 is default (open)
        subs['FLAG'] = int(bool(line.closed))

        result = templates['line'].substitute(subs)
        return result

    def convert_arc(arc, vertices):
        info = arc.center(vertices)
        subs = entity_info(arc)

        center = info['center']
        if len(center) == 2:
            center = np.append(center, 0.0)
        data = '10\n{:.12f}\n20\n{:.12f}\n30\n{:.12f}'.format(*center)
        data += '\n40\n{:.12f}'.format(info['radius'])

        if arc.closed:
            subs['TYPE'] = 'CIRCLE'
        else:
            subs['TYPE'] = 'ARC'
            # an arc is the same as a circle, with an added start
            # and end angle field
            data += '\n100\nAcDbArc'
            data += '\n50\n{:.12f}\n51\n{:.12f}'.format(
                *np.degrees(info['angles']))
        subs['DATA'] = data

        result = templates['arc'].substitute(subs)

        return result

    def convert_bspline(spline, vertices):
        # points formatted with group code
        points = format_points(vertices[spline.points],
                               increment=False)

        # (n,) float knots, formatted with group code
        knots = ('40\n{:.12f}\n' * len(spline.knots)
                 ).format(*spline.knots)[:-1]

        # bit coded
        flags = {'closed': 1,
                 'periodic': 2,
                 'rational': 4,
                 'planar': 8,
                 'linear': 16}

        flag = flags['planar']
        if spline.closed:
            flag = flag | flags['closed']

        normal = [0.0, 0.0, 1.0]
        n_code = [210, 220, 230]
        n_str = '\n'.join('{:d}\n{:.12f}'.format(i, j)
                          for i, j in zip(n_code, normal))

        subs = entity_info(spline)
        subs.update({'TYPE': 'SPLINE',
                     'POINTS': points,
                     'KNOTS': knots,
                     'NORMAL': n_str,
                     'DEGREE': 3,
                     'FLAG': flag,
                     'FCOUNT': 0,
                     'KCOUNT': len(spline.knots),
                     'PCOUNT': len(spline.points)})
        # format into string template
        result = templates['bspline'].substitute(subs)

        return result

    def convert_generic(entity, vertices):
        """
        For entities we don't know how to handle, return their
        discrete form as a polyline
        """
        return convert_line(entity, vertices)

    def convert_metadata():
        """
        Save path metadata as a DXF Xrecord object.
        """
        if len(path.metadata) == 0:
            return ''
        # dump metadata to compact JSON and make sure there are no newlines
        as_json = util.jsonify(path.metadata,
                               separators=(',', ':')).replace('\n', ' ')
        xrecord = templates['xrecord'].substitute({'INDEX': XRECORD_METADATA,
                                                   'DATA': as_json})
        result = templates['objects'].substitute({'OBJECTS': xrecord})
        return result

    templates = _TEMPLATES
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

    hsub = {'BOUNDS_MIN': format_points([path.bounds[0]]),
            'BOUNDS_MAX': format_points([path.bounds[1]]),
            'LUNITS': '1'}
    if path.units in _UNITS_TO_DXF:
        hsub['LUNITS'] = _UNITS_TO_DXF[path.units]

    header = templates['header'].substitute(hsub)
    entities = templates['entities'].substitute({'ENTITIES': entities_str})
    footer = templates['footer'].substitute()
    objects = convert_metadata()
    export = '\n'.join([header, entities, objects, footer])
    return export
