import json
import numpy as np

from string import Template

from ..arc import to_threepoint
from ..entities import Line, Arc, BSpline

from ...constants import log
from ...constants import tol_path as tol
from ...resources import get_resource
from ... import util
from ... import grouping

# stuff for DWG loading
import os
import shutil
import tempfile
import subprocess
from distutils.spawn import find_executable


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

# save metadata to a DXF Xrecord starting here
# Valid values are 1-369 (except 5 and 105)
XRECORD_METADATA = 134
# the sentinel string for trimesh metadata
# this should be seen at XRECORD_METADATA
XRECORD_SENTINEL = 'TRIMESH_METADATA:'
# the maximum line length before we split lines
XRECORD_MAX_LINE = 200
# the maximum index of XRECORDS
XRECORD_MAX_INDEX = 368

# get the TEMPLATES for exporting DXF files
TEMPLATES = {k: Template(v) for k, v in json.loads(
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


def load_dxf(file_obj, **kwargs):
    """
    Load a DXF file to a dictionary containing vertices and
    entities.

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
        Pull metadata based on group code, and return as a dict.
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
        """
        Convert DXF LINE entities into trimesh Line entities.
        """
        # create a single Line entity
        entities.append(Line(points=len(vertices) + np.arange(2),
                             **info(e)))
        # add the vertices to our collection
        vertices.extend(np.array([[e['10'], e['20']],
                                  [e['11'], e['21']]],
                                 dtype=np.float64))

    def convert_circle(e):
        """
        Convert DXF CIRCLE entities into trimesh Circle entities
        """
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
        """
        Convert DXF ARC entities into into trimesh Arc entities.
        """
        # the radius of the circle
        R = float(e['40'])
        # the center point of the circle
        C = np.array([e['10'],
                      e['20']], dtype=np.float64)
        # the start and end angle of the arc, in degrees
        # this may depend on an AUNITS header data
        A = np.radians(np.array([e['50'],
                                 e['51']], dtype=np.float64))
        # convert center/radius/angle representation
        # to three points on the arc representation
        points = to_threepoint(center=C[0:2],
                               radius=R,
                               angles=A)
        # add a single Arc entity
        entities.append(Arc(points=len(vertices) + np.arange(3),
                            closed=False,
                            **info(e)))
        # add the three vertices
        vertices.extend(points)

    def convert_polyline(e):
        """
        Convert DXF LWPOLYLINE entities into trimesh Line entities.
        """
        # load the points in the line
        lines = np.column_stack((
            e['10'], e['20'])).astype(np.float64)

        # save entity info so we don't have to recompute
        polyinfo = info(e)

        # 70 is the closed flag for polylines
        # if the closed flag is set make sure to close
        if ('70' in e and int(e['70'][0]) & 1):
            lines = np.vstack((lines, lines[:1]))

        # 42 is the vertex bulge flag for LWPOLYLINE entities
        # "bulge" is autocad for "add a stupid arc using flags
        # in my otherwise normal polygon", it's like SVG arc
        # flags but somehow even more annoying
        if '42' in e:
            # from Autodesk reference:
            # The bulge is the tangent of one fourth the included
            # angle for an arc segment, made negative if the arc
            # goes clockwise from the start point to the endpoint.
            # A bulge of 0 indicates a straight segment, and a
            # bulge of 1 is a semicircle.
            log.debug('polyline bulge: {}'.format(e['42']))
            # what position were vertices stored at
            vid = np.nonzero(chunk[:, 0] == '10')[0]
            # what position were bulges stored at in the chunk
            bid = np.nonzero(chunk[:, 0] == '42')[0]
            # which vertex index is bulge value associated with
            bulge_idx = np.searchsorted(vid, bid)

            # the actual bulge float values
            bulge = np.array(e['42'], dtype=np.float64)
            # use bulge to calculate included angle of the arc
            angle = np.arctan(bulge) * 4.0
            # the indexes making up a bulged segment
            tid = np.column_stack((bulge_idx, bulge_idx - 1))
            # the vector connecting the two ends of the arc
            vector = lines[tid[:, 0]] - lines[tid[:, 1]]
            # the length of the connector segment
            length = (np.linalg.norm(vector, axis=1))

            # perpendicular vectors by crossing vector with Z
            perp = np.cross(
                np.column_stack((vector, np.zeros(len(vector)))),
                np.ones((len(vector), 3)) * [0, 0, 1])
            # strip the zero Z
            perp = util.unitize(perp[:, :2])

            # midpoint of each line
            midpoint = lines[tid].mean(axis=1)

            # calculate the signed radius of each arc segment
            radius = (length / 2.0) / np.sin(angle / 2.0)

            # offset magnitude to point on arc
            offset = radius - np.cos(angle / 2) * radius

            # convert each arc to three points:
            # start, any point on arc, end
            three = np.column_stack((
                lines[tid[:, 0]],
                midpoint + perp * offset.reshape((-1, 1)),
                lines[tid[:, 1]])).reshape((-1, 3, 2))

            # if we're in strict mode make sure our arcs
            # have the same magnitude as the input data
            if tol.strict:
                from .. import arc as arcmod
                check_angle = [arcmod.arc_center(i)['span']
                               for i in three]
                assert np.allclose(np.abs(angle),
                                   np.abs(check_angle))

                check_radii = [arcmod.arc_center(i)['radius']
                               for i in three]
                assert np.allclose(check_radii, np.abs(radius))

            # if there are unconsumed line
            # segments add them to drawing
            if (len(lines) - 1) > len(bulge):
                # indexes of line segments
                existing = util.stack_lines(np.arange(len(lines)))
                # remove line segments replaced with arcs
                for line_idx in grouping.boolean_rows(
                        existing,
                        np.sort(tid, axis=1),
                        np.setdiff1d):
                    # add a single line entity and vertices
                    entities.append(Line(
                        points=np.arange(2) + len(vertices),
                        **polyinfo))
                    vertices.extend(lines[line_idx])
            # add the three point arcs to the result
            for arc_points in three:
                entities.append(Arc(
                    points=np.arange(3) + len(vertices),
                    **polyinfo))
                vertices.extend(arc_points)
            # done with this polyline
            return

        # we have a normal polyline so just add it
        # as single line entity and vertices
        entities.append(Line(
            points=np.arange(len(lines)) + len(vertices),
            **polyinfo))
        vertices.extend(lines)

    def convert_bspline(e):
        """
        Convert DXF Spline entities into trimesh BSpline entities.
        """
        # in the DXF there are n points and n ordered fields
        # with the same group code

        points = np.column_stack((e['10'],
                                  e['20'])).astype(np.float64)
        knots = np.array(e['40']).astype(np.float64)

        # if there are only two points, save it as a line
        if len(points) == 2:
            # create a single Line entity
            entities.append(Line(points=len(vertices) +
                                 np.arange(2),
                                 **info(e)))
            # add the vertices to our collection
            vertices.extend(points)
            return

        # check bit coded flag for closed
        # closed = bool(int(e['70'][0]) & 1)
        # check euclidean distance to see if closed
        closed = np.linalg.norm(points[0] -
                                points[-1]) < tol.merge

        # create a BSpline entity
        entities.append(BSpline(
            points=np.arange(len(points)) + len(vertices),
            knots=knots,
            closed=closed,
            **info(e)))
        # add the vertices
        vertices.extend(points)

    # in a DXF file, lines come in pairs,
    # a group code then the next line is the value
    # we are removing all whitespace then splitting with the
    # splitlines function which uses the universal newline method
    raw = file_obj.read()
    # if we've been passed bytes
    if hasattr(raw, 'decode'):
        # search for the sentinel string indicating binary DXF
        # do it by encoding sentinel to bytes and subset searching
        if raw[:22].find(b'AutoCAD Binary DXF') != -1:
            if _teigha is None:
                # no converter to ASCII DXF available
                raise ValueError('binary DXF not supported!')
            else:
                # convert to R14 ASCII DXF
                raw = _teigha_convert(raw, extension='dxf')
        else:
            # we've been passed bytes that don't have the
            # header for binary DXF so try decoding as UTF-8
            raw = raw.decode('utf-8', errors='ignore')

    # remove spaces and leading/trailing whitespace
    raw = str(raw).replace(' ', '').strip()
    # a version of data in upper case
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

    # store loaded vertices
    vertices = []
    # store loaded entities
    entities = []

    # an old-style polyline entity strings its data across
    # multiple vertex entities like a real asshole
    polyline = None

    # loop through chunks of entity information
    # chunk will be an (n, 2) array of (group code, data) pairs
    for chunk in np.array_split(entity_blob, inflection):
        if len(chunk) >= 1:
            # the string representing entity type
            entity_type = chunk[0][1]

            ############
            # special case old- style polyline entities
            if entity_type == 'POLYLINE':
                polyline = [dict(chunk)]
            # if we are collecting vertex entities
            elif polyline is not None and entity_type == 'VERTEX':
                polyline.append(dict(chunk))
            # the end of a polyline
            elif polyline is not None and entity_type == 'SEQEND':
                # pull the geometry information for the entity
                lines = [[i['10'], i['20']]
                         for i in polyline[1:]]
                # check for a closed flag on the polyline
                if '70' in polyline[0]:
                    # flag is bit- coded integer
                    flag = int(polyline[0]['70'])
                    # first bit represents closed
                    if bool(flag & 1):
                        lines.append(lines[0])
                # create a single Line entity
                entities.append(Line(
                    points=np.arange(len(lines)) + len(vertices),
                    **info(dict(polyline[0]))))
                # add the vertices to our collection
                vertices.extend(lines)
                # we no longer have an active polyline
                polyline = None

            # if the entity contains all relevant data we can
            # cleanly load it from inside a single function
            elif entity_type in loaders:
                # the chunker converts an (n,2) list into a dict
                chunker, loader = loaders[entity_type]
                # convert data to dict
                entity_data = chunker(chunk)
                # append data to the lists we're collecting
                loader(entity_data)
            else:
                log.debug('Entity type %s not supported',
                          entity_type)

    # stack vertices into single array
    vertices = util.vstack_empty(vertices).astype(np.float64)

    # return result as kwargs for trimesh.path.Path2D constructor
    result = {'vertices': vertices,
              'entities': np.array(entities),
              'metadata': metadata}

    return result


def export_dxf(path, include_metadata=False):
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
            group = np.tile(
                np.arange(len(three), dtype=np.int).reshape((-1, 1)),
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

        result = TEMPLATES['line'].substitute(subs)
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

        result = TEMPLATES['arc'].substitute(subs)

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
        result = TEMPLATES['bspline'].substitute(subs)

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
        if (not include_metadata) or len(path.metadata) == 0:
            return ''
        # dump metadata to compact JSON
        # make sure there are no newlines to break DXF
        # util.jsonify will be able to convert numpy arrays
        as_json = util.jsonify(
            path.metadata,
            separators=(',', ':')).replace('\n', ' ')

        # create an XRECORD for our use
        xrecord = TEMPLATES['xrecord'].substitute({
            'INDEX': XRECORD_METADATA,
            'DATA': as_json})
        # add the XRECORD to an objects section
        result = TEMPLATES['objects'].substitute({
            'OBJECTS': xrecord})
        return result

    # make sure we're not losing a ton of
    # precision in the string conversion
    np.set_printoptions(precision=12)
    # trimesh entity to DXF entity converters
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

    # sections of the DXF
    header = TEMPLATES['header'].substitute(hsub)
    # entities section
    entities = TEMPLATES['entities'].substitute({
        'ENTITIES': entities_str})
    footer = TEMPLATES['footer'].substitute()
    # metadata encoded as objects section
    objects = convert_metadata()
    # filter out empty sections
    sections = [i for i in [header,
                            entities,
                            objects,
                            footer]
                if len(i) > 0]
    # append them all into one DXF file
    export = '\n'.join(sections)
    return export


def load_dwg(file_obj, **kwargs):
    """
    Load DWG files by converting them to DXF files using
    TeighaFileConverter.

    Parameters
    -------------
    file_obj : file- like object

    Returns
    -------------
    loaded : dict
        kwargs for a Path2D constructor
    """
    # read the DWG data into a bytes object
    data = file_obj.read()
    # convert data into R14 ASCII DXF
    converted = _teigha_convert(data)
    # load into kwargs for Path2D constructor
    result = load_dxf(util.wrap_as_stream(converted))

    return result


def _teigha_convert(data, extension='dwg'):
    """
    Convert any DXF/DWG to R14 ASCII DXF using Teigha Converter.

    Parameters
    ---------------
    data : str or bytes
       The contents of a DXF or DWG file
    extension : str
       The format of data: 'dwg' or 'dxf'

    Returns
    --------------
    converted : str
       Result as R14 ASCII DXF
    """
    # temp directory for DWG file
    dir_dwg = tempfile.mkdtemp()
    # temp directory for DXF output
    dir_out = tempfile.mkdtemp()

    # put together the subprocess command
    cmd = [_xvfb_run,  # suppress the GUI QT status bar
           '-a',      # use an automatic screen
           _teigha,   # run the converter
           dir_dwg,   # the directory containing DWG files
           dir_out,   # the directory for output DXF files
           'ACAD14',  # the revision of DXF
           'DXF',     # the output format
           '1',       # recurse input folder
           '1']       # audit each file

    # if Xvfb is already running it probably
    # has a working configuration so use it
    running = b'Xvfb' in subprocess.check_output(['ps', '-eaf'])
    # chop off XVFB if it isn't installed or is running
    if running or _xvfb_run is None:
        cmd = cmd[2:]

    # create file in correct mode for data
    if hasattr(data, 'encode'):
        # data is a string which can be encoded to bytes
        mode = 'w'
    else:
        # data is already bytes
        mode = 'wb'

    # write the file_obj in the temp directory
    dwg_name = os.path.join(dir_dwg, 'drawing.' + extension)
    with open(dwg_name, mode) as f:
        f.write(data)

    # run the conversion
    output = subprocess.check_output(cmd)

    # load the ASCII DXF produced from the conversion
    name_result = os.path.join(dir_out, 'drawing.dxf')
    # if the conversion failed log things before failing
    if not os.path.exists(name_result):
        log.error('teigha convert failed!\nls {}: {}\n\n {}'.format(
            dir_out,
            os.listdir(dir_out),
            output))
        raise ValueError('conversion using Teigha failed!')

    # load converted file into a string
    with open(name_result, 'r') as f:
        converted = f.read()

    # remove the temporary directories
    shutil.rmtree(dir_out)
    shutil.rmtree(dir_dwg)

    return converted


# the DWG to DXF converter
for _name in ['ODAFileConverter',
              'TeighaFileConverter']:
    _teigha = find_executable(_name)
    if _teigha is not None:
        break
# suppress X11 output
_xvfb_run = find_executable('xvfb-run')

# store the loaders we have available
_dxf_loaders = {'dxf': load_dxf}
# DWG is only available if teigha converter is installed
if _teigha is not None:
    _dxf_loaders['dwg'] = load_dwg
