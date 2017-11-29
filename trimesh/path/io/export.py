import numpy as np
import json
from string import Template

from . import svg_io
    
from ..arc import arc_center
from ...resources import get_resource
from ...util import three_dimensionalize
from ...constants import log
from ...constants import res_path as res

_templates_dxf = {k: Template(v) for k, v in json.loads(
    get_resource('dxf.json.template')).items()}

_template_svg = Template(get_resource('svg.xml.template'))


def export_path(path, file_type, file_obj=None):
    '''
    Export a Path object to a file- like object, or to a filename

    Parameters
    ---------
    file_obj:  a filename string or a file-like object
    file_type: str representing file type (eg: 'svg')
    process:   boolean flag, whether to process the mesh on load

    Returns
    ---------
    mesh: a single Trimesh object, or a list of Trimesh objects,
          depending on the file format.

    '''
    if ((not hasattr(file_obj, 'read')) and
            (file_obj is not None)):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'wb')
    export = _path_exporters[file_type](path)
    return _write_export(export, file_obj)


def export_dict(path):
    export_entities = [e.to_dict() for e in path.entities]
    export_object = {'entities': export_entities,
                     'vertices': path.vertices.tolist()}
    return export_object


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

    def entity_color_layer(entity):
        color, layer = 0, 0
        if hasattr(entity, 'color'):
            color = int(entity.color)
        if hasattr(entity, 'layer'):
            layer = int(entity.layer)
        return color, layer

    def convert_line(line, vertices):
        points = line.discrete(vertices)
        color, layer = entity_color_layer(line)
        is_poly = len(points) > 2
        line_type = ['LINE', 'LWPOLYLINE'][int(is_poly)]
        result = templates['line'].substitute(
            {
                'TYPE': line_type, 'POINTS': format_points(
                    points, increment=not is_poly), 'NAME': str(
                    id(line))[
                    :16], 'LAYER_NUMBER': layer, 'COLOR_NUMBER': color})
        return result

    def convert_arc(arc, vertices):
        info = arc.center(vertices)
        color, layer = entity_color_layer(arc)
        angles = np.degrees(info['angles'])
        arc_type = ['ARC', 'CIRCLE'][int(arc.closed)]
        result = templates['arc'].substitute({'TYPE': arc_type,
                                              'CENTER_POINT': format_points([info['center']]),
                                              'ANGLE_MIN': angles[0],
                                              'ANGLE_MAX': angles[1],
                                              'RADIUS': info['radius'],
                                              'LAYER_NUMBER': layer,
                                              'COLOR_NUMBER': color})
        return result

    def convert_generic(entity, vertices):
        return convert_line(entity, vertices)

    templates = _templates_dxf
    np.set_printoptions(precision=12)
    conversions = {'Line': convert_line,
                   'Arc': convert_arc,
                   'Bezier': convert_generic,
                   'BSpline': convert_generic}
    entities_str = ''
    for e in path.entities:
        name = type(e).__name__
        if name in conversions:
            entities_str += conversions[name](e, path.vertices)
        else:
            log.debug('Entity type %s not exported!', name)

    header = templates['header'].substitute({'BOUNDS_MIN': format_points([path.bounds[0]]),
                                             'BOUNDS_MAX': format_points([path.bounds[1]]),
                                             'UNITS_CODE': '1'})
    entities = templates['entities'].substitute({'ENTITIES': entities_str})
    footer = templates['footer'].substitute()
    export = '\n'.join([header, entities, footer])
    return export


def _write_export(export, file_obj=None):
    '''
    Write a string to a file.
    If file_obj isn't specified, return the string

    Parameters
    ---------
    export: a string of the export data
    file_obj: a file-like object or a filename
    '''

    if file_obj is None:
        return export
    elif hasattr(file_obj, 'write'):
        out_file = file_obj
    else:
        out_file = open(file_obj, 'wb')
    try:
        out_file.write(export)
    except TypeError:
        out_file.write(export.encode('utf-8'))
    out_file.close()
    return export


_path_exporters = {'dxf': export_dxf,
                   'svg': svg_io.export_svg,
                   'dict': export_dict}
