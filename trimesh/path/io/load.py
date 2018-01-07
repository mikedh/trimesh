import numpy as np
import os

from .dxf import load_dxf
from .svg_io import svg_to_path
from ..path import Path, Path2D, Path3D

from . import misc
from ... import util


def load_path(obj, file_type=None):
    '''
    Load a file to a Path object.

    Parameters
    -----------
    obj: one of the following:
         - Path, Path2D, or Path3D objects
         - open file object
         - file name
         - shapely.geometry.Polygon
         - shapely.geometry.MultiLineString
         - dict with kwargs for Path constructor
         - (n,2,(2|3)) float, line segments

    file_type: str, type of file is required if file
               object passed. Currently supported:
               - 'dxf'
               - 'svg'

    Returns
    ---------
    path: Path2D or Path3D object
    '''

    if isinstance(obj, Path):
        return obj
    elif util.is_file(obj):
        loaded = _LOADERS[file_type](obj)
        obj.close()
    elif util.is_string(obj):
        with open(obj, 'rb') as file_obj:
            file_type = os.path.splitext(obj)[-1][1:].lower()
            loaded = _LOADERS[file_type](file_obj)
    elif util.is_instance_named(obj, 'Polygon'):
        loaded = misc.polygon_to_path(obj)
    elif util.is_instance_named(obj, 'MultiLineString'):
        loaded = misc.linestrings_to_path(obj)
    elif util.is_instance_named(obj, 'dict'):
        loaded = misc.dict_to_path(obj)
    elif util.is_sequence(obj):
        loaded = misc.lines_to_path(obj)
    else:
        raise ValueError('Not a supported object type!')
    path = _create_path(**loaded)
    return path


def _create_path(entities, vertices, metadata=None, **kwargs):
    vertices = np.asanyarray(vertices)

    if len(vertices.shape) != 2:
        path_type = Path
    else:
        path_type = [Path2D, Path3D][int(vertices.shape[1] == 3)]

    return path_type(entities=entities,
                     vertices=vertices,
                     metadata=metadata,
                     **kwargs)


def path_formats():
    return list(_LOADERS.keys())


_LOADERS = {'dxf': load_dxf,
            'svg': svg_to_path}
