import numpy as np
import os

from .dxf_load import load_dxf
from .svg_io import svg_to_path
from .misc import lines_to_path, polygon_to_path, dict_to_path
from ..path import Path, Path2D, Path3D
from ...util import is_sequence, is_file, is_string, is_instance_named


def load_path(obj, file_type=None):
    '''
    Utility function which can be passed a filename, 
    file object, or list of lines

    Parameters
    -----------
    obj: one of the following:
         - Path, Path2D, or Path3D objects
         - open file object
         - file name
         - shapely Polygon
         - dict with kwargs for Path constructor
         - (n,2,(2|3) float, lines in space

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
    elif is_file(obj):
        loaded = _LOADERS[file_type](obj)
        obj.close()
    elif is_string(obj):
        file_obj = open(obj, 'rb')
        file_type = os.path.splitext(obj)[-1][1:].lower()
        loaded = _LOADERS[file_type](file_obj)
        file_obj.close()
    elif is_instance_named(obj, 'Polygon'):
        loaded = polygon_to_path(obj)
    elif is_instance_named(obj, 'dict'):
        loaded = dict_to_path(obj)
    elif is_sequence(obj):
        loaded = lines_to_path(obj)
    else:
        raise ValueError('Not a supported object type!')
    path = _create_path(**loaded)
    return path


def _create_path(entities, vertices, metadata=None, **kwargs):
    shape = np.shape(vertices)
    if ((len(shape) != 2) or
            (not shape[1] in [2, 3])):
        raise ValueError('Vertices must be 2D or 3D!')
    path = [Path2D, Path3D][shape[1] == 3](entities=entities,
                                           vertices=vertices,
                                           metadata=metadata,
                                           **kwargs)
    return path


def path_formats():
    return list(_LOADERS.keys())


_LOADERS = {'dxf': load_dxf,
            'svg': svg_to_path}
