import numpy as np
import json
import os 
import sys

from .dxf_load   import dxf_to_vector
from .svg_load   import svg_to_path
from .misc       import lines_to_path, polygon_to_lines, dict_to_path

from ..constants import log
from ..path      import Path2D, Path3D

from ...util     import is_sequence

if sys.version_info.major >= 3:
    basestring = str

def load_path(obj, file_type=None):
    '''
    Utility function which can be passed a filename, file object, or list of lines
    '''
    type_name = obj.__class__.__name__
    if hasattr(obj, 'read'):
        loaded = _LOADERS[file_type](obj)
        obj.close()
    elif isinstance(obj, basestring):
        file_obj  = open(obj, 'rb')
        file_type = os.path.splitext(obj)[-1][1:].lower()
        loaded = _LOADERS[file_type](file_obj)
        file_obj.close()
    elif type_name == 'Polygon':
        lines  = polygon_to_lines(obj)
        loaded = lines_to_path(lines)
    elif type_name == 'dict':
        loaded = dict_to_path(obj)
    elif is_sequence(obj):
        loaded = lines_to_path(obj)
    else:
        raise ValueError('Not a supported object type!')
    path = _create_path(**loaded)
    return path

def _create_path(entities, vertices):
    shape = np.shape(vertices)
    if ((len(shape) != 2) or 
        (not shape[1] in [2,3])):
        raise ValueError('Vertices must be 2D or 3D!')
    path = [Path2D, Path3D][shape[1] == 3](entities = entities, 
                                           vertices = vertices)
    return path

def available_formats():
    return _LOADERS.keys()

_LOADERS = {'dxf': dxf_to_vector,
            'svg': svg_to_path}
