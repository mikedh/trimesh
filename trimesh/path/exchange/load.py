import os

from .dxf import _dxf_loaders
from .svg_io import svg_to_path
from ..path import Path

from . import misc
from ... import util


def load_path(obj, file_type=None, **kwargs):
    """
    Load a file to a Path object.

    Parameters
    -----------
    obj : One of the following:
         - Path, Path2D, or Path3D objects
         - open file object (dxf or svg)
         - file name (dxf or svg)
         - shapely.geometry.Polygon
         - shapely.geometry.MultiLineString
         - dict with kwargs for Path constructor
         - (n,2,(2|3)) float, line segments
    file_type : str
        Type of file is required if file
        object passed.

    Returns
    ---------
    path : Path, Path2D, Path3D object
        Data as a native trimesh Path object
    """

    if isinstance(obj, Path):
        # we have been passed a Path object so
        # do nothing and return the passed object
        return obj
    elif util.is_file(obj):
        # for open file objects use loaders
        kwargs.update(path_loaders[file_type](
            obj, file_type=file_type))
    elif util.is_string(obj):
        # strings passed are evaluated as file objects
        with open(obj, 'rb') as file_obj:
            # get the file type from the extension
            file_type = os.path.splitext(obj)[-1][1:].lower()
            # call the loader
            kwargs.update(path_loaders[file_type](
                file_obj, file_type=file_type))
    elif util.is_instance_named(obj, 'Polygon'):
        # convert from shapely polygons to Path2D
        kwargs.update(misc.polygon_to_path(obj))
    elif util.is_instance_named(obj, 'MultiLineString'):
        # convert from shapely LineStrings to Path2D
        kwargs.update(misc.linestrings_to_path(obj))
    elif isinstance(obj, dict):
        # load as kwargs
        from ...exchange.load import load_kwargs
        return load_kwargs(obj)
    elif util.is_sequence(obj):
        # load as lines in space
        kwargs.update(misc.lines_to_path(obj))
    else:
        raise ValueError('Not a supported object type!')

    from ...exchange.load import load_kwargs
    return load_kwargs(kwargs)


def path_formats():
    """
    Get a list of supported path formats.

    Returns
    ------------
    loaders : list of str
        Extensions of loadable formats, ie:
        ['svg', 'dxf']
    """
    return list(path_loaders.keys())


path_loaders = {'svg': svg_to_path}
path_loaders.update(_dxf_loaders)
