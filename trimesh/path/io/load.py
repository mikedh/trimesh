import numpy as np
import os

from .dxf import _dxf_loaders
from .svg_io import svg_to_path
from ..path import Path, Path2D, Path3D

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
        loaded = path_loaders[file_type](obj,
                                         file_type=file_type)
        obj.close()
    elif util.is_string(obj):
        # strings passed are evaluated as file objects
        with open(obj, 'rb') as file_obj:
            # get the file type from the extension
            file_type = os.path.splitext(obj)[-1][1:].lower()
            # call the loader
            loaded = path_loaders[file_type](file_obj,
                                             file_type=file_type)
    elif util.is_instance_named(obj, 'Polygon'):
        # convert from shapely polygons to Path2D
        loaded = misc.polygon_to_path(obj)
    elif util.is_instance_named(obj, 'MultiLineString'):
        # convert from shapely LineStrings to Path2D
        loaded = misc.linestrings_to_path(obj)
    elif util.is_instance_named(obj, 'dict'):
        # load as kwargs
        loaded = misc.dict_to_path(obj)
    elif util.is_sequence(obj):
        # load as lines in space
        loaded = misc.lines_to_path(obj)
    else:
        raise ValueError('Not a supported object type!')

    # pass kwargs through to path loader
    kwargs.update(loaded)
    # convert the kwargs to a Path2D or Path3D object
    path = _create_path(**kwargs)

    return path


def _create_path(entities,
                 vertices,
                 metadata=None,
                 **kwargs):
    """
    Turn entities and vertices into a Path2D or a Path3D
    object depending on dimension of vertices.

    Parameters
    -----------
    entities : list
        Entity objects that reference vertex indices
    vertices : (n, 2) or (n, 3) float
        Vertices in space
    metadata : dict
        Any metadata about the path object

    Returns
    -----------
    as_path : Path2D or Path3D object
        Args in native trimesh object form

    """
    # make sure vertices are numpy array
    vertices = np.asanyarray(vertices,
                             dtype=np.float64)

    if len(vertices.shape) != 2:
        raise ValueError(
            'vertices must be (n, dimension), not {}'.format(
                vertices.shape))

    # check dimension of vertices to decide on object type
    if vertices.shape[1] == 2:
        path_type = Path2D
    elif vertices.shape[1] == 3:
        path_type = Path3D
    else:
        # weird or empty vertices, just use default Path object
        path_type = Path

    # create the object
    as_path = path_type(entities=entities,
                        vertices=vertices,
                        metadata=metadata,
                        **kwargs)
    return as_path


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
