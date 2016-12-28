import numpy as np
import collections
import traceback
import os

from .. import util

from ..base import Trimesh
from ..constants import _log_time, log
from ..util import is_file, is_string, make_sequence, is_instance_named

from .assimp import _assimp_loaders
from .stl import _stl_loaders
from .misc import _misc_loaders
from .step import _step_loaders
from .ply import _ply_loaders

try:
    from ..path.io.load import load_path, path_formats
except:
    _path_traceback = traceback.format_exc(4)

    def load_path(*args, **kwargs):
        '''
        Dummy load path function that will raise an exception on use.
        Import of path failed, probably because a dependency is not installed.
        '''
        print(_path_traceback)
        raise ImportError('No path functionality available!')

    def path_formats():
        return []


def mesh_formats():
    return list(mesh_loaders.keys())


def available_formats():
    return np.append(mesh_formats(), path_formats())


def load(file_obj, file_type=None, **kwargs):
    '''
    Load a mesh or vectorized path into a Trimesh, Path2D, or Path3D object.

    Arguments
    ---------
    file_obj: a filename string or a file-like object
    file_type: str representing file type (eg: 'stl')

    Returns
    ---------
    geometry: Trimesh, Path2D, Path3D, or list of same.
    '''
    # check to see if we're trying to load something that is already a Trimesh
    out_types = ('Trimesh', 'Path')
    if any(is_instance_named(file_obj, t) for t in out_types):
        log.info('Loaded called on %s object, returning input',
                 file_obj.__class__.__name__)
        return file_obj

    file_obj, file_type, metadata = _parse_file_args(file_obj, file_type)

    if file_type in path_formats():
        loaded = load_path(file_obj, file_type, **kwargs)
    elif file_type in mesh_formats():
        loaded = load_mesh(file_obj, file_type, **kwargs)
    else:
        raise ValueError('File type: %s not supported', str(file_type))

    for i in util.make_sequence(loaded):
        # check to make sure loader actually loaded something
        # assert any(is_instance_named(i, t) for t in out_types)
        i.metadata.update(metadata)

    return loaded


@_log_time
def load_mesh(file_obj, file_type=None, **kwargs):
    '''
    Load a mesh file into a Trimesh object

    Arguments
    ---------
    file_obj:  str or file-like object
    file_type: str representing file type (eg: 'stl')
    kwargs:    passed to Trimesh constructor

    Returns:
    ----------
    mesh: Trimesh object, or a list of Trimesh objects
          depending on the file format.

    '''
    # turn a string into a file obj and type
    (file_obj, 
     file_type, 
     metadata) = _parse_file_args(file_obj, file_type)

    loaded = mesh_loaders[file_type](file_obj,
                                      file_type)
    if is_file(file_obj):
        file_obj.close()

    log.debug('loaded mesh using %s',
              mesh_loaders[file_type].__name__)

    meshes = collections.deque()
    for mesh_kwargs in make_sequence(loaded):
        mesh_kwargs.update(kwargs)
        mesh = Trimesh(**mesh_kwargs)
        mesh.metadata.update(metadata)
        meshes.append(mesh)

    if len(meshes) == 1:
        return meshes[0]
    return np.array(meshes)


def _parse_file_args(file_obj, file_type):
    '''
    Given a file_obj and a file_type, try to turn them into a file-like object
    and a lowercase string of file type

    Arguments
    -----------
    file_obj:  str: if string represents a file path, returns
                    -------------------------------------------
                    file_obj:   an 'rb' opened file object of the path
                    file_type:  the extension from the file path

               str: if string is NOT a path, but has JSON-like special characters
                    -------------------------------------------
                    file_obj:   the same string passed as file_obj
                    file_type:  set to 'json'

               str: string is not an existing path or a JSON-like object
                    -------------------------------------------
                    ValueError will be raised as we can't do anything with input

               file like object: we cannot grab information on file_type automatically
                    -------------------------------------------
                    ValueError will be raised if file_type is None
                    file_obj:  same as input
                    file_type: same as input

               other object: like a shapely.geometry.Polygon, etc:
                    -------------------------------------------
                    file_obj:  same as input
                    file_type: if None initially, set to the class name
                               (in lower case), otherwise passed through

    file_type: str, type of file and handled according to above

    Returns
    -----------
    file_obj:  loadable object
    file_type: str, lower case of the type of file (eg 'stl', 'dae', etc)
    '''
    metadata = {}

    if util.is_file(file_obj) and file_type is None:
        raise ValueError(
            'File type must be specified when passing file objects!')
    if util.is_string(file_obj):
        try:
            exists = os.path.isfile(file_obj)
        except:
            exists = False
        if exists:
            metadata['file_path'] = file_obj
            metadata['file_name'] = os.path.basename(file_obj)
            # if file_obj is a path that exists use extension as file_type
            file_type = (str(file_obj).split('.')[-1])
            file_obj = open(file_obj, 'rb')
        else:
            if file_type is not None:
                return file_obj, file_type, metadata
            elif '{' in file_obj:
                # if a dict bracket is in the string, its probably a straight
                # JSON
                file_type = 'json'
            else:
                raise ValueError(
                    'File object passed as string that is not a file!')

    if file_type is None:
        file_type = file_obj.__class__.__name__

    if is_string(file_type) and '.' in file_type:
        # if someone has passed the whole filename as the file_type
        # use the file extension as the file_type
        metadata['file_name'] = os.path.basename(file_type)
        file_type = file_type.split('.')[-1]
    file_type = file_type.lower()
    return file_obj, file_type, metadata

mesh_loaders = {}
# assimp has a lot of loaders, but they are all quite slow
# so we load them first and replace them with native loaders if possible
mesh_loaders.update(_assimp_loaders)
mesh_loaders.update(_stl_loaders)
mesh_loaders.update(_misc_loaders)
mesh_loaders.update(_step_loaders)
mesh_loaders.update(_ply_loaders)
