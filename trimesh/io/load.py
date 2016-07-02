import numpy as np
import os

from .. import util

from ..base      import Trimesh
from ..constants import _log_time, log
from ..util      import is_file, is_string, make_sequence, is_instance_named

from .assimp import _assimp_loaders
from .stl    import _stl_loaders
from .misc   import _misc_loaders
from .step   import _step_loaders
from .ply    import _ply_loaders

try:
    from ..path.io.load import load_path, path_formats
except:
    log.warning('No path functionality available!', exc_info=True)
    def load_path(*args, **kwargs):
        raise ImportError('No path functionality available!')
    def path_formats():
        return []

def mesh_formats():
    return list(_mesh_loaders.keys())

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
    # check to make sure we have a type specified if a file object is passed
    if (util.is_file(file_obj) and 
        file_type is None):
        raise ValueError('If file object is passed file_type must be specified!')

    # check to see if we're trying to load something that is already a Trimesh
    out_types = ('Trimesh', 'Path')
    if any(is_instance_named(file_obj, t) for t in out_types):
        log.info('Loaded called on %s object, returning input',
                 file_obj.__class__.__name__)
        return file_obj

    if is_string(file_obj):
        # if file_obj is a path that exists use extension as file_type
        if os.path.isfile(file_obj):
            file_type = (str(file_obj).split('.')[-1])
            file_obj = open(file_obj, 'rb')
        # if a dict bracket is in the string, its probably a straight JSON
        elif '{' in file_obj:
            file_type = 'json'
        else:
            raise ValueError('File object passed as string that is not a file!')
            
    if file_type is None:
        file_type = file_obj.__class__.__name__

    # if someone has passed the whole filename as the file_type
    # use the file extension as the file_type
    if is_string(file_type) and '.' in file_type:
        file_type = file_type.split('.')[-1]
    file_type = file_type.lower()

    if file_type in path_formats():
        loaded = load_path(file_obj, file_type, **kwargs)
    elif file_type in mesh_formats():
        loaded = load_mesh(file_obj, file_type, **kwargs)
    else:
        raise ValueError('File type: %s not supported', str(file_type))

    for i in util.make_sequence(loaded):
        # check to make sure loader actually loaded something
        assert any(is_instance_named(i, t) for t in out_types)
    return loaded

@_log_time
def load_mesh(file_obj, file_type=None):
    '''
    Load a mesh file into a Trimesh object

    Arguments
    ---------
    file_obj: a filename string or a file-like object
    file_type: str representing file type (eg: 'stl')
 
    Returns:
    ----------
    mesh: a single Trimesh object, or a list of Trimesh objects, 
          depending on the file format. 
    
    '''    

    if is_string(file_obj):
        # if file_obj is a path that exists use extension as file_type
        if os.path.isfile(file_obj):
            file_type = (str(file_obj).split('.')[-1])
            file_obj = open(file_obj, 'rb')
        else:
            raise ValueError('File does not exist!')
    file_type = file_type.lower()

    loaded = _mesh_loaders[file_type](file_obj, 
                                      file_type)
    if is_file(file_obj): 
        file_obj.close()
    
    log.debug('loaded mesh using %s',
              _mesh_loaders[file_type].__name__)

    meshes = [Trimesh(**i) for i in make_sequence(loaded)]
    if len(meshes) == 1: 
        return meshes[0]
    return meshes

_mesh_loaders = {}
# assimp has a lot of loaders, but they are all quite slow
# so we load them first and replace them with native loaders if possible
_mesh_loaders.update(_assimp_loaders)
_mesh_loaders.update(_stl_loaders)
_mesh_loaders.update(_misc_loaders)
_mesh_loaders.update(_step_loaders)
_mesh_loaders.update(_ply_loaders)
