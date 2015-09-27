import numpy as np

from ..base      import Trimesh

from ..constants import log_time, log
from ..util      import is_file, is_string, make_sequence

from .assimp import _assimp_loaders
from .stl    import _stl_loaders
from .misc   import _misc_loaders
from .step   import _step_loaders

def available_formats():
    return _mesh_loaders.keys()

@log_time
def load_mesh(file_obj, file_type=None, process=True):
    '''
    Load a mesh file into a Trimesh object

    Arguments
    ---------
    file_obj: a filename string or a file-like object
    file_type: str representing file type (eg: 'stl')
    process:   boolean flag, whether to process the mesh on load

    Returns:
    mesh: a single Trimesh object, or a list of Trimesh objects, 
          depending on the file format. 
    
    '''

    if is_string(file_obj):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')
        
    file_type = str(file_type).lower()
    
    loaded = _mesh_loaders[file_type](file_obj, file_type)
    file_obj.close()
    
    log.debug('loaded mesh using %s',
              _mesh_loaders[file_type].__name__)

    meshes = [Trimesh(process=process, **i) for i in make_sequence(loaded)]
    
    if len(meshes) == 1: return meshes[0]
    return meshes

_mesh_loaders = {}
_mesh_loaders.update(_assimp_loaders)
_mesh_loaders.update(_stl_loaders)
_mesh_loaders.update(_misc_loaders)
_mesh_loaders.update(_step_loaders)
