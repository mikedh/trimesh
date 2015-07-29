import numpy as np

from ..constants import log_time, log

from .assimp import _assimp_loaders
from .stl    import _stl_loaders
from .misc   import _misc_loaders

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

    if not hasattr(file_obj, 'read'):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')

    mesh = _mesh_loaders[file_type](file_obj, file_type)
    file_obj.close()
    
    log.debug('loaded mesh using %s',
              _mesh_loaders[file_type].__name__)

    if process: 
        # if mesh is multi-body, process all bodies
        [i.process() for i in np.append(mesh, [])]

    return mesh

_mesh_loaders = {}
_mesh_loaders.update(_assimp_loaders)
_mesh_loaders.update(_stl_loaders)
_mesh_loaders.update(_misc_loaders)

try: 
    from .step import _step_loaders
    _mesh_loaders.update(_step_loaders)
except ImportError:
    pass
