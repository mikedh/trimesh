import numpy as np

from ..constants import *

def load_assimp(file_obj, file_type=None):
    '''
    Use the assimp library to load a mesh, from a file object and type,
    or filename (if file_obj is a string)

    Assimp supports a huge number of mesh formats.

    Performance notes: in tests on binary STL pyassimp was ~10x 
    slower than the native loader included in this package. 
    This is probably due to their recursive prettifying of the data structure.
    
    Also, you need a very recent version of PyAssimp for this function to work 
    (the commit was merged into the assimp github master on roughly 9/5/2014)
    '''

    def LPMesh_to_Trimesh(lp):
        colors = (np.reshape(lp.colors, (-1,4))[:,0:3] * 255).astype(np.int)
        return {'vertices'       : lp.vertices,
                'vertex_normals' : lp.normals,
                'faces'          : lp.faces,
                'vertex_colors'  : colors}

    if not hasattr(file_obj, 'read'):
        # if there is no read attribute, we assume we've been passed a file name
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')

    scene  = pyassimp.load(file_obj, file_type=file_type)
    meshes = list(map(LPMesh_to_Trimesh, scene.meshes))
    pyassimp.release(scene)

    if len(meshes) == 1: 
        return meshes[0]
    return meshes
 
_assimp_loaders = {}
try: 
    import pyassimp

    # this function was added to the master on github on 9/2014
    if hasattr(pyassimp, 'available_formats'):
        _assimp_formats = [i.lower() for i in pyassimp.available_formats()]
    else: 
        log.warning('Older version of assimp detected, using hardcoded format list!')
        _assimp_formats = ['dae', 'blend', '3ds', 'ase',  'obj', 
                           'ifc', 'xgl',   'zgl', 'ply',  'lwo',
                           'lxo', 'x',     'ac',  'ms3d', 'cob', 'scn']
    _assimp_loaders.update(zip(_assimp_formats,
                               [load_assimp]*len(_assimp_formats)))
except ImportError:
    log.warning('No pyassimp, only native loaders available!')
