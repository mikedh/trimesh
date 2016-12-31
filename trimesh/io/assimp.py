import tempfile
import numpy as np

from ..constants import log


def load_pyassimp(file_obj, file_type=None):
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
        colors = (np.reshape(lp.colors, (-1, 4))[:, 0:3] * 255).astype(np.uint8)
        return {'vertices': lp.vertices,
                'vertex_normals': lp.normals,
                'faces': lp.faces,
                'vertex_colors': colors}

    if not hasattr(file_obj, 'read'):
        # if there is no read attribute, we assume we've been passed a file
        # name
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'rb')

    scene = pyassimp.load(file_obj, file_type=file_type)
    meshes = [LPMesh_to_Trimesh(i) for i in scene.meshes]
    pyassimp.release(scene)

    if len(meshes) == 1:
        return meshes[0]
    return meshes

def load_cyassimp(file_obj, file_type=None):
    '''
    Load a file using the cyassimp bindings.

    The easiest way to install these is with conda:
    conda install -c menpo/label/master cyassimp
    '''

    if hasattr(file_obj, 'read'):
        # if it has a read attribute it is probably a file object
        with tempfile.NamedTemporaryFile(suffix=str(file_type)) as file_temp:
            file_temp.write(file_obj.read)
            scene = cyassimp.AIImporter(file_temp.name)
            scene.build_scene()
    else:
        scene = cyassimp.AIImporter(file_obj.encode('utf-8'))
        scene.build_scene()

    meshes = [{'vertices' : i.points,
               'faces'    : i.trilist} for i in scene.meshes]

    if len(meshes) == 1:
        return meshes[0]
    return meshes

def load_assimp(file_obj, file_type=None):
    try: 
        return load_cyassimp(file_obj=file_obj,
                             file_type=file_type)
    except:
        log.error('Cyassimp had error loading file, trying pyassimp',
                  exc_info=True)
        return load_pyassimp(file_obj=file_obj,
                             file_type=file_type)

_assimp_formats = [
    '.fbx',
    '.dae',
    '.gltf', 
    '.glb',
    '.blend',
    '.3ds',
    '.ase',
    '.obj',
    '.ifc',
    '.xgl',
    '.zgl',
    '.ply',
    '.dxf',
    '.lwo',
    '.lws',
    '.lxo',
    '.stl',
    '.x',
    '.ac',
    '.ms3d',
    '.cob',
    '.scn',
    '.bvh',
    '.csm',
    '.xml',
    '.irrmesh',
    '.irr',
    '.mdl',
    '.md2',
    '.md3',
    '.pk3',
    '.mdc',
    '.md5*',
    '.smd',
    '.vta',
    '.ogex',
    '.3d',
    '.b3d',
    '.q3d',
    '.q3s',
    '.nff',
    '.off',
    '.raw',
    '.ter',
    '3dgs',
    '.mdl',
    '.hmp',
    '.ndo']
_assimp_loaders = {}

try:
    import pyassimp
    if hasattr(pyassimp, 'available_formats'):
        _assimp_formats = [i.lower() for i in pyassimp.available_formats()]
    _assimp_loaders.update(zip(_assimp_formats,
                               [load_pyassimp] * len(_assimp_formats)))
except ImportError:
    pass

try:
    import cyassimp
    _assimp_loaders.update(zip(_assimp_formats,
                               [load_cyassimp] * len(_assimp_formats)))
except ImportError:
    pass
