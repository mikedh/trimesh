import copy
import tempfile

import numpy as np

from .. import util


def load_pyassimp(file_obj,
                  file_type=None,
                  resolver=None,
                  **kwargs):
    """
    Use the pyassimp library to load a mesh from a file object
    and type or file name if file_obj is a string

    Parameters
    ---------
    file_obj: str, or file object
      File path or object containing mesh data
    file_type : str
      File extension, aka 'stl'
    resolver : trimesh.visual.resolvers.Resolver
      Used to load referenced data (like texture files)
    kwargs : dict
      Passed through to mesh constructor

    Returns
    ---------
    meshes : (n,) list of dict
      Contain kwargs for Trimesh constructor
    """

    def LPMesh_to_Trimesh(lp):
        # Try to get the vertex colors attribute
        colors = (np.reshape(lp.colors, (-1, 4))
                  [:, :3] * 255).round().astype(np.uint8)
        # If no vertex colors, try to extract them from the material
        if len(colors) == 0:
            if 'diffuse' in lp.material.properties.keys():
                colors = np.array(lp.material.properties['diffuse'])

        # pass kwargs through to mesh constructor
        mesh_kwargs = copy.deepcopy(kwargs)
        # add data from the LP_Mesh
        mesh_kwargs.update({'vertices': lp.vertices,
                            'vertex_normals': lp.normals,
                            'faces': lp.faces,
                            'vertex_colors': colors})
        return mesh_kwargs

    opened = False
    if not hasattr(file_obj, 'read'):
        # if there is no read attribute
        # we assume we've been passed a file name
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'rb')
        opened = True
    elif not hasattr(file_obj, 'mode') or file_obj.mode != 'rb':
        # assimp will crash on anything that isn't binary
        # so if we have a text mode file or anything else
        # grab the data, encode as bytes, and then use stream
        data = file_obj.read()
        if hasattr(data, 'encode'):
            data = data.encode('utf-8')
        file_obj = util.wrap_as_stream(data)

    # load the scene
    scene = pyassimp.load(file_obj,
                          file_type=file_type)
    meshes = [LPMesh_to_Trimesh(i) for i in scene.meshes]

    # release the loaded mesh
    pyassimp.release(scene)

    # if we opened the file in this function close it
    if opened:
        file_obj.close()

    return meshes


def load_cyassimp(file_obj,
                  file_type=None,
                  resolver=None,
                  **kwargs):
    """
    Load a file using the cyassimp bindings.

    The easiest way to install these is with conda:
    conda install -c menpo/label/master cyassimp

    Parameters
    ---------
    file_obj: str, or file object
      File path or object containing mesh data
    file_type : str
      File extension, aka 'stl'
    resolver : trimesh.visual.resolvers.Resolver
      Used to load referenced data (like texture files)
    kwargs : dict
      Passed through to mesh constructor

    Returns
    ---------
    meshes : (n,) list of dict
      Contain kwargs for Trimesh constructor
    """

    if hasattr(file_obj, 'read'):
        # if it has a read attribute it is probably a file object
        with tempfile.NamedTemporaryFile(
                suffix=str(file_type)) as file_temp:

            file_temp.write(file_obj.read())
            # file name should be bytes
            scene = cyassimp.AIImporter(
                file_temp.name.encode('utf-8'))
            scene.build_scene()
    else:
        scene = cyassimp.AIImporter(file_obj.encode('utf-8'))
        scene.build_scene()

    meshes = []
    for m in scene.meshes:
        mesh_kwargs = kwargs.copy()
        mesh_kwargs.update({'vertices': i.points,
                            'faces': i.trilist})
        meshes.append(mesh_kwargs)

    if len(meshes) == 1:
        return meshes[0]
    return meshes


_assimp_formats = [
    'fbx',
    'dae',
    'gltf',
    'glb',
    'blend',
    '3ds',
    'ase',
    'obj',
    'ifc',
    'xgl',
    'zgl',
    'ply',
    'lwo',
    'lws',
    'lxo',
    'stl',
    'x',
    'ac',
    'ms3d',
    'cob',
    'scn',
    'bvh',
    'csm',
    'xml',
    'irrmesh',
    'irr',
    'mdl',
    'md2',
    'md3',
    'pk3',
    'mdc',
    'md5',
    'smd',
    'vta',
    'ogex',
    '3d',
    'b3d',
    'q3d',
    '.q3s',
    'nff',
    'off',
    'raw',
    'ter',
    '3dgs',
    'mdl',
    'hmp',
    'ndo']
_assimp_loaders = {}


# try importing both assimp bindings but prefer cyassimp
loader = None
try:
    import pyassimp
    loader = load_pyassimp
except ImportError:
    pass

try:
    import cyassimp
    loader = load_cyassimp
except ImportError:
    pass


if loader:
    _assimp_loaders.update(zip(_assimp_formats,
                               [loader] * len(_assimp_formats)))
