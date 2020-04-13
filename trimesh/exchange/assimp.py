import copy
import tempfile
import collections

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
    scene : trimesh.Scene
      Native trimesh copy of assimp scene
    """

    def LP_to_TM(lp):
        # try to get the vertex colors attribute
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
                            'vertex_colors': colors,
                            'faces': lp.faces})

        return mesh_kwargs

    # did we open the file inside this function
    opened = False
    # not a file object
    if not hasattr(file_obj, 'read'):
        # if there is no read attribute
        # we assume we've been passed a file name
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'rb')
        opened = True
    # we need files to be bytes
    elif not hasattr(file_obj, 'mode') or file_obj.mode != 'rb':
        # assimp will crash on anything that isn't binary
        # so if we have a text mode file or anything else
        # grab the data, encode as bytes, and then use stream
        data = file_obj.read()
        if hasattr(data, 'encode'):
            data = data.encode('utf-8')
        file_obj = util.wrap_as_stream(data)

    # load the scene using pyassimp
    scene = pyassimp.load(file_obj,
                          file_type=file_type)

    # save a record of mesh names used so we
    # don't have to do queries on mesh_id.values()
    mesh_names = set()
    # save a mapping for {id(mesh) : name}
    mesh_id = {}
    # save results as {name : Trimesh}
    meshes = {}
    # loop through scene LPMesh objects
    for m in scene.meshes:
        # skip meshes without tri/quad faces
        if m.faces.shape[1] not in [3, 4]:
            continue
        # if this mesh has the name of an existing mesh
        if m.name in mesh_names:
            # make it the name plus the unique ID of the object
            name = m.name + str(id(m))
        else:
            # otherwise just use the name it calls itself by
            name = m.name

        # save the name to mark as consumed
        mesh_names.add(name)
        # save the id:name mapping
        mesh_id[id(m)] = name
        # convert the mesh to a trimesh object
        meshes[name] = LP_to_TM(m)

    # now go through and collect the transforms from the scene
    # we are going to save them as a list of dict kwargs
    transforms = []
    # nodes as (parent, node) tuples
    # use deque so we can pop from both ends
    queue = collections.deque(
        [('world', n) for
         n in scene.rootnode.children])

    # consume the queue
    while len(queue) > 0:
        # parent name, node object
        parent, node = queue.pop()

        # assimp uses weirdly duplicate node names
        # object ID's are actually unique and consistent
        node_name = id(node)
        transforms.append({'frame_from': parent,
                           'frame_to': node_name,
                           'matrix': node.transformation})

        # loop through meshes this node uses
        # note that they are the SAME objects as converted
        # above so we can find their reference using id()
        for m in node.meshes:
            if id(m) not in mesh_id:
                continue

            # create kwargs for graph.update
            edge = {'frame_from': node_name,
                    'frame_to': str(id(m)) + str(node_name),
                    'matrix': np.eye(4),
                    'geometry': mesh_id[id(m)]}
            transforms.append(edge)

        # add any children to the queue to be visited
        for child in node.children:
            queue.appendleft((node_name, child))

    # release the loaded scene
    pyassimp.release(scene)

    # if we opened the file in this function close it
    if opened:
        file_obj.close()

    # create kwargs for trimesh.exchange.load.load_kwargs
    result = {'class': 'Scene',
              'geometry': meshes,
              'graph': transforms,
              'base_frame': 'world'}

    return result


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
        mesh_kwargs.update({'vertices': m.points,
                            'faces': m.trilist})
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
except BaseException:
    pass

try:
    import cyassimp
    loader = load_cyassimp
except BaseException:
    pass


if loader:
    _assimp_loaders.update(zip(_assimp_formats,
                               [loader] * len(_assimp_formats)))
