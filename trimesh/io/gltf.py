'''
This module provides GLTF 2.0 exports
'''

import json
import collections

import numpy as np

# magic numbers which have meaning in GLTF
# most are uint32's of UTF-8 text
_magic = {'gltf': 1179937895,
          'json': 1313821514,
          'bin': 5130562}

# GLTF data type codes: numpy dtypes
_types = {5120: np.bool,
          5122: np.int16,
          5123: np.uint16,
          5125: np.uint32,
          5126: np.float32}

# GLTF data formats: numpy shapes
_shapes = {'SCALAR': -1,
           'VEC2': (-1, 2),
           'VEC3': (-1, 3),
           'VEC4': (-1, 4),
           'MAT2': (2, 2),
           'MAT3': (3, 3),
           'MAT4': (4, 4)}


def export_gltf(scene):
    '''
    Export a scene object as a GLTF directory.

    This has the advantage of putting each mesh into a separate file (buffer)
    as opposed to one large file, but means multiple files need to be tracked.

    Parameters
    -----------
    scene: trimesh.Scene object

    Returns
    ----------
    export: dict, {file name : file data}
    '''

    tree, buffer_items = _gltf_structure(scene)

    buffers = []
    views = []
    files = {}
    for i, name in zip(range(0, len(buffer_items), 2),
                       scene.geometry.keys()):

        # create the buffer views
        current_pos = 0
        for j in range(2):
            current_item = buffer_items[i + j]
            views.append({"buffer": len(buffers),
                          "byteOffset": current_pos,
                          "byteLength": len(current_item)})
            current_pos += len(current_item)

        # the data is just appended
        buffer_data = bytes().join(buffer_items[i:i + 2])
        buffer_name = 'mesh_' + name + '.bin'
        buffers.append({'uri': buffer_name,
                        'byteLength': len(buffer_data)})
        files[buffer_name] = buffer_data

    tree['buffers'] = buffers
    tree['bufferViews'] = views

    files['model.gltf'] = json.dumps(tree).encode('utf-8')
    return files


def export_glb(scene):
    '''
    Export a scene as a binary GLTF (GLB) file.

    Parameters
    ------------
    scene: trimesh.Scene object

    Returns
    ----------
    exported: bytes, exported result
    '''

    tree, buffer_items = _gltf_structure(scene)

    # A bufferView is a slice of a file
    views = []
    # create the buffer views
    current_pos = 0
    for current_item in buffer_items:
        views.append({"buffer": 0,
                      "byteOffset": current_pos,
                      "byteLength": len(current_item)})
        current_pos += len(current_item)

        # the data is just appended
    buffer_data = bytes().join(buffer_items)

    tree['buffers'] = [{'byteLength': len(buffer_data)}]
    tree['bufferViews'] = views

    # export the tree to JSON for the content of the file
    content = json.dumps(tree)
    # add spaces to content, so the start of the data
    # is 4 byte aligned as per spec
    content += ((len(content) + 20) % 4) * ' '
    content = content.encode('utf-8')

    # the initial header of the file
    header = np.array([_magic['gltf'],  # magic, turns into glTF
                       2,  # GLTF version
                       # length is the total length of the Binary glTF
                       # including Header and all Chunks, in bytes.
                       len(content) + len(buffer_data) + 28,
                       # contentLength is the length, in bytes,
                       # of the glTF content (JSON)
                       len(content),
                       # magic number which is 'JSON'
                       1313821514],
                      dtype=np.uint32)

    # the header of the binary data section
    bin_header = np.array([len(buffer_data),
                           0x004E4942],
                          dtype=np.uint32)

    exported = (header.tostring() +
                content +
                bin_header.tostring() +
                buffer_data)

    return exported


def _gltf_structure(scene):
    # we are defining a single scene, and will be setting the
    # world node to the 0th index
    tree = {'scene': 0,
            'scenes': [{'nodes': [0]}],
            "asset": {"version": "2.0",
                      "generator": "github.com/mikedh/trimesh"},
            'accessors': [],
            'meshes': []}

    # GLTF references meshes by index, so store them here
    mesh_index = {name: i for i, name in enumerate(scene.geometry.keys())}
    # grab the flattened scene graph in GLTF's format
    nodes = scene.graph.to_gltf(mesh_index=mesh_index)
    tree.update(nodes)

    buffer_items = []
    for name, mesh in scene.geometry.items():
        # meshes reference accessor indexes
        tree['meshes'].append({"name": name,
                               "primitives": [
                                   {"attributes": {"POSITION": len(tree['accessors']) + 1},
                                    "indices": len(tree['accessors']),
                                    "mode": 4}]})  # mode 4 is GL_TRIANGLES

        # accessors refer to data locations
        # mesh faces are stored as flat list of integers
        tree['accessors'].append({"bufferView": len(buffer_items),
                                  "componentType": 5125,
                                  "count": len(mesh.faces) * 3,
                                  "max": [int(mesh.faces.max())],
                                  "min": [0],
                                  "type": "SCALAR"})

        # the vertex accessor
        tree['accessors'].append({"bufferView": len(buffer_items) + 1,
                                  "componentType": 5126,
                                  "count": len(mesh.vertices),
                                  "type": "VEC3",
                                  "max": mesh.vertices.max(axis=0).tolist(),
                                  "min": mesh.vertices.min(axis=0).tolist()})

        # convert to the correct dtypes
        # 5126 is a float32
        # 5125 is an unsigned 32 bit integer
        # add faces, then vertices
        buffer_items.append(mesh.faces.astype(np.uint32).tostring())
        buffer_items.append(mesh.vertices.astype(np.float32).tostring())

    return tree, buffer_items


def load_glb(file_obj, **passed):
    '''
    Load a GLTF file in the binary GLB format into a trimesh.Scene.

    Implemented from specification:
    https://github.com/KhronosGroup/glTF/tree/master/specification/2.0

    Parameters
    ------------
    file_obj: file- like object containing GLB file

    Returns
    ------------
    scene: trimesh.Scene object, containing data from GLB file
    '''

    # save the start position of the file for referencing
    # against lengths
    start = file_obj.tell()
    # read the first 20 bytes which contain section lengths
    head_data = file_obj.read(20)
    head = np.fromstring(head_data,
                         dtype=np.uint32)

    # check to make sure first index is gltf
    # and second is 2, for GLTF 2.0
    if head[0] != _magic['gltf'] or head[1] != 2:
        raise ValueError('file is not GLTF 2.0')

    # overall file length
    # first chunk length
    # first chunk type
    length, chunk_length, chunk_type = head[2:]

    # first chunk should be JSON header
    if chunk_type != _magic['json']:
        raise ValueError('no initial JSON header!')

    # np.uint32 causes an error in read, so we convert to native int
    # for the length passed to read, for the JSON header
    json_data = file_obj.read(int(chunk_length))

    # load the json header to native dict
    j = json.loads(json_data)

    # read the binary data referred to by GLTF as 'buffers'
    buffers = []
    while (file_obj.tell() - start) < length:
        # the last read put us past the JSON chunk
        # we now read the chunk header, which is 8 bytes
        chunk_head = file_obj.read(8)
        if len(chunk_head) != 8:
            break
        chunk_length, chunk_type = np.fromstring(chunk_head,
                                                 dtype=np.uint32)
        if chunk_type != _magic['bin']:
            raise ValueError('not binary GLTF!')
        chunk_data = file_obj.read(int(chunk_length))
        if len(chunk_data) != chunk_length:
            raise ValueError('chunk was not expected length!')
        buffers.append(chunk_data)

    # split buffer data into buffer views
    views = []
    for view in j['bufferViews']:
        start = view['byteOffset']
        end = start + view['byteLength']
        views.append(buffers[view['buffer']][start:end])
        assert len(views[-1]) == view['byteLength']

    # load data from buffers and bufferviews into numpy arrays
    access = []
    for a in j['accessors']:
        data = views[a['bufferView']]
        dtype = _types[a['componentType']]
        shape = _shapes[a['type']]
        array = np.fromstring(data,
                              dtype=dtype).reshape(shape)
        assert len(array) == a['count']
        access.append(array)

    # load data from accessors into Trimesh objects
    meshes = collections.OrderedDict()
    for m in j['meshes']:
        kwargs = collections.defaultdict(list)
        for p in m['primitives']:
            if p['mode'] != 4:
                raise ValueError('only GL_TRIANGLES meshes supported!')
            kwargs['faces'].append(access[p['indices']].reshape((-1, 3)))
            kwargs['vertices'].append(access[p['attributes']['POSITION']])
        for key, value in kwargs.items():
            kwargs[key] = np.vstack(value)
        meshes[m['name']] = kwargs

    # the index of the node which is the root of the tree
    root = j['scenes'][j['scene']]['nodes']

    if len(root) != 1:
        raise ValueError('multiple scene roots')
    root = root[0]

    # make it easier to reference nodes
    nodes = j['nodes']

    # nodes are referenced by index
    # save their string names if they have one
    names = {}
    for i, n in enumerate(nodes):
        if 'name' in n:
            names[i] = n['name']
        else:
            names[i] = str(i)

    # visited, kwargs for scene.graph.update
    graph = collections.deque()
    # unvisited, pairs of indexes for nodes
    queue = collections.deque([root, c] for c in
                              nodes[root]['children'])

    # meshes are listed by index rather than name
    # replace the index with a nicer name
    mesh_names = list(meshes.keys())

    # go through the nodes tree to populate
    # kwargs for scene graph loader
    while len(queue) > 0:
        # (int, int) pair of node indexes
        edge = queue.pop()

        # dict of child node
        child = nodes[edge[1]]
        # add edges of children to be processed
        if 'children' in child:
            queue.extend([[edge[0], i] for i in child['children']])

        # kwargs to be passed to scene.graph.update
        kwargs = {'frame_from': names[edge[0]],
                  'frame_to': names[edge[1]]}

        # grab matrix from child
        # parent -> child relationships have matrix stored in child
        # for the transform from parent to child
        if 'matrix' in child:
            kwargs['matrix'] = np.array(child['matrix']).reshape((4, 4)).T
        if 'mesh' in child:
            kwargs['geometry'] = mesh_names[child['mesh']]
        graph.append(kwargs)

    # kwargs to be loaded
    result = {'class': 'Scene',
              'geometry': meshes,
              'graph': graph}

    return result


_gltf_loaders = {'glb': load_glb}
