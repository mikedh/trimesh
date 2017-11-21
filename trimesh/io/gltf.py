import json

import numpy as np


def export_gltf(scene):
    '''
    Export a scene object as a GLTF directory

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
    header = np.array([1179937895,  # magic, turns into glTF
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
