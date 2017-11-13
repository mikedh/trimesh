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

    # GLTF stores multiple files
    # we don't care where they get IO'd
    # so just put them in a dict and deal with elsewhere
    files = {}
    
    # we are defining a single scene, and will be setting the
    # world node to the 0th index
    result = {'scene'   : 0,
              'scenes'  : [{'nodes' : [0]}],
              "asset": {"version"   : "2.0",
                        "generator" : "github.com/mikedh/trimesh"}}

    # GLTF references meshes by index, so store them here
    mesh_index = {name : i for i, name in enumerate(scene.geometry.keys())}
    # grab the flattened scene graph in GLTF's format
    nodes = scene.graph.to_gltf(mesh_index=mesh_index)
    result.update(nodes)

    #A buffer is basically a file 
    buffers = []
    #A bufferView is a slice of a file
    views = []
    #An Accessor defines the datatype of that slice
    accessors = []
    # a mesh uses data from accessors
    meshes = []
        
    for name, mesh in scene.geometry.items():
        # meshes reference accessor indexes
        meshes.append({"name" : name,
                       "primitives" : [
                           {"attributes" : {"POSITION" : len(accessors)+1},
                            "indices" : len(accessors),
                            "mode" : 4}]}) # mode 4 is GL_TRIANGLES

        # mesh faces are stored as flat list of integers
        accessors.append({"bufferView": len(views),
                          "componentType": 5125,
                          "count": len(mesh.faces) * 3,
                          "max": [int(mesh.faces.max())],
                          "min": [0],
                          "type": "SCALAR"})

        accessors.append({"bufferView": len(views)+1,
                          "componentType": 5126,
                          "count": len(mesh.vertices),
                          "type": "VEC3",
                          "max": mesh.vertices.max(axis=0).tolist(),
                          "min": mesh.vertices.min(axis=0).tolist()})


        # convert to the correct dtypes
        # 5126 is a float32
        # 5125 is an unsigned 32 bit integer
        buffer_items = [mesh.faces.astype(np.uint32).tostring(),
                        mesh.vertices.astype(np.float32).tostring()]

        # create the buffer views 
        current_pos = 0
        for current_item in buffer_items:
            views.append({"buffer": len(buffers),
                          "byteOffset": current_pos,
                          "byteLength": len(current_item)})
            current_pos += len(current_item)

        # the data is just appended
        buffer_data = bytes().join(buffer_items)
        buffer_name = 'mesh_' + name + '.bin'
        buffers.append({'uri' : buffer_name,
                        'byteLength' : len(buffer_data)})
        files[buffer_name] = buffer_data
            
    result['meshes'] = meshes
    result['buffers'] = buffers
    result['bufferViews'] = views
    result['accessors'] = accessors

    files['model.gltf'] = json.dumps(result)
    return files
