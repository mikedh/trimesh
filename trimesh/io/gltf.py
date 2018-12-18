"""
gltf.py
------------

Provides GLTF 2.0 exports of trimesh.Trimesh objects
as GL_TRIANGLES, and trimesh.Path2D/Path3D as GL_LINES
"""

import json
import collections

import numpy as np

from .. import util
from .. import rendering

# magic numbers which have meaning in GLTF
# most are uint32's of UTF-8 text
_magic = {'gltf': 1179937895,
          'json': 1313821514,
          'bin': 5130562}

# GLTF data type codes: little endian numpy dtypes
_types = {5120: '<i1',
          5121: '<u1',
          5122: '<i2',
          5123: '<u2',
          5125: '<u4',
          5126: '<f4'}

# GLTF data formats: numpy shapes
_shapes = {'SCALAR': -1,
           'VEC2': (-1, 2),
           'VEC3': (-1, 3),
           'VEC4': (-1, 4),
           'MAT2': (2, 2),
           'MAT3': (3, 3),
           'MAT4': (4, 4)}

# a default PBR metallic material
_default_material = {
    "pbrMetallicRoughness": {
        "baseColorFactor": [0, 0, 0, 0],
        "metallicFactor": 0,
        "roughnessFactor": 0}}

# specify common dtypes with forced little endian
float32 = np.dtype('<f4')
uint32 = np.dtype('<u4')
uint8 = np.dtype('<u1')


def export_gltf(scene):
    """
    Export a scene object as a GLTF directory.

    This has the advantage of putting each mesh into a separate
    file (buffer) as opposed to one large file.

    Parameters
    -----------
    scene: trimesh.Scene object

    Returns
    ----------
    export: dict, {file name : file data}
    """

    tree, buffer_items = _create_gltf_structure(scene)

    buffers = []
    views = []
    files = {}
    for i, name in zip(range(0, len(buffer_items), 2),
                       scene.geometry.keys()):

        # create the buffer views
        current_pos = 0
        # TODO this breaks for GL modes
        # that don't have 2 buffers (e.g. GL_LINES)
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


def export_glb(scene, include_normals=False):
    """
    Export a scene as a binary GLTF (GLB) file.

    Parameters
    ------------
    scene: trimesh.Scene object

    Returns
    ----------
    exported: bytes, exported result
    """

    tree, buffer_items = _create_gltf_structure(scene,
                                                include_normals=include_normals)

    # A bufferView is a slice of a file
    views = []
    # create the buffer views
    current_pos = 0
    for current_item in buffer_items:
        views.append({"buffer": 0,
                      "byteOffset": current_pos,
                      "byteLength": len(current_item)})
        current_pos += len(current_item)

    buffer_data = bytes().join(buffer_items)

    tree['buffers'] = [{'byteLength': len(buffer_data)}]
    tree['bufferViews'] = views

    # export the tree to JSON for the content of the file
    content = json.dumps(tree)
    # add spaces to content, so the start of the data
    # is 4 byte aligned as per spec
    content += (4 - ((len(content) + 20) % 4)) * ' '
    content = content.encode('utf-8')
    # make sure we didn't screw it up
    assert (len(content) % 4) == 0

    # the initial header of the file
    header = _byte_pad(
        np.array([_magic['gltf'],  # magic, turns into glTF
                  2,  # GLTF version
                  # length is the total length of the Binary glTF
                  # including Header and all Chunks, in bytes.
                  len(content) + len(buffer_data) + 28,
                  # contentLength is the length, in bytes,
                  # of the glTF content (JSON)
                  len(content),
                  # magic number which is 'JSON'
                  1313821514],
                 dtype='<u4').tobytes())

    # the header of the binary data section
    bin_header = _byte_pad(
        np.array([len(buffer_data),
                  0x004E4942],
                 dtype='<u4').tobytes())

    exported = bytes().join([header,
                             content,
                             bin_header,
                             buffer_data])

    return exported


def load_glb(file_obj, **passed):
    """
    Load a GLTF file in the binary GLB format into a trimesh.Scene.

    Implemented from specification:
    https://github.com/KhronosGroup/glTF/tree/master/specification/2.0

    Parameters
    ------------
    file_obj : file- like object
       Containing GLB data

    Returns
    ------------
    kwargs : dict
      Kwargs to instantiate a trimesh.Scene
    """

    # save the start position of the file for referencing
    # against lengths
    start = file_obj.tell()
    # read the first 20 bytes which contain section lengths
    head_data = file_obj.read(20)
    head = np.frombuffer(head_data,
                         dtype='<u4')

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

    # uint32 causes an error in read, so we convert to native int
    # for the length passed to read, for the JSON header
    json_data = file_obj.read(int(chunk_length))
    # convert to text
    if hasattr(json_data, 'decode'):
        json_data = json_data.decode('utf-8')
    # load the json header to native dict
    header = json.loads(json_data)

    # read the binary data referred to by GLTF as 'buffers'
    buffers = []
    while (file_obj.tell() - start) < length:
        # the last read put us past the JSON chunk
        # we now read the chunk header, which is 8 bytes
        chunk_head = file_obj.read(8)
        if len(chunk_head) != 8:
            # double check to make sure we didn't
            # read the whole file
            break

        chunk_length, chunk_type = np.frombuffer(chunk_head,
                                                 dtype='<u4')
        # make sure we have the right data type
        if chunk_type != _magic['bin']:
            raise ValueError('not binary GLTF!')
        # read the chunk
        chunk_data = file_obj.read(int(chunk_length))
        if len(chunk_data) != chunk_length:
            raise ValueError('chunk was not expected length!')
        buffers.append(chunk_data)

    # turn the layout header and data into kwargs
    # that can be used to instantiate a trimesh.Scene object
    kwargs = _read_buffers(header=header,
                           buffers=buffers)
    return kwargs


def _mesh_to_material(mesh, metallic=0.0, rough=0.0):
    """
    Create a simple GLTF material for a mesh using the most
    commonly occurring color in that mesh.

    Parameters
    ------------
    mesh: Trimesh object

    Returns
    ------------
    material: dict, in GLTF material format
    """
    # just get the most commonly occurring color
    color = mesh.visual.main_color
    # convert uint color to 0-1.0 float color
    color = color.astype(float32) / (
        (2 ** (8 * color.dtype.itemsize)) - 1)

    material = {'pbrMetallicRoughness':
                {'baseColorFactor': color.tolist(),
                 'metallicFactor': metallic,
                 'roughnessFactor': rough}}
    return material


def _create_gltf_structure(scene, include_normals=False):
    """
    Generate a GLTF header.

    Parameters
    -------------
    scene : trimesh.Scene
      Input scene data

    Returns
    ---------------
    tree : dict
      Contains required keys for a GLTF scene
    buffer_items : list
      Contains bytes of data
    """
    # we are defining a single scene, and will be setting the
    # world node to the 0th index
    tree = {'scene': 0,
            'scenes': [{'nodes': [0]}],
            "asset": {"version": "2.0",
                      "generator": "github.com/mikedh/trimesh"},
            'accessors': [],
            'meshes': [],
            'materials': []}

    # GLTF references meshes by index, so store them here
    mesh_index = {name: i for i, name in
                  enumerate(scene.geometry.keys())}
    # grab the flattened scene graph in GLTF's format
    nodes = scene.graph.to_gltf(mesh_index=mesh_index)
    tree.update(nodes)

    buffer_items = []
    for name, geometry in scene.geometry.items():
        if util.is_instance_named(geometry, 'Trimesh'):
            # add the junk
            _append_mesh(mesh=geometry,
                         name=name,
                         tree=tree,
                         buffer_items=buffer_items,
                         include_normals=include_normals)
        elif util.is_instance_named(geometry, 'Path'):
            _append_path(path=geometry,
                         name=name,
                         tree=tree,
                         buffer_items=buffer_items)

    # if nothing defined a material remove it from the structure
    if len(tree['materials']) == 0:
        tree.pop('materials')

    return tree, buffer_items


def _append_mesh(mesh,
                 name,
                 tree,
                 buffer_items,
                 include_normals):
    """
    Append a mesh to the scene structure and put the
    data into buffer_items.

    Parameters
    -------------
    mesh : trimesh.Trimesh
      Source geometry
    name : str
      Name of geometry
    tree : dict
      Will be updated with data from mesh
    buffer_items
      Will have buffer appended with mesh data
    """
    # meshes reference accessor indexes
    tree['meshes'].append({
        "name": name,
        "primitives": [
            {"attributes":
             {"POSITION": len(tree['accessors']) + 1},
             "indices": len(tree['accessors']),
             "mode": 4  # mode 4 is GL_TRIANGLES
             }]})

    # if units are defined, store them as an extra:
    # https://github.com/KhronosGroup/glTF/tree/master/extensions
    if mesh.units is not None:
        tree['meshes'][-1]['extras'] = {'units': str(mesh.units)}

    # accessors refer to data locations
    # mesh faces are stored as flat list of integers
    tree['accessors'].append({
        "bufferView": len(buffer_items),
        "componentType": 5125,
        "count": len(mesh.faces) * 3,
        "max": [int(mesh.faces.max())],
        "min": [0],
        "type": "SCALAR"})
    # convert mesh data to the correct dtypes
    # faces: 5125 is an unsigned 32 bit integer
    buffer_items.append(
        _byte_pad(mesh.faces.astype(uint32).tobytes()))

    # the vertex accessor
    tree['accessors'].append({
        "bufferView": len(buffer_items),
        "componentType": 5126,
        "count": len(mesh.vertices),
        "type": "VEC3",
        "byteOffset": 0,
        "max": mesh.vertices.max(axis=0).tolist(),
        "min": mesh.vertices.min(axis=0).tolist()})
    # vertices: 5126 is a float32
    buffer_items.append(
        _byte_pad(mesh.vertices.astype(float32).tobytes()))

    # make sure to append colors after other stuff to
    # not screw up the indexes of accessors or buffers
    if mesh.visual.kind is not None:
        # make sure colors are RGBA, this should always be true
        assert mesh.visual.vertex_colors.shape == (len(mesh.vertices), 4)

        # add the reference for vertex color
        tree['meshes'][-1]['primitives'][0]['attributes']['COLOR_0'] = len(
            tree['accessors'])

        color_data = _byte_pad(
            mesh.visual.vertex_colors.astype(
                uint8).tobytes())

        # the vertex color accessor data
        tree['accessors'].append({
            "bufferView": len(buffer_items),
            "componentType": 5121,
            "normalized": True,
            "count": len(mesh.vertices),
            "type": "VEC4",
            "byteOffset": 0})

        # the actual color data
        buffer_items.append(color_data)
    else:
        # if no colors, set a material
        tree['meshes'][-1]['primitives'][0]['material'] = len(
            tree['materials'])
        # add a default- ish material
        tree['materials'].append(_mesh_to_material(mesh))

    if include_normals:
        # add the reference for vertex color
        tree['meshes'][-1]['primitives'][0]['attributes']['NORMAL'] = len(
            tree['accessors'])

        normal_data = _byte_pad(
            mesh.vertex_normals.astype(
                float32).tobytes())
        # the vertex color accessor data
        tree['accessors'].append({
            "bufferView": len(buffer_items),
            "componentType": 5126,
            "count": len(mesh.vertices),
            "type": "VEC3",
            "byteOffset": 0})
        # the actual color data
        buffer_items.append(normal_data)


def _byte_pad(data, bound=4):
    """
    GLTF wants chunks aligned with 4- byte boundaries
    so this function will add padding to the end of a
    chunk of bytes so that it aligns with a specified
    boundary size

    Parameters
    --------------
    data : bytes
      Data to be padded
    bound : int
      Length of desired boundary

    Returns
    --------------
    padded : bytes
      Result where: (len(padded) % bound) == 0
    """
    bound = int(bound)
    if len(data) % bound != 0:
        pad = bytes(bound - (len(data) % bound))
        result = bytes().join([data, pad])
        assert (len(result) % bound) == 0
        return result

    return data


def _append_path(path, name, tree, buffer_items):
    """
    Append a 2D or 3D path to the scene structure and put the
    data into buffer_items.

    Parameters
    -------------
    path : trimesh.Path2D or trimesh.Path3D
      Source geometry
    name : str
      Name of geometry
    tree : dict
      Will be updated with data from path
    buffer_items
      Will have buffer appended with path data
    """

    # convert the path to the unnamed args for
    # a pyglet vertex list
    vxlist = rendering.path_to_vertexlist(path)

    tree['meshes'].append({
        "name": name,
        "primitives": [
            {"attributes":
             {"POSITION": len(tree['accessors'])},
             "mode": 1,  # mode 1 is GL_LINES
             "material": len(tree['materials'])
             }]})

    # if units are defined, store them as an extra:
    # https://github.com/KhronosGroup/glTF/tree/master/extensions
    if path.units is not None:
        tree['meshes'][-1]['extras'] = {'units': str(path.units)}

    tree['accessors'].append({
        "bufferView": len(buffer_items),
        "componentType": 5126,
        "count": vxlist[0],
        "type": "VEC3",
        "byteOffset": 0,
        "max": path.vertices.max(axis=0).tolist(),
        "min": path.vertices.min(axis=0).tolist()})

    # TODO add color support to Path object
    # this is just exporting everying as black
    tree['materials'].append(_default_material)

    # data is the second value of the fourth field
    # which is a (data type, data) tuple
    buffer_items.append(
        _byte_pad(vxlist[4][1].astype(float32).tobytes()))


def _read_buffers(header, buffers):
    """
    Given a list of binary data and a layout, return the
    kwargs to create a scene object.

    Parameters
    -----------
    header : dict
      With GLTF keys
    buffers : list of bytes
      Stored data

    Returns
    -----------
    kwargs : dict
      Can be passed to load_kwargs for a trimesh.Scene
    """
    # split buffer data into buffer views
    views = []
    for view in header['bufferViews']:
        start = view['byteOffset']
        end = start + view['byteLength']
        views.append(buffers[view['buffer']][start:end])
        assert len(views[-1]) == view['byteLength']

    # load data from buffers and bufferviews into numpy arrays
    # using the layout described by accessors
    access = []
    for a in header['accessors']:
        data = views[a['bufferView']]
        dtype = _types[a['componentType']]
        shape = _shapes[a['type']]

        # is the accessor offset in a buffer
        if 'byteOffset' in a:
            start = a['byteOffset']
        else:
            start = 0
        # basically the number of columns
        per_count = np.abs(np.product(shape))
        # length is the number of bytes per item times total
        length = np.dtype(dtype).itemsize * a['count'] * per_count
        end = start + length

        array = np.frombuffer(data[start:end], dtype=dtype).reshape(shape)

        assert len(array) == a['count']
        access.append(array)

    # turn materials into a simple list of colors if populated
    colors = []
    if 'materials' in header:
        for mat in header['materials']:
            # get the base color of the material
            try:
                color = np.array(mat['pbrMetallicRoughness']['baseColorFactor'],
                                 dtype='<f8')
            except BaseException:
                color = np.array([.5, .5, .5, 1])

            # convert float 0-1 colors to uint8 colors and append
            colors.append((color * 255).astype(uint8))

    # load data from accessors into Trimesh objects
    meshes = collections.OrderedDict()
    for index, m in enumerate(header['meshes']):
        color = None
        kwargs = collections.defaultdict(list)
        for p in m['primitives']:
            if p['mode'] != 4:
                continue

            faces = access[p['indices']].reshape((-1, 3))
            verts = access[p['attributes']['POSITION']]

            kwargs['faces'].append(faces)
            kwargs['vertices'].append(verts)

            if 'material' in p:
                color = colors[p['material']]
            else:
                color = [128, 128, 128, 255]
            # stack colors to line up with faces
            kwargs['face_colors'].append(np.tile(color, (len(faces), 1)))
        # re- index faces
        (kwargs['vertices'],
         kwargs['faces']) = util.append_faces(kwargs['vertices'],
                                              kwargs['faces'])
        # stack colors
        kwargs['face_colors'] = np.vstack(kwargs['face_colors'])

        # try loading units from the GLTF extra
        if 'extras' in m and 'units' in m['extras']:
            try:
                units = str(m['extras']['units'])
                kwargs['metadata'] = {'units': units}
            except BaseException:
                pass

        if 'name' in m:
            meshes[m['name']] = kwargs
        else:
            meshes[index] = kwargs

    # make it easier to reference nodes
    nodes = header['nodes']
    # nodes are referenced by index
    # save their string names if they have one
    # node index (int) : name (str)
    names = {}
    for i, n in enumerate(nodes):
        if 'name' in n:
            names[i] = n['name']
        else:
            names[i] = str(i)

    # make sure we have a unique base frame name
    base_frame = 'world'
    if base_frame in names:
        base_frame = str(int(np.random.random() * 1e10))
    names[base_frame] = base_frame

    # visited, kwargs for scene.graph.update
    graph = collections.deque()
    # unvisited, pairs of node indexes
    queue = collections.deque()

    # the index of the node which is the root of the tree
    for root in header['scenes'][header['scene']]['nodes']:
        # add transform from base frame to these root nodes
        queue.append([base_frame, root])
        # add any children to the queue
        if 'children' in nodes[root]:
            queue.extend([root, c] for c in
                         nodes[root]['children'])
    # meshes are listed by index rather than name
    # replace the index with a nicer name
    mesh_names = list(meshes.keys())

    # go through the nodes tree to populate
    # kwargs for scene graph loader
    while len(queue) > 0:
        # (int, int) pair of node indexes
        a, b = queue.pop()

        # dict of child node
        child = nodes[b]
        # add edges of children to be processed
        if 'children' in child:
            queue.extend([[b, i] for i in child['children']])

        # kwargs to be passed to scene.graph.update
        kwargs = {'frame_from': names[a],
                  'frame_to': names[b]}

        # grab matrix from child
        # parent -> child relationships have matrix stored in child
        # for the transform from parent to child
        if 'matrix' in child:
            kwargs['matrix'] = np.array(child['matrix']).reshape((4, 4)).T
        else:
            kwargs['matrix'] = np.eye(4)

        if 'mesh' in child:
            kwargs['geometry'] = mesh_names[child['mesh']]
        graph.append(kwargs)

    # kwargs to be loaded
    result = {'class': 'Scene',
              'geometry': meshes,
              'graph': graph,
              'base_frame': base_frame}

    return result


_gltf_loaders = {'glb': load_glb}
