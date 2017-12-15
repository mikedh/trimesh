import numpy as np

import collections
import json
import re

from .. import util
from .. import geometry


def load_off(file_obj, file_type=None):
    '''
    Load an OFF file into the kwargs for a Trimesh constructor


    Parameters
    ----------
    file_obj: file object containing an OFF file
    file_type: not used

    Returns
    ----------
    loaded: dict with kwargs for Trimesh constructor (vertices, faces)

    '''
    header_string = file_obj.readline()
    if hasattr(header_string, 'decode'):
        header_string = header_string.decode('utf-8')
    header_string = header_string.strip()

    if not header_string == 'OFF':
        raise NameError('Not an OFF file! Header was ' + header_string)

    header = np.array(file_obj.readline().split()).astype(int)
    blob = np.array(file_obj.read().split())
    data_ok = np.sum(header * [3, 4, 0]) == len(blob)
    if not data_ok:
        raise NameError('Incorrect number of vertices or faces!')

    vertices = blob[0:(header[0] * 3)].astype(float).reshape((-1, 3))
    faces = blob[(header[0] * 3):].astype(int).reshape((-1, 4))[:, 1:]

    return {'vertices': vertices,
            'faces': faces}

def load_wavefront(file_obj, file_type=None):
    '''
    Loads an ascii Wavefront OBJ file_obj into kwargs
    for the Trimesh constructor.

    Vertices with the same position but different normals or uvs are split
    into multiple vertices.

    Colors are discarded.

    Parameters
    ----------
    file_obj: file object containing a wavefront file
    file_type: not used

    Returns
    ----------
    loaded: dict with kwargs for Trimesh constructor (vertices, faces)
    '''

    # make sure text is UTF-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []
    def append_mesh():
        # append kwargs for a Trimesh constructor to our list of meshes
        if len(current['f']) > 0:
            loaded = {'vertices': np.array(current['v']),
                      'vertex_normals': np.array(current['vn']),
                      'faces': np.array(current['f'], dtype=np.int64).reshape((-1,3)),
                      'metadata': {}}
            if len(current['vt']) > 0:
                loaded['metadata']['vertex_texture'] = np.array(current['vt'])
            if len(current['g']) > 0:
                # build face groups information
                face_groups = np.zeros(len(current['f'])//3, dtype=int)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups
            meshes.append(loaded)


    attribs = {k : [] for k in ['v', 'vt', 'vn']}
    current = {k : [] for k in ['v', 'vt', 'vn', 'f', 'g']}
    remap_table = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs: 
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values-- colors shoved into vertices are ignored
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3) 
                if not f in remap_table:
                    remap_table[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0])-1])
                    if len(f_split) > 1 and f_split[1] != '':
                        current['vt'].append(attribs['vt'][int(f_split[1])-1])
                    if len(f_split) > 2:
                        current['vn'].append(attribs['vn'][int(f_split[2])-1])
                current['f'].append(remap_table[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            current['v'] = []
            current['vn'] = []
            current['f'] = []
            current['g'] = []
            remap_table = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes


def load_msgpack(blob, file_type=None):
    '''
    Load a dict packed with msgpack into kwargs for Trimesh constructor

    Parameters
    ----------
    blob: msgpack packed dict with keys for 'vertices' and 'faces'
    file_type: not used

    Returns
    ----------
    loaded: kwargs for Trimesh constructor (aka mesh=trimesh.Trimesh(**loaded))
    '''

    import msgpack
    if hasattr(blob, 'read'):
        data = msgpack.load(blob)
    else:
        data = msgpack.loads(blob)
    loaded = load_dict(data)
    return loaded


def load_dict(data, file_type=None):
    '''
    Load multiple input types into kwargs for a Trimesh constructor.
    Tries to extract keys ['faces', 'vertices', 'face_normals', 'vertex_normals'].

    Parameters
    ----------
    data: accepts multiple forms
          -dict: has keys for vertices and faces as (n,3) numpy arrays
          -dict: has keys for vertices/faces (n,3) arrays encoded as dicts/base64
                 with trimesh.util.array_to_encoded/trimesh.util.encoded_to_array
          -str:  json blob as dict with either straight array or base64 values
          -file object: json blob of dict
    file_type: not used

    Returns
    -----------
    loaded: dict with keys
            -vertices: (n,3) float
            -faces:    (n,3) int
            -face_normals: (n,3) float (optional)
    '''
    if data is None:
        raise ValueError('data passed to load_dict was None!')
    if util.is_instance_named(data, 'Trimesh'):
        return data
    if util.is_string(data):
        if not '{' in data:
            raise ValueError('Object is not a JSON encoded dictionary!')
        data = json.loads(data.decode('utf-8'))
    elif util.is_file(data):
        data = json.load(data)

    # what shape should the data be to be usable
    mesh_data = {'vertices': (-1, 3),
                 'faces': (-1, (3, 4)),
                 'face_normals': (-1, 3),
                 'face_colors': (-1, (3, 4)),
                 'vertex_normals': (-1, 3),
                 'vertex_colors': (-1, (3, 4))}

    # now go through data structure and if anything is encoded as base64
    # pull it back into numpy arrays
    if util.is_dict(data):
        loaded = {}
        data = util.decode_keys(data, 'utf-8')
        for key, shape in mesh_data.items():
            if key in data:
                loaded[key] = util.encoded_to_array(data[key])
                if not util.is_shape(loaded[key], shape):
                    raise ValueError('Shape of %s is %s, not %s!',
                                     key,
                                     str(loaded[key].shape),
                                     str(shape))
        if len(key) == 0:
            raise ValueError('Unable to extract any mesh data!')
        return loaded
    else:
        raise ValueError('%s object passed to dict loader!',
                         data.__class__.__name__)


_misc_loaders = {'obj': load_wavefront,
                 'off': load_off,
                 'dict': load_dict,
                 'dict64': load_dict,
                 'json': load_dict,
                 'msgpack': load_msgpack}
