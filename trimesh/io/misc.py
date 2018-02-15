import numpy as np

import json

from .. import util


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
    if isinstance(data, dict):
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


_misc_loaders = {'off': load_off,
                 'dict': load_dict,
                 'dict64': load_dict,
                 'json': load_dict,
                 'msgpack': load_msgpack}
