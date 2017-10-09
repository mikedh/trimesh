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

    Discards texture normals and vertex color information.


    Parameters
    ----------
    file_obj: file object containing a wavefront file
    file_type: not used

    Returns
    ----------
    loaded: dict with kwargs for Trimesh constructor (vertices, faces)
    '''
    text_original = file_obj.read()
    if hasattr(text_original, 'decode'):
        text_original = text_original.decode('utf-8')
    # get rid of stupid newlines
    text_original = text_original.replace(
        '\r\n', '\n').replace(
        '\r', '\n') + ' \n'

    # for faces, remove the '/' notation in the raw text
    # the regex does the following:
    # find charecter '/' and then any non- whitespace charecter,
    # up to a newline or space, then stop. Example:
    # test = "f 233/233/233 12//12//12\nf 233/233/233 12//12//12 "
    # In [0]: re.split('/\S*[ \n]', test)
    # Out[0]: ['f 233', '12', 'f 233', '12', '']
    # we then re-join it into a larger string so we
    # can split by just whitespace
    # we add a space before every newline to make things easy on ourselves
    text = ' '.join(re.split('/\S* ',
                             text_original.replace('\n', ' \n')))

    # remove all comments
    # regex:
    # # : comments start with pounds
    # .*: zero or more of any charecter
    # \n: up to a newline
    text = re.sub('#.*\n', '\n', text)

    # more impenetrable regexes
    # this one to pulls faces from directly from the text
    # - find the 'f' char
    # - followed by one or more spaces
    # - followed by one or more 0-9 digits
    # - followed by one or more spaces
    # - (repeat for exactly 3 integers for tris, 4 for quads)
    # - followed by zero or more spaces
    # - followed by exactly one newline
    re_tris = 'f +\d+ +\d+ +\d+ *\n'
    re_quad = 'f +\d+ +\d+ +\d+ +\d+ *\n'

    # Split the file on object lines -- any line that begins with an 'o'
    # indicates a new mesh.
    # regex does:
    # '^' : match the first charecter of each newline (with multiline flag)
    # 'o' : match the charecter 'o'
    # '.*': match zero or more of any charecter
    # '\n': up to a newline
    re_obj = '^o.*\n'

    count_vertices = 0
    loaded_data = collections.deque()

    # loop through each sub- mesh
    for text in re.split(re_obj, text, flags=re.MULTILINE):
        if text is None:
            continue

        # find all triangular faces with a regex
        face_tri = ' '.join(re.findall(re_tris, text)
                            ).replace('f', ' ').split()
        # convert triangular faces into a numpy array
        face_tri = np.array(face_tri, dtype=np.int64).reshape((-1, 3))

        # find all quad faces with a regex
        face_quad = ' '.join(re.findall(re_quad, text)
                             ).replace('f', ' ').split()
        # convert quad faces into a numpy array
        face_quad = np.array(face_quad, dtype=np.int64).reshape((-1, 4))

        # stack the faces into a single (n,3) list
        # triangulate any quad faces
        # this thin wrapper for vstack will ignore empty lists
        faces = util.vstack_empty((face_tri,
                                   geometry.triangulate_quads(face_quad)))

        # If we didn't load any faces, we snagged extra propertie
        # around an object split -- for now, avoid dealing with them.
        if len(faces) == 0:
            continue

        # wavefront has 1- indexed faces, as opposed to 0- indexed
        # additionally, decrement by number of vertices used in prior objects
        faces = faces.astype(np.int64) - 1 - count_vertices

        # find the data with predictable lengths using numpy
        data = np.array(text.split())
        # find the locations of keys, then find the proceeding values
        # indexes which contain vertex information
        vid = np.nonzero(data == 'v')[0].reshape((-1, 1)) + np.arange(3) + 1
        # indexes which contain vertex normal information
        nid = np.nonzero(data == 'vn')[0].reshape((-1, 1)) + np.arange(3) + 1
        # some varients of the format have face groups
        gid = np.nonzero(data == 'g')[0].reshape((-1, 1)) + 1

        loaded = {'vertices': data[vid].astype(float),
                  'vertex_normals': data[nid].astype(float),
                  'faces': faces}

        # if face groups have been defined add them to metadata
        if len(gid) > 0:
            # indexes which contain face information
            face_key = np.nonzero(data == 'f')[0]
            groups = np.zeros(len(faces), dtype=int)
            for i, g in enumerate(gid):
                groups[np.nonzero(face_key > g)[0]] = i
            loaded['metadata'] = {'face_groups': groups}

        count_vertices += len(loaded['vertices'])

        loaded_data.append(loaded)

    return list(loaded_data)


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
