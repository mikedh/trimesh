import numpy as np
import struct

from ..util import base64_to_array

def load_off(file_obj, file_type=None):
    header_string = file_obj.readline().decode().strip()
    if not header_string == 'OFF': 
        raise NameError('Not an OFF file! Header was ' + header_string)

    header  = np.array(file_obj.readline().split()).astype(int)
    blob    = np.array(file_obj.read().split())
    data_ok = np.sum(header * [3,4,0]) == len(blob)    
    if not data_ok: raise NameError('Incorrect number of vertices or faces!')

    vertices = blob[0:(header[0]*3)].astype(float).reshape((-1,3))
    faces    = blob[(header[0]*3):].astype(int).reshape((-1,4))[:,1:]
  
    return {'vertices' : vertices,
            'faces'    : faces}

def load_wavefront(file_obj, file_type=None):
    '''
    Loads a Wavefront .obj file_obj into a Trimesh object
    Discards texture normals and vertex color information
    https://en.wikipedia.org/wiki/Wavefront_.obj_file
    '''
    data     = np.array(file_obj.read().split())
    data_str = data.astype(str)
    
    # find the locations of keys, then find the proceeding values
    vid = np.nonzero(data_str == 'v')[0].reshape((-1,1))  + np.arange(3) + 1
    nid = np.nonzero(data_str == 'vn')[0].reshape((-1,1)) + np.arange(3) + 1
    fid = np.nonzero(data_str == 'f')[0].reshape((-1,1))  + np.arange(3) + 1
    
    # if we wanted to use the texture/vertex normals, we could slice this differently.
    faces = np.array([i.split(b'/') for i in data[fid].reshape(-1)])[:,0].reshape((-1,3))
    # wavefront has 1- indexed faces, as opposed to 0- indexed
    faces = faces.astype(int) - 1
    return {'vertices'       : data[vid].astype(float),
            'vertex_normals' : data[nid].astype(float),
            'faces'          : faces}

def load_dict(data, file_type=None):
    return data

def load_dict64(data, file_type=None):
    for key in ('vertices', 'faces', 'face_normals'):
        data[key] = base64_to_array(data[key])
    return data

_misc_loaders = {'obj'    : load_wavefront,
                 'off'    : load_off,
                 'dict'   : load_dict,
                 'dict64' : load_dict64}
