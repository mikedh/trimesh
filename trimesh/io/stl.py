import numpy as np
import struct

from ..util      import is_binary_file

# define a numpy datatype for the STL file
_stl_dtype = np.dtype([('normals',    np.float32, (3)), 
                       ('vertices',   np.float32, (3,3)), 
                       ('attributes', np.uint16)])
 
def load_stl(file_obj, file_type=None):
    if is_binary_file(file_obj): return load_stl_binary(file_obj)
    else:                        return load_stl_ascii(file_obj)
        
def load_stl_binary(file_obj):
    '''
    Load a binary STL file into a trimesh object. 
    Uses a single main struct.unpack call, and is significantly faster
    than looping methods or ASCII STL. 
    '''
    # get the file_obj header
    header = file_obj.read(80)
    
    # get the file information about the number of triangles
    tri_count    = int(struct.unpack("@i", file_obj.read(4))[0])
    
    # now we check the length from the header versus the length of the file
    # data_start should always be position 84, but hard coding that felt ugly
    data_start = file_obj.tell()
    # this seeks to the end of the file (position 0, relative to the end of the file 'whence=2')
    file_obj.seek(0, 2)
    # we save the location of the end of the file and seek back to where we started from
    data_end = file_obj.tell()
    file_obj.seek(data_start)
    
    # the binary format has a rigidly defined structure, and if the length
    # of the file doesn't match the header, the loaded version is almost
    # certainly going to be garbage. 
    data_ok = (data_end - data_start) == (tri_count * _stl_dtype.itemsize)
   
    # this check is to see if this really is a binary STL file. 
    # if we don't do this and try to load a file that isn't structured properly 
    # the struct.unpack call uses 100% memory until the whole thing crashes, 
    # so it's much better to raise an exception here. 
    if not data_ok:
        raise ValueError('Binary STL has incorrect length in header!')
    
    # all of our vertices will be loaded in order due to the STL format, 
    # so faces are just sequential indices reshaped. 
    faces = np.arange(tri_count*3).reshape((-1,3))
    blob  = np.fromstring(file_obj.read(), dtype=_stl_dtype)
    
    result =  {'vertices'     : blob['vertices'].reshape((-1,3)),
               'face_normals' : blob['normals'].reshape((-1,3)),
               'faces'        : faces}
    return result
    
def load_stl_ascii(file_obj):
    '''
    Load an ASCII STL file.
    '''
    header = file_obj.readline()
    
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.lower().split('endsolid')[0]
    blob = np.array(text.split())

    # there are 21 'words' in each face
    face_len   = 21
    face_count = len(blob) / face_len
    if (len(blob) % face_len) != 0:
        raise ValueError('Incorrect number of values in STL file!')

    face_count   = int(face_count)
    # this offset is to be added to a fixed set of indices that is tiled
    offset       = face_len * np.arange(face_count).reshape((-1,1))
    normal_index = np.tile([2,3,4], (face_count, 1)) + offset
    vertex_index = np.tile([8,9,10,12,13,14,16,17,18], (face_count, 1)) + offset
    
    # faces are groups of three sequential vertices, as vertices are not references
    faces        = np.arange(face_count*3).reshape((-1,3))
    face_normals = blob[normal_index].astype(float)
    vertices     = blob[vertex_index.reshape((-1,3))].astype(float)

    return {'vertices'     : vertices,
            'faces'        : faces, 
            'face_normals' : face_normals}

_stl_loaders = {'stl':load_stl}

