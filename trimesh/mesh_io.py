import numpy as np
import struct

from mesh_base import Trimesh
from constants import *

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def available_formats():
    return _MESH_LOADERS.keys()

def load_mesh(file_obj, file_type=None):
    '''
    Load a mesh file into a Trimesh object

    file_obj: a filename string, or a file object
    '''

    if not hasattr(file_obj, 'read'):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')

    mesh = _MESH_LOADERS[file_type](file_obj, file_type)
    file_obj.close()
    
    log.debug('loaded mesh using function %s, containing %i faces', 
             _MESH_LOADERS[file_type].__name__, 
             len(mesh.faces))
             
    return mesh

def load_assimp(file_obj, file_type=None):
    '''
    Use the assimp library to load a mesh, from a file object and type,
    or filename (if file_obj is a string)

    Assimp supports a huge number of mesh formats.

    Performance notes: in tests on binary STL pyassimp was ~10x 
    slower than the native loader included in this package. 
    This is probably due to their recursive prettifying of the data structure.
    
    Also, you need a very recent version of PyAssimp for this function to work 
    (the commit was merged into the assimp github master on roughly 9/5/2014)
    '''

    def LPMesh_to_Trimesh(lp):
        return Trimesh(vertices       = lp.vertices,
                       vertex_normals = lp.normals,
                       faces          = lp.faces)

    if not hasattr(file_obj, 'read'):
        # if there is no read attribute, we assume we've been passed a file name
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')

    scene  = pyassimp.load(file_obj, file_type=file_type)
    meshes = list(map(LPMesh_to_Trimesh, scene.meshes))
    pyassimp.release(scene)

    if len(meshes) == 1: 
        return meshes[0]
    return meshes
 
def load_stl(file_obj, file_type=None):
    if detect_binary_file(file_obj): return load_stl_binary(file_obj)
    else:                            return load_stl_ascii(file_obj)
        
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
    data_ok = (data_end - data_start) == (tri_count * 50)
   
    # this check is to see if this really is a binary STL file. 
    # if we don't do this and try to load a file that isn't structured properly 
    # the struct.unpack call uses 100% memory until the whole thing crashes, 
    # so it's much better to raise an exception here. 
    if not data_ok:
        raise NameError('Attempted to load binary STL with incorrect length in header!')
    
    # all of our vertices will be loaded in order due to the STL format, 
    # so faces are just sequential indices reshaped. 
    faces        = np.arange(tri_count*3).reshape((-1,3))

    # this blob extracts 12 float values, with 2 pad bytes per face
    # the first three floats are the face normal
    # the next 9 are the three vertices 
    blob = np.array(struct.unpack("<" + "12fxx"*tri_count, 
                                  file_obj.read())).reshape((-1,4,3))

    face_normals = blob[:,0]
    vertices     = blob[:,1:].reshape((-1,3))
    
    return Trimesh(vertices     = vertices,
                   faces        = faces, 
                   face_normals = face_normals)

def load_stl_ascii(file_obj):
    '''
    Load an ASCII STL file.
    
    Should be pretty robust to whitespace changes due to the use of split()
    '''
    
    header = file_obj.readline()
    blob   = np.array(file_obj.read().split())
    
    # there are 21 'words' in each face
    face_len     = 21
    face_count   = float(len(blob) - 1) / face_len
    if (face_count % 1) > TOL_ZERO:
        raise NameError('Incorrect number of values in STL file!')
    face_count   = int(face_count)
    # this offset is to be added to a fixed set of indices that is tiled
    offset       = face_len * np.arange(face_count).reshape((-1,1))
    # these hard coded indices will break if the exporter adds unexpected junk
    # but then it wouldn't really be an STL file... 
    normal_index = np.tile([2,3,4], (face_count, 1)) + offset
    vertex_index = np.tile([8,9,10,12,13,14,16,17,18], (face_count, 1)) + offset
    
    # faces are groups of three sequential vertices, as vertices are not references
    faces        = np.arange(face_count*3).reshape((-1,3))
    face_normals = blob[normal_index].astype(float)
    vertices     = blob[vertex_index.reshape((-1,3))].astype(float)
    
    return Trimesh(vertices     = vertices,
                   faces        = faces, 
                   face_normals = face_normals)
                   
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
    mesh     = Trimesh(vertices=vertices, faces=faces)
    return mesh

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
    mesh  = Trimesh(vertices       = data[vid].astype(float),
                    vertex_normals = data[nid].astype(float),
                    faces          = faces)
    mesh.generate_face_normals()

    return mesh
    
def export_stl(mesh, filename):
    '''
    Saves a Trimesh object as a binary STL file.
    '''
    def write_face(file_object, vertices, normal):
        #vertices: (3,3) array of floats
        #normal:   (3) array of floats
        file_object.write(struct.pack('<3f', *normal))
        for vertex in vertices: 
            file_object.write(struct.pack('<3f', *vertex))
        file_object.write(struct.pack('<h', 0))
    if len(mesh.face_normals) == 0: mesh.generate_normals()
    with open(filename, 'wb') as file_object:
        #write a blank header
        file_object.write(struct.pack("<80x"))
        #write the number of faces
        file_object.write(struct.pack("@i", len(mesh.faces)))
        # write the faces
        # TODO: remove the for loop and do this as a single struct.pack operation
        # like we do in the loader, as it is way, way faster.
        for index in range(len(mesh.faces)):
            write_face(file_object, 
                       mesh.vertices[[mesh.faces[index]]], 
                       mesh.face_normals[index])

def export_off(mesh, filename):
    file_obj = open(filename, 'wb')
    file_obj.write('OFF\n')
    file_obj.write(str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n')
    file_obj.close()

    file_obj      = open(filename, 'ab')
    faces_stacked = np.column_stack((np.ones(len(mesh.faces))*3, mesh.faces))
    np.savetxt(file_obj, mesh.vertices, fmt='%.14f')
    np.savetxt(file_obj, faces_stacked, fmt='%i')
    file_obj.close()

def export_collada(mesh, filename):
    '''
    Export a mesh as collada, to filename
    '''
    import os, inspect
    from string import Template
    
    MODULE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    template    = Template(open(os.path.join(MODULE_PATH, 
                                             'templates', 
                                             'collada_template.dae'), 'rb').read())

    # we bother setting this because np.array2string uses these printoptions 
    np.set_printoptions(threshold=np.inf, precision=5, linewidth=np.inf)

    replacement = dict()
    replacement['VERTEX']   = np.array2string(mesh.vertices.reshape(-1))[1:-1]
    replacement['FACES']    = np.array2string(mesh.faces.reshape(-1))[1:-1]
    replacement['NORMALS']  = np.array2string(mesh.vertex_normals.reshape(-1))[1:-1]
    replacement['VCOUNT']   = str(len(mesh.vertices))
    replacement['VCOUNTX3'] = str(len(mesh.vertices) * 3)
    replacement['FCOUNT']   = str(len(mesh.faces))
    with open(filename, 'wb') as outfile:
        outfile.write(template.substitute(replacement))
        
def detect_binary_file(file_obj):
    '''
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    '''
    start  = file_obj.tell()
    fbytes = file_obj.read(1024)
    file_obj.seek(start)
    is_str = isinstance(fbytes, str)
    for fbyte in fbytes:
        if is_str: code = ord(fbyte)
        else:      code = fbyte
        if code > 127: return True
    return False
        
_MESH_LOADERS   = {'stl': load_stl,
                   'off': load_off,
                   'obj': load_wavefront}

_ASSIMP_FORMATS = ['dae', 
                   'blend', 
                   '3ds', 
                   'ase', 
                   'obj', 
                   'ifc', 
                   'xgl', 
                   'zgl',
                   'ply',
                   'lwo',
                   'lxo',
                   'x',
                   'ac',
                   'ms3d',
                   'cob',
                   'scn']
try: 
    import pyassimp
    _MESH_LOADERS.update(zip(_ASSIMP_FORMATS, 
                             [load_assimp]*len(_ASSIMP_FORMATS)))
except:
    log.warn('No pyassimp, only native loaders available!')
        
