import numpy as np
import struct

#from ..constants import *
    
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

def export_collada(mesh, file_obj=None):
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

    export = template.substitute(replacement)

    return _write_export(export, file_obj)

def export_json(mesh, file_obj=None):
    mesh.verify_vertex_normals()
    # the zeros indicate triangular faces
    indices = np.column_stack((np.zeros(len(mesh.faces), dtype=int), 
                               mesh.faces)).reshape(-1)
    export = {"metadata": {"version": 4,
                           "type": "Geometry"},
              "indices" : indices.tolist(),
              "vertices": mesh.vertices.reshape(-1).tolist(),
              "normals" : mesh.vertex_normals.reshape(-1).tolist()}
    return _write_export(export, file_obj)
        
def _write_export(export, file_obj=None):
    '''
    Write a string to a file, and return the string.
    If file_obj isn't specified, do nothing. 

    Arguments
    ---------
    export: a string of the export data
    file_obj: a file-like object or a filename
    '''

    if file_obj is None:             
        return export
    elif hasattr(file_obj, 'write'): 
        out_file = file_obj
    else: 
        out_file = open(file_obj, 'wb')
    out_file.write(export)
    return export
