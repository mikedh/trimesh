import numpy as np
import struct
import json

#python 3
try:                from cStringIO import StringIO
except ImportError: from io import StringIO

def export_mesh(mesh, file_obj, file_type=None):
    '''
    Export a Trimesh object to a file- like object, or to a filename

    Arguments
    ---------
    file_obj: a filename string or a file-like object
    file_type: str representing file type (eg: 'stl')
    process:   boolean flag, whether to process the mesh on load

    Returns:
    mesh: a single Trimesh object, or a list of Trimesh objects, 
          depending on the file format. 
    
    '''

    if ((not hasattr(file_obj, 'read')) and 
        (not file_obj is None)):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'wb')

    export = _mesh_exporters[file_type](mesh, file_obj)
    if not (file_obj is None): 
        file_obj.close()
    return export

def export_stl(mesh, file_obj=None):
    '''
    Saves a Trimesh object as a binary STL file.
    '''
    
    temp_file = StringIO()
    if len(np.shape(mesh.face_normals)) != 2: 
        mesh.generate_normals()

    def write_face(vertices, normal):
        #vertices: (3,3) array of floats
        #normal:   (3) array of floats
        temp_file.write(struct.pack('<3f', *normal))
        for vertex in vertices: 
            temp_file.write(struct.pack('<3f', *vertex))
        temp_file.write(struct.pack('<h', 0))
    
    #write a blank header
    temp_file.write(struct.pack("<80x"))
    #write the number of faces
    temp_file.write(struct.pack("@i", len(mesh.faces)))
    # write the faces
    # TODO: remove the for loop and do this as a single struct.pack operation
    # like we do in the loader, as it is way, way faster.
    for index in range(len(mesh.faces)):
        write_face(mesh.vertices[[mesh.faces[index]]], 
                   mesh.face_normals[index])
    temp_file.seek(0)
    return _write_export(temp_file.read(), file_obj)

def export_off(mesh, file_obj=None):
    export = 'OFF\n'
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    temp_obj = StringIO()
    faces_stacked = np.column_stack((np.ones(len(mesh.faces))*3, mesh.faces))
    np.savetxt(temp_obj, mesh.vertices, fmt='%.14f')
    np.savetxt(temp_obj, faces_stacked, fmt='%i')
    temp_obj.seek(0)
    export += temp_obj.read()
    return _write_export(export, file_obj)

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

    export = json.dumps(export)

    return _write_export(export, file_obj)
        
def _write_export(export, file_obj=None):
    '''
    Write a string to a file.
    If file_obj isn't specified, return the string

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
    return True

_mesh_exporters = {'stl'  : export_stl,
                   'json' : export_json,
                   'dae'  : export_collada,
                   'off'  : export_off}
