import numpy as np
import struct
import json

from ..constants import log
from ..util      import tolist_dict, is_string

#python 3
try:                from cStringIO import StringIO
except ImportError: from io import StringIO

from io import BytesIO

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
    if is_string(file_obj):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'wb')
    file_type = str(file_type).lower()
    
    log.info('Exporting %d faces as %s', len(mesh.faces), file_type.upper())
    export = _mesh_exporters[file_type](mesh, file_obj)
    
    if hasattr(file_obj, 'flush'):
        file_obj.flush()
        file_obj.close()
    else:
        return export

def export_stl(mesh, file_obj=None):
    '''
    Saves a Trimesh object as a binary STL file.
    '''
    
    temp_file = BytesIO()
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
    data = temp_file.read()
    return _write_export(data, file_obj)

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

def export_dict(mesh, file_obj=None):
    if file_obj is not None:
        raise ValueError('Cannot export raw dict to file! Use json!')
    export = {'metadata': tolist_dict(mesh.metadata),
              'faces'   : mesh.faces.tolist(),
              'vertices': mesh.vertices.tolist()}
    return export
        
def export_json(mesh, file_obj=None):
    return _write_export(json.dumps(export_dict(mesh)), 
                         file_obj)

def _write_export(export, file_obj=None):
    '''
    Write a string to a file.
    If file_obj isn't specified, return the string

    Arguments
    ---------
    export: a string of the export data
    file_obj: a file-like object or a filename
    '''
    if hasattr(file_obj, 'write'):
        file_obj.write(export)
    return export

_mesh_exporters = {'stl'  : export_stl,
                   'dict' : export_dict,
                   'json' : export_json,
                   'off'  : export_off,
                   'dae'  : export_collada,
                   'collada': export_collada}
