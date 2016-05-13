import numpy as np
import json

from ..constants import log
from ..util      import tolist_dict, is_string, array_to_base64

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

    dtype_stl = np.dtype([('normals',    np.float32, (3)), 
                          ('vertices',   np.float32, (3,3)), 
                          ('attributes', np.uint16)])
    dtype_header = np.dtype([('header', np.void, 80),
                             ('face_count', np.int32)])

    header = np.zeros(1, dtype = dtype_header)
    header['face_count'] = len(mesh.faces)

    packed = np.zeros(len(mesh.faces), dtype=dtype_stl)
    packed['normals']  = mesh.face_normals
    packed['vertices'] = mesh.triangles

    export  = header.tostring()
    export += packed.tostring()
    
    return _write_export(export, file_obj)

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
    Export a mesh as a COLLADA file.
    '''
    from ..templates import get_template
    from string import Template

    template_string = get_template('collada.dae.template')
    template = Template(template_string)

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
    result = _write_export(export, file_obj)
    return result

def export_dict64(mesh, file_obj=None):
    if file_obj is not None:
        raise ValueError('Cannot export raw dict to file! Use json!')
    export = {'metadata'     : tolist_dict(mesh.metadata),
              'faces'        : array_to_base64(mesh.faces,        np.uint32),
              'face_normals' : array_to_base64(mesh.face_normals, np.float32),
              'vertices'     : array_to_base64(mesh.vertices,     np.float32)}
    return export

def export_dict(mesh, file_obj=None):
    if file_obj is not None:
        raise ValueError('Cannot export raw dict to file! Use json!')
    export = {'metadata'     : tolist_dict(mesh.metadata),
              'faces'        : mesh.faces.tolist(),
              'face_normals' : mesh.face_normals.tolist(),
              'vertices'     : mesh.vertices.tolist()}
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
                   'dict64' : export_dict64,
                   'json' : export_json,
                   'off'  : export_off,
                   'dae'  : export_collada,
                   'collada': export_collada}
