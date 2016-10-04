import numpy as np
import json

from ..constants import log
from .. import util

from .stl import export_stl, export_stl_ascii
from .ply import export_ply

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
    if util.is_string(file_obj):
        if file_type is None:
            file_type = (str(file_obj).split('.')[-1]).lower()        
        if file_type in _mesh_exporters:
            file_obj  = open(file_obj, 'wb')
    file_type = str(file_type).lower()
            
    if not (file_type in _mesh_exporters):
        raise ValueError('%s exporter not available!', file_type)

    log.info('Exporting %d faces as %s', len(mesh.faces), file_type.upper())
    export = _mesh_exporters[file_type](mesh)
    
    if hasattr(file_obj, 'write'):
        file_obj.write(export)
        file_obj.flush()
        file_obj.close()
    else:
        return export

def export_off(mesh):
    '''
    Export a mesh as an OFF file, a simple text format
    
    Arguments
    -----------
    mesh: Trimesh object

    Returns
    -----------
    export: str, string of OFF format output
    '''
    faces_stacked = np.column_stack((np.ones(len(mesh.faces))*3, mesh.faces)).astype(np.int64)
    # numpy arrays to string methods (array2string based ones anyway)
    # are a terrible clusterfuck, so we use our own method which is json dumps based
    export = 'OFF\n'
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += util.array_to_string(mesh.vertices) + '\n'
    export += util.array_to_string(faces_stacked)
    return export

def export_collada(mesh):
    '''
    Export a mesh as a COLLADA file.
    '''
    from ..resources import get_resource

    from string import Template

    template_string = get_resource('collada.dae.template')
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
    return export

def export_dict64(mesh):
    return export_dict(mesh, encoding='base64')

def export_dict(mesh, encoding=None):
    def encode(item, dtype=None):
        if encoding is None:
            return item.tolist()
        else:
            if dtype is None: 
                dtype = item.dtype
            return util.array_to_encoded(item, 
                                    dtype = dtype, 
                                    encoding = encoding)
                                    
    export = {'metadata'     : util.tolist_dict(mesh.metadata),
              'faces'        : encode(mesh.faces),
              'face_normals' : encode(mesh.face_normals),
              'vertices'     : encode(mesh.vertices)}
    return export
        
def export_json(mesh):
    blob = export_dict(mesh, encoding='base64')
    export = json.dumps(blob)
    return export
    
def export_msgpack(mesh):
    import msgpack
    blob = export_dict(mesh, encoding='binary')
    export = msgpack.dumps(blob)
    return export
                         
_mesh_exporters = {'ply'  : export_ply, 
                   'stl'  : export_stl,
                   'dict' : export_dict,
                   'json' : export_json,
                   'off'  : export_off,
                   'dae'  : export_collada,
                   'dict64'  : export_dict64,
                   'msgpack' : export_msgpack,
                   'collada' : export_collada,
                   'stl_ascii' : export_stl_ascii}
