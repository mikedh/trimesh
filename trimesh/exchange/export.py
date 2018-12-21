import numpy as np
import json

from ..constants import log
from .. import util

from .wavefront import _obj_exporters
from .urdf import export_urdf
from .stl import export_stl, export_stl_ascii
from .ply import _ply_exporters


def export_mesh(mesh, file_obj, file_type=None, **kwargs):
    """
    Export a Trimesh object to a file- like object, or to a filename

    Parameters
    ---------
    file_obj: a filename string or a file-like object
    file_type: str representing file type (eg: 'stl')
    process:   boolean flag, whether to process the mesh on load

    Returns:
    mesh: a single Trimesh object, or a list of Trimesh objects,
          depending on the file format.

    """
    # if we opened a file object in this function
    # we will want to close it when we're done
    was_opened = False

    if util.is_string(file_obj):
        if file_type is None:
            file_type = (str(file_obj).split('.')[-1]).lower()
        if file_type in _mesh_exporters:
            was_opened = True
            file_obj = open(file_obj, 'wb')
    file_type = str(file_type).lower()

    if not (file_type in _mesh_exporters):
        raise ValueError('%s exporter not available!', file_type)

    log.debug('Exporting %d faces as %s', len(mesh.faces), file_type.upper())
    export = _mesh_exporters[file_type](mesh, **kwargs)

    if hasattr(file_obj, 'write'):
        result = util.write_encoded(file_obj, export)
    else:
        result = export

    if was_opened:
        file_obj.close()

    return result


def export_off(mesh, digits=10):
    """
    Export a mesh as an OFF file, a simple text format

    Parameters
    -----------
    mesh   : Trimesh object
    digits : int
               number of digits to include on floats

    Returns
    -----------
    export : str
              OFF format output
    """
    # make sure specified digits is an int
    digits = int(digits)
    # prepend a 3 (face count) to each face
    faces_stacked = np.column_stack((
        np.ones(len(mesh.faces)) * 3,
        mesh.faces)).astype(np.int64)
    export = 'OFF\n'
    # the header is vertex count, face count, another number
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += util.array_to_string(mesh.vertices,
                                   col_delim=' ',
                                   row_delim='\n',
                                   digits=digits) + '\n'
    export += util.array_to_string(faces_stacked,
                                   col_delim=' ',
                                   row_delim='\n')
    return export


def export_collada(mesh, digits=8):
    """
    Export a mesh as a COLLADA file.

    Parameters
    --------------
    mesh   : Trimesh object
               Mesh to be exported
    digits : int
              Number of ASCII digits to include for
              floating point variables

    Returns
    ------------
    dae : str
            Mesh as a COLLADA file
    """
    from ..resources import get_resource
    from string import Template

    template_string = get_resource('collada.dae.template')
    template = Template(template_string)

    # keys for template
    replacement = {
        'VERTEX': util.array_to_string(mesh.vertices,
                                       col_delim=' ',
                                       row_delim=' ',
                                       digits=digits),
        'FACES': util.array_to_string(mesh.faces,
                                      col_delim=' ',
                                      row_delim=' ',
                                      digits=digits),
        'NORMALS': util.array_to_string(mesh.vertex_normals,
                                        col_delim=' ',
                                        row_delim=' ',
                                        digits=digits),
        'VCOUNT': str(len(mesh.vertices)),
        'VCOUNTX3': str(len(mesh.vertices) * 3),
        'FCOUNT': str(len(mesh.faces))}
    dae = template.substitute(replacement)
    return dae


def export_dict64(mesh):
    """
    Export a mesh as a dictionary, with data encoded
    to base64.
    """
    return export_dict(mesh, encoding='base64')


def export_dict(mesh, encoding=None):
    """
    Export a mesh to a dict

    Parameters
    ------------
    mesh : Trimesh object
             Mesh to be exported
    encoding : str, or None
                 'base64'

    Returns
    -------------

    """
    def encode(item, dtype=None):
        if encoding is None:
            return item.tolist()
        else:
            if dtype is None:
                dtype = item.dtype
            return util.array_to_encoded(item,
                                         dtype=dtype,
                                         encoding=encoding)
    export = {'metadata': util.tolist(mesh.metadata),
              'faces': encode(mesh.faces),
              'face_normals': encode(mesh.face_normals),
              'vertices': encode(mesh.vertices)}
    if mesh.visual.kind == 'face':
        export['face_colors'] = encode(mesh.visual.face_colors)
    elif mesh.visual.kind == 'vertex':
        export['vertex_colors'] = encode(mesh.visual.vertex_colors)

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


_mesh_exporters = {'stl': export_stl,
                   'dict': export_dict,
                   'json': export_json,
                   'off': export_off,
                   'dae': export_collada,
                   'dict64': export_dict64,
                   'msgpack': export_msgpack,
                   'collada': export_collada,
                   'stl_ascii': export_stl_ascii}

_mesh_exporters.update(_ply_exporters)
_mesh_exporters.update(_obj_exporters)
