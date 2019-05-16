import numpy as np
import json

from ..constants import log
from .. import util

from .wavefront import _obj_exporters
from .urdf import export_urdf  # NOQA
from .gltf import export_glb
from .stl import export_stl, export_stl_ascii
from .ply import _ply_exporters
from .dae import _collada_exporters


def export_mesh(mesh, file_obj, file_type=None, **kwargs):
    """
    Export a Trimesh object to a file- like object, or to a filename

    Parameters
    ---------
    file_obj : str, file-like
      Where should mesh be exported to
    file_type : str or None
      Represents file type (eg: 'stl')

    Returns
    ----------
    exported : bytes or str
      Result of exporter
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

    if isinstance(mesh, (list, tuple, set, np.ndarray)):
        faces = 0
        for m in mesh:
            faces += len(m.faces)
        log.debug('Exporting %d meshes with a total of %d faces as %s',
                  len(mesh), faces, file_type.upper())
    else:
        log.debug('Exporting %d faces as %s', len(mesh.faces),
                  file_type.upper())
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
    mesh : trimesh.Trimesh
      Geometry to export
    digits : int
      Number of digits to include on floats

    Returns
    -----------
    export : str
      OFF format output
    """
    # make sure specified digits is an int
    digits = int(digits)
    # prepend a 3 (face count) to each face
    faces_stacked = np.column_stack((np.ones(len(mesh.faces)) * 3,
                                     mesh.faces)).astype(np.int64)
    export = 'OFF\n'
    # the header is vertex count, face count, another number
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += util.array_to_string(
        mesh.vertices, col_delim=' ', row_delim='\n', digits=digits) + '\n'
    export += util.array_to_string(
        faces_stacked, col_delim=' ', row_delim='\n')
    return export


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
            return util.array_to_encoded(item, dtype=dtype, encoding=encoding)

    # metadata keys we explicitly want to preserve
    # sometimes there are giant datastructures we don't
    # care about in metadata which causes exports to be
    # extremely slow, so skip all but known good keys
    meta_keys = ['units', 'file_name', 'file_path']
    metadata = {k: v for k, v in mesh.metadata.items() if k in meta_keys}

    export = {
        'metadata': metadata,
        'faces': encode(mesh.faces),
        'face_normals': encode(mesh.face_normals),
        'vertices': encode(mesh.vertices)
    }
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


_mesh_exporters = {
    'stl': export_stl,
    'dict': export_dict,
    'json': export_json,
    'off': export_off,
    'glb': export_glb,
    'dict64': export_dict64,
    'msgpack': export_msgpack,
    'stl_ascii': export_stl_ascii
}

_mesh_exporters.update(_ply_exporters)
_mesh_exporters.update(_obj_exporters)
_mesh_exporters.update(_collada_exporters)
