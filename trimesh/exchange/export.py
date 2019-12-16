import os
import json
import numpy as np

from ..constants import log
from .. import util

from .urdf import export_urdf  # NOQA
from .gltf import export_glb, export_gltf
from .obj import _obj_exporters
from .off import _off_exporters
from .stl import export_stl, export_stl_ascii
from .ply import _ply_exporters
from .dae import _collada_exporters
from .xyz import _xyz_exporters


def export_mesh(mesh, file_obj, file_type=None, **kwargs):
    """
    Export a Trimesh object to a file- like object, or to a filename

    Parameters
    -----------
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

    if util.is_pathlib(file_obj):
        # handle `pathlib` objects by converting to string
        file_obj = str(file_obj.absolute())

    if util.is_string(file_obj):
        if file_type is None:
            # get file type from file name
            file_type = (str(file_obj).split('.')[-1]).lower()
        if file_type in _mesh_exporters:
            was_opened = True
            # get full path of file before opening
            file_path = os.path.abspath(os.path.expanduser(file_obj))
            file_obj = open(file_path, 'wb')

    # make sure file type is lower case
    file_type = str(file_type).lower()

    if not (file_type in _mesh_exporters):
        raise ValueError('%s exporter not available!', file_type)

    if isinstance(mesh, (list, tuple, set, np.ndarray)):
        faces = 0
        for m in mesh:
            faces += len(m.faces)
        log.debug('Exporting %d meshes with a total of %d faces as %s',
                  len(mesh), faces, file_type.upper())
    elif hasattr(mesh, 'faces'):
        # if the mesh has faces log the number
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
    export : dict
      Data stored in dict
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


def scene_to_dict(scene, use_base64=False):
    """
    Export a Scene object as a dict.

    Parameters
    -------------
    scene : trimesh.Scene
      Scene object to be exported

    Returns
    -------------
    as_dict : dict
      Scene as a dict
    """

    # save some basic data about the scene
    export = {'graph': scene.graph.to_edgelist(),
              'geometry': {},
              'scene_cache': {'bounds': scene.bounds.tolist(),
                              'extents': scene.extents.tolist(),
                              'centroid': scene.centroid.tolist(),
                              'scale': scene.scale}}

    # encode arrays with base64 or not
    if use_base64:
        file_type = 'dict64'
    else:
        file_type = 'dict'

    # if the mesh has an export method use it
    # otherwise put the mesh itself into the export object
    for geometry_name, geometry in scene.geometry.items():
        if hasattr(geometry, 'export'):
            # export the data
            exported = {'data': geometry.export(file_type=file_type),
                        'file_type': file_type}
            export['geometry'][geometry_name] = exported
        else:
            # case where mesh object doesn't have exporter
            # might be that someone replaced the mesh with a URL
            export['geometry'][geometry_name] = geometry
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
    'glb': export_glb,
    'gltf': export_gltf,
    'dict64': export_dict64,
    'msgpack': export_msgpack,
    'stl_ascii': export_stl_ascii
}
_mesh_exporters.update(_ply_exporters)
_mesh_exporters.update(_obj_exporters)
_mesh_exporters.update(_off_exporters)
_mesh_exporters.update(_collada_exporters)
_mesh_exporters.update(_xyz_exporters)
