from . import svg_io
from . import dxf


def export_path(path, file_type, file_obj=None, **kwargs):
    """
    Export a Path object to a file- like object, or to a filename

    Parameters
    ---------
    file_obj:  a filename string or a file-like object
    file_type: str representing file type (eg: 'svg')
    process:   boolean flag, whether to process the mesh on load

    Returns
    ---------
    mesh: a single Trimesh object, or a list of Trimesh objects,
          depending on the file format.

    """
    if ((not hasattr(file_obj, 'read')) and
            (file_obj is not None)):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'wb')
    export = _path_exporters[file_type](path, **kwargs)
    return _write_export(export, file_obj)


def export_dict(path):
    """
    Export a path as a dict of kwargs for the Path constructor.
    """
    export_entities = [e.to_dict() for e in path.entities]
    export_object = {'entities': export_entities,
                     'vertices': path.vertices.tolist()}
    return export_object


def _write_export(export, file_obj=None):
    """
    Write a string to a file.
    If file_obj isn't specified, return the string

    Parameters
    ---------
    export: a string of the export data
    file_obj: a file-like object or a filename
    """

    if file_obj is None:
        return export
    elif hasattr(file_obj, 'write'):
        out_file = file_obj
    else:
        out_file = open(file_obj, 'wb')
    try:
        out_file.write(export)
    except TypeError:
        out_file.write(export.encode('utf-8'))
    out_file.close()
    return export


_path_exporters = {'dxf': dxf.export_dxf,
                   'svg': svg_io.export_svg,
                   'dict': export_dict}
