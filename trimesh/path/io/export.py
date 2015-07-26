import numpy as np

#python 3
try:                from cStringIO import StringIO
except ImportError: from io        import StringIO

def export_path(path, file_obj, file_type=None):
    '''
    Export a Path object to a file- like object, or to a filename

    Arguments
    ---------
    file_obj:  a filename string or a file-like object
    file_type: str representing file type (eg: 'svg')
    process:   boolean flag, whether to process the mesh on load

    Returns:
    mesh: a single Trimesh object, or a list of Trimesh objects, 
          depending on the file format. 
    
    '''
    if ((not hasattr(file_obj, 'read')) and 
        (not file_obj is None)):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'wb')
    export = _path_exporters[file_type](path)
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
    out_file.close()
    return export

_path_exporters = {'svg'  : export_svg,
                   'dict' : export_json}
