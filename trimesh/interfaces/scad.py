import os
import platform

from .generic import MeshScript
from ..constants import log

from distutils.spawn import find_executable

_search_path = os.environ['PATH']
if platform.system() == 'Windows':
    # split existing path by delimiter
    _search_path = [i for i in _search_path.split(';') if len(i) > 0]
    _search_path.append(os.path.normpath('C:\Program Files\OpenSCAD'))
    _search_path.append(os.path.normpath('C:\Program Files (x86)\OpenSCAD'))
    _search_path = ';'.join(_search_path)
    log.debug('searching for scad in: ', _search_path)


_scad_executable = find_executable('openscad', path=_search_path)
exists = _scad_executable is not None


def interface_scad(meshes, script):
    '''
    A way to interface with openSCAD which is itself an interface
    to the CGAL CSG bindings.
    CGAL is very stable if difficult to install/use, so this function provides a
    tempfile- happy solution for getting the basic CGAL CSG functionality.

    Parameters
    ---------
    meshes: list of Trimesh objects
    script: string of the script to send to scad.
            Trimesh objects can be referenced in the script as
            $mesh_0, $mesh_1, etc.
    '''
    if not exists:
        raise ValueError('No SCAD available!')
    with MeshScript(meshes=meshes, script=script) as scad:
        result = scad.run(_scad_executable + ' $script -o $mesh_post')
    return result


def boolean(meshes, operation='difference'):
    '''
    Run an operation on a set of meshes
    '''
    script = operation + '(){'
    for i in range(len(meshes)):
        script += 'import(\"$mesh_' + str(i) + '\");'
    script += '}'
    return interface_scad(meshes, script)
