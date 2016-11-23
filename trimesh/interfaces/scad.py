from .generic import MeshScript
from distutils.spawn import find_executable

_scad_executable = find_executable('openscad')
exists = _scad_executable is not None


def interface_scad(meshes, script):
    '''
    A way to interface with openSCAD which is itself an interface
    to the CGAL CSG bindings.
    CGAL is very stable if difficult to install/use, so this function provides a
    tempfile- happy solution for getting the basic CGAL CSG functionality.

    Arguments
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
