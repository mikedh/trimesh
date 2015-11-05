import numpy as np

from .io.stl import load_stl

from string import Template

from tempfile   import NamedTemporaryFile
from subprocess import check_call

def difference(a, *args):
    '''
    Compute the boolean difference between a mesh an n other meshes.

    Arguments
    ----------
    a:    Trimesh object 
    args: Trimesh object or list of Trimesh objects

    Returns
    ----------
    difference: a - (other meshes), **kwargs for a Trimesh
    '''
    meshes = np.append(a,args)
    result = _scad_operation(meshes, operation='difference')
    return result

def union(a, *args):
    '''
    Compute the boolean union between a mesh an n other meshes.
   
    Arguments
    ----------
    a:    Trimesh object 
    args: Trimesh object or list of Trimesh objects

    Returns
    ----------
    union: a + (other meshes), **kwargs for a Trimesh
    '''

    meshes = np.append(a, args)
    result = _scad_operation(meshes, operation='union')
    return result

def intersection(a, *args):
    '''
    Compute the boolean intersection between a mesh an n other meshes.
   
    Arguments
    ----------
    a:    Trimesh object 
    args: Trimesh object or list of Trimesh objects

    Returns
    ----------
    intersection: **kwargs for a Trimesh object of the
                    volume that is contained by all meshes
    '''

    meshes = np.append(a,args)
    result = _scad_operation(meshes, operation='intersection')
    return result
    
def scad_interface(meshes, script):
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
    mesh_out   = [NamedTemporaryFile(suffix='.STL') for i in meshes]
    mesh_in    =  NamedTemporaryFile(suffix='.STL')
    script_out =  NamedTemporaryFile(suffix='.scad')

    # export the meshes to a temporary STL container
    for m, f in zip(meshes, mesh_out):
        m.export(f.name, file_type='stl')
    
    replacement = {'mesh_' + str(i) : m.name for i,m in enumerate(mesh_out)}
    script_text = Template(script).substitute(replacement)
    script_out.write(script_text.encode('utf-8'))
    script_out.flush()

    # run the SCAD binary
    check_call(['openscad', 
                script_out.name,
                '-o', 
                mesh_in.name])

    # bring the SCAD result back as a Trimesh object
    mesh_in.seek(0)
    mesh_result = load_stl(mesh_in)
    
    # close all the freaking temporary files
    mesh_in.close()
    script_out.close()
    for f in mesh_out:
        f.close()

    return mesh_result

def _scad_operation(meshes, operation='difference'):
    '''
    Run an operation on a set of meshes
    '''
    script = operation + '(){'
    for i in range(len(meshes)):
        script += 'import(\"$mesh_' + str(i) + '\");'
    script += '}'
    return scad_interface(meshes, script)
