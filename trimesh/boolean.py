import numpy as np

from .io.stl import load_stl

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
    result = _scad_interface(meshes, operation='difference')
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
    result = _scad_interface(meshes, operation='union')
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
    result = _scad_interface(meshes, operation='intersection')
    return result

def _scad_interface(meshes, operation='difference'):
    '''
    A hacky way to interface with openSCAD which is itself an interface
    to the CGAL CSG bindings. 

    CGAL is very stable if difficult to install/use, so this is a 
    tempfile- happy solution for getting the basic CGAL CSG functionality. 
    '''
    files  = [NamedTemporaryFile(suffix='.STL') for i in meshes]
    script =  NamedTemporaryFile(suffix='.scad')
    result =  NamedTemporaryFile(suffix='.STL')

    # export the meshes to a temporary STL container
    for m, f in zip(meshes, files):
        m.export(f.name, file_type='stl')
    
    # write the SCAD script to do the boolean operation
    script.write(operation + '(){')
    for f in files:
        script.write('import(\"' + f.name + '\");')
    script.write('}')
    script.flush()

    # run the SCAD binary
    check_call(['openscad', 
                script.name,
                '-o', 
                result.name])

    # bring the SCAD result back as a Trimesh object
    result.seek(0)
    result_mesh = load_stl(result)
    
    # close all the freaking temporary files
    result.close()
    script.close()
    for f in files:
        f.close()

    return result_mesh
