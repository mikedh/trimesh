from .interfaces.scad    import boolean_scad
from .interfaces.blender import boolean_blender

_engines = { None     : boolean_scad,
            'scad'    : boolean_scad,
            'blender' : boolean_blender}

def difference(meshes, engine='scad'):
    '''
    Compute the boolean difference between a mesh an n other meshes.

    Arguments
    ----------
    meshes: list of Trimesh object
    engine: string, which backend to use. 
            valid choices are 'blender' or 'scad'

    Returns
    ----------
    difference: a - (other meshes), **kwargs for a Trimesh
    '''
    result = _engines[engine](meshes, operation='difference')
    return result

def union(meshes, engine='scad'):
    '''
    Compute the boolean union between a mesh an n other meshes.
   
    Arguments
    ----------
    meshes: list of Trimesh object
    engine: string, which backend to use. 
            valid choices are 'blender' or 'scad'

    Returns
    ----------
    union: a + (other meshes), **kwargs for a Trimesh
    '''
    result = _engines[engine](meshes, operation='union')
    return result

def intersection(meshes, engine='scad'):
    '''
    Compute the boolean intersection between a mesh an n other meshes.
   
    Arguments
    ----------
    meshes: list of Trimesh object
    engine: string, which backend to use. 
            valid choices are 'blender' or 'scad'

    Returns
    ----------
    intersection: **kwargs for a Trimesh object of the
                    volume that is contained by all meshes
    '''
    result = _engines[engine](meshes, operation='intersection')
    return result
    
