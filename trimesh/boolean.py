from . import interfaces

def difference(meshes, engine=None):
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

def union(meshes, engine=None):
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

def intersection(meshes, engine=None):
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
    
def boolean_automatic(meshes, operation):
    if interfaces.blender.exists:
        result = interfaces.blender.boolean(meshes, operation)
    elif interfaces.scad.exists:
        result = interfaces.scad.boolean(meshes, operation)
    else:
        raise ValueError('No backends available for boolean operations!')
    return result

_engines = { None     : boolean_automatic,
            'auto'    : boolean_automatic,
            'scad'    : interfaces.scad.boolean,
            'blender' : interfaces.blender.boolean}
