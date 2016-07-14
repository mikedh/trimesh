import numpy as np

from . import transformations
from . import util
from . import points

def transform(mesh, matrix=None):
    '''
    Return a permutated variant of a mesh by randomly reording faces 
    and rotatating + translating a mesh by a random matrix.
    
    Arguments
    ----------
    mesh:   Trimesh object (will not be altered by this function)
    matrix: (4,4) float, transformation matrix. Optional, if not 
             passed a random matrix will be used.
             
    Returns
    ----------
    permutated: Trimesh object, same faces as input mesh but
                rotated and reordered.
    '''
    if matrix is None:
        matrix = transformations.random_rotation_matrix()
        matrix[0:3,3] = np.random.random(3)*1000
    else:
        matrix = np.asanyarray(matrix)
        if matrix.shape != (4,4):
            raise ValueError('Matrix must be (4,4)!')

    triangles = np.random.permutation(mesh.triangles).reshape((-1,3))
    triangles = points.transform_points(triangles, matrix)

    mesh_type = util.type_named(mesh, 'Trimesh')
    permutated = mesh_type(vertices = triangles,
                           faces    = np.arange(len(mesh.faces)*3).reshape((-1,3)))

    return permutated
    
def noise(mesh, magnitude=None):
    '''
    Add gaussian noise to every vertex of a mesh.
    Makes no effort to maintain topology or sanity.
    
    Arguments
    ----------
    mesh:      Trimesh object (will not be mutated)
    magnitude: float, what is the maximum distance per axis we can displace a vertex. 
               Default value is mesh.scale/100.0
           
    Returns
    ----------
    permutated: Trimesh object, input mesh with noise applied
    '''
    if magnitude is None:
        magnitude = mesh.scale / 100.0
    
    # make sure we've re- ordered faces randomly before adding noise
    faces = np.random.permutation(mesh.faces)
    vertices = mesh.vertices + ((np.random.random(mesh.vertices.shape) - .5) * magnitude)
    
    mesh_type = util.type_named(mesh, 'Trimesh')
    permutated = mesh_type(vertices = vertices,
                           faces    = faces)
    return permutated
    
def facets(mesh):
    '''
    Randomly remesh facets of mesh and reorder faces.
    Will produce a mesh that has the exact surface area and
    volume of the input but with a different tesselation, assuming
    the input mesh does in fact have facets (coplanar groups of faces).
    
    Arguments
    ----------
    mesh: Trimesh object
    
    Returns
    ----------
    permutated: Trimesh object with remeshed facets
    '''
    pass
    
