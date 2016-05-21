'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np

from .constants import tol, log

from .util   import type_named, diagonal_dot
from .points import project_to_plane

try:
    from scipy.spatial import ConvexHull
except ImportError:
    log.warning('Scipy import failed!')

def convex_hull(mesh, clean=True):
    '''
    Get a new Trimesh object representing the convex hull of the 
    current mesh. Requires scipy >.12.
    
    Argments
    --------
    clean: boolean, if True will fix normals and winding
           to be coherent (as qhull/scipy outputs are not)

    Returns
    --------
    convex: Trimesh object of convex hull of current mesh
    '''

    type_trimesh = type_named(mesh, 'Trimesh')
    c = ConvexHull(mesh.vertices.view(np.ndarray).reshape((-1,3)))
    
    vid = np.sort(c.vertices)
    mask = np.zeros(len(c.points), dtype=np.int64)
    mask[vid] = np.arange(len(vid))
        
    faces    = mask[c.simplices]
    vertices = c.points[vid].copy()

    convex = type_trimesh(vertices = vertices, 
                          faces    = faces,
                          process  = True)
    if clean:
        # the normals and triangle winding returned by scipy/qhull's
        # ConvexHull are apparently random, so we need to completely fix them
        convex.fix_normals()
    return convex

def is_convex(mesh, chunks=None):
    '''
    Test if a mesh is convex by projecting the vertices of 
    a triangle onto the normal of its adjacent face.
    
    Arguments
    ----------
    mesh: Trimesh object
    
    Returns
    ----------
    convex: bool, is the mesh convex or not
    '''
    chunk_block = 5e4
    if chunks is None:
        chunks = int(np.clip(len(mesh.faces) / chunk_block,1,10))

    # triangles from the second column of face adjacency
    triangles = mesh.triangles.copy()[mesh.face_adjacency[:,1]]
    # normals and origins from the first column of face adjacency
    normals = mesh.face_normals[mesh.face_adjacency[:,0]]
    origins = mesh.vertices[mesh.face_adjacency_edges[:,0]]

    # reshape and tile everything to be the same dimension
    triangles = triangles.reshape((-1,3))
    normals = np.tile(normals, (1,3)).reshape((-1,3))
    origins = np.tile(origins, (1,3)).reshape((-1,3))
    triangles -= origins

    # in non- convex meshes, we don't necessarily have to compute all 
    # dots of every face since we are looking for logical ALL
    for chunk_tri, chunk_norm in zip(np.array_split(triangles, chunks),
                                     np.array_split(normals,   chunks)):
        # project vertices of adjacent triangle onto normal
        # note that two of these are always going to be zero so we 
        # are doing more dot products than we really have to but finding
        # the index of the the third vertex through graph op is way slower 
        # than doing extra dot products. 
        # there is probably a clever way to use the winding to get this forfree
        dots = diagonal_dot(chunk_tri, chunk_norm)
        # if all projections are negative, or 'behind' the triangle
        # the mesh is convex
        if not bool((dots < tol.merge).all()):
            return False
    return True
        
    
def planar_hull(points, normal, origin=None, input_convex=False):
    '''
    Find the convex outline of a set of points projected to a plane.

    Arguments
    -----------
    points: (n,3) float, input points
    normal: (3) float vector, normal vector of plane
    origin: (3) float, location of plane origin
    input_convex: bool, if True we assume the input points are already from
                  a convex hull which provides a speedup. 

    Returns
    -----------
    hull_lines: (n,2,2) set of unordered line segments
    T:          (4,4) float, transformation matrix 
    '''
    if origin is None:
        origin = np.zeros(3)
    if not input_convex:
        pass
    planar, T = project_to_plane(points,
                                 plane_normal     = normal,
                                 plane_origin     = origin,
                                 return_planar    = False,
                                 return_transform = True)
    hull_edges = ConvexHull(planar[:,0:2]).simplices
    hull_lines = planar[hull_edges]
    planar_z = planar[:,2]
    height = np.array([planar_z.min(),
                       planar_z.max()])
    return hull_lines, T, height
