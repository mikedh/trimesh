'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np

from .constants import tol

from .util   import type_named, diagonal_dot
from .points import project_to_plane

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
    from scipy.spatial import ConvexHull


    type_trimesh = type_named(mesh, 'Trimesh')
    faces  = ConvexHull(mesh.vertices.view(np.ndarray)).simplices
    convex = type_trimesh(vertices = mesh.vertices.view(np.ndarray).copy(), 
                            faces    = faces,
                            process  = clean)
    if clean:
        # the normals and triangle winding returned by scipy/qhull's
        # ConvexHull are apparently random, so we need to completely fix them
        convex.fix_normals()
        # since we just copied all the vertices over, we will have a bunch
        # of unreferenced vertices, so it is best to remove them
        convex.remove_unreferenced_vertices()
    return convex

def is_convex(mesh):
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
    # triangles from the second column of face adjacency
    triangles = mesh.triangles[mesh.face_adjacency[:,1]]
    # normals and origins from the first column of face adjacency
    normals = mesh.face_normals[mesh.face_adjacency[:,0]]
    origins = mesh.vertices[mesh.face_adjacency_edges[:,0]]

    # reshape and tile everything to be the same dimension
    triangles = triangles.reshape((-1,3))
    normals = np.tile(normals, (1,3)).reshape((-1,3))
    origins = np.tile(origins, (1,3)).reshape((-1,3))
    # project vertices of adjacent triangle onto normal
    dots = diagonal_dot(triangles-origins, normals)
    # if all projections are negative, or 'behind' the triangle
    # the mesh is convex
    convex = (dots < tol.zero).all()    
    return convex

def planar_hull(vertices,
                normal, 
                origin           = [0,0,0], 
                return_transform = False):
    planar , T = project_to_plane(vertices,
                                  plane_normal     = normal,
                                  plane_origin     = origin,
                                  return_transform = True)
    hull_edges = ConvexHull(planar).simplices
    if return_transform:
        return planar[hull_edges], T
    return planar[hull_edges]
