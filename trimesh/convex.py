'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np

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

    faces  = ConvexHull(mesh.vertices.view(np.ndarray)).simplices
    convex = mesh.__class__(vertices = mesh.vertices.view(np.ndarray).copy(), 
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
    Test whether a mesh is convex or not.
    '''
    test = convex_hull(mesh, clean=False)
    convex = test.faces.shape == mesh.faces.shape
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
