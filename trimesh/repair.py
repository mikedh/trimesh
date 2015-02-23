import numpy as np
import networkx as nx
from collections import deque

from .geometry  import faces_to_edges
from .grouping  import group_rows
from .triangles import normals
from .constants import *

def _hole_to_faces(hole, vertices=None):
    '''
    Given a loop of vertex indices  representing a hole, turn it into 
    triangular faces.
    If unable to do so, return None

    Arguments
    ---------
    hole:     ordered loop of vertex indices
    vertices: the vertices referenced by hole. 
              If we were to add more involved remeshing algorithms, these 
              would be required, however since we are only adding triangles
              and triangulated quads it is not necessary. 

    Returns
    ---------
    if succesfull: (n, 3) integer vertex indices
    else:          None

    '''
    hole = np.array(hole)
    if len(hole) == 3: 
        return [hole]
    if len(hole) == 4: 
        face_A = hole[[0,1,2]]
        face_B = hole[[2,3,0]]
        return [face_A, face_B]
    log.debug('Cannot fill %d length hole!', len(hole))
    return None

def fill_holes(mesh, raise_watertight=True):
    '''
    Fill single- triangle holes on triangular meshes by adding new triangles
    to fill the holes. New triangles will have proper winding and normals, 
    and if face colors exist the color of the last face will be assigned
    to the new triangles. 
    
    Arguments
    ---------
    mesh: Trimesh object
    raise_watertight: boolean, if True will raise an error if a 
                      watertight mesh cannot be created. 

    '''
    edges        = faces_to_edges(mesh.faces, sort=False)
    edges_sorted = np.sort(edges, axis=1)
    # we know that in a watertight mesh, every edge will be included twice
    # thus, every edge which appears only once is part of the boundary of a hole.
    boundary_groups = group_rows(edges_sorted, require_count=1)

    if len(boundary_groups) < 3: return
    
    boundary_edges  = edges[boundary_groups]
    index_as_dict   = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles. 
    # if the cycle is more than a single face, we could fill it by remeshing
    # but this is not an exact representation of the geometry
    graph  = nx.from_edgelist(np.column_stack((boundary_edges, index_as_dict)))
    cycles = np.array(nx.cycle_basis(graph))

    new_faces = deque()
    for hole in cycles:
        try:
            new_faces.extend(_hole_to_faces(hole))
        except TypeError: 
            if raise_watertight: 
                raise MeshError('Cannot create a watertight mesh!')
    new_faces = np.array(new_faces)

    for face_index, face in enumerate(new_faces):
        # we compare the edge from the new face with 
        # the boundary edge from the source mesh
        edge_test     = face[0:2]
        edge_boundary = edges[graph.get_edge_data(*edge_test)['index']]
        
        # in a well construced mesh, the winding is such that adjacent triangles
        # have reversed edges to each other. Here we check to make sure the 
        # edges are reversed, and if they aren't we simply reverse the face
        reversed = edge_test[0] == edge_boundary[1]
        if not reversed:
            new_faces[face_index] = face[::-1]

    # since the winding is now correct, we can get consistant normals
    # just by doing the cross products on the face edges 
    if len(new_faces) == 0: return

    new_normals, valid = normals(mesh.vertices[new_faces])

    # if face colors exist, assign the last face color to the new faces
    # note that this is a little cheesey, but it is very inexpensive and 
    # is the right thing to do if the mesh is a single color. 
    color_shape = np.shape(mesh.face_colors)
    if len(color_shape) == 2 and color_shape[1] == 3:
        new_colors = np.tile(mesh.face_colors[-1], (np.sum(valid), 1))
        mesh.face_colors = np.vstack((mesh.face_colors, new_colors))

    mesh.faces        = np.vstack((mesh.faces, new_faces[valid]))
    mesh.face_normals = np.vstack((mesh.face_normals, new_normals[valid]))

    log.debug('Filled in mesh with %i triangles', np.sum(valid))
