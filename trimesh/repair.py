import numpy as np
import networkx as nx
from collections import deque

from .geometry  import faces_to_edges, unitize
from .grouping  import group_rows
from .triangles import normals
from .constants import *

def fill_holes(mesh, raise_watertight=True, fill_planar=False):
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
    graph  = nx.from_edgelist(np.column_stack((boundary_edges, index_as_dict)))
    cycles = np.array(nx.cycle_basis(graph))

    new_faces  = deque()
    new_vertex = deque()
    for hole in cycles:
        # convert the hole, which is a polygon of vertex indices
        # to triangles and new vertices
        faces, vertex = _hole_to_faces(hole        = hole, 
                                       vertices    = mesh.vertices,
                                       fill_planar = fill_planar)
        if len(faces) == 0:
            if raise_watertight: raise MeshError('Cannot create watertight mesh!')
            continue
        # remeshing returns new vertices as negative indices, so change those
        # to absolute indices which won't be screwed up by the later appends
        faces = np.array(faces)
        faces[faces < 0] += len(new_vertex) + len(mesh.vertices) + len(vertex) 
        new_vertex.extend(vertex)
        new_faces.extend(faces)
    new_faces  = np.array(new_faces)
    new_vertex = np.array(new_vertex)

    # no new faces have been added, so nothing further to do
    if len(new_faces) == 0: return

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

    if len(new_vertex) != 0:
        mesh.vertices = np.vstack((mesh.vertices, new_vertex))

    # since the winding is now correct, we can get consistant normals
    # just by doing the cross products on the face edges 
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

def _hole_to_faces(hole, vertices=None, fill_planar=False):
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
    fill_planar: fill larger holes if every vertex of the boundary lies on a plane
                 this will not work reliably on non- convex holes 
    Returns
    ---------
    (n, 3) new faces 
    (m, 3) new vertices
    '''
    hole = np.array(hole)
    # the case where the hole is just a single missing triangle
    if len(hole) == 3: 
        return [hole], []
    # the hole is a quad, which we fill with two triangles
    if len(hole) == 4: 
        face_A = hole[[0,1,2]]
        face_B = hole[[2,3,0]]
        return [face_A, face_B], []
    # if the hole is larger, and the user has selected, we see if all
    # points of the hole lie on a plane, and then create triangles on that plane
    if (fill_planar) and (not vertices is None):
        points  = vertices[hole]
        vectors = np.diff(points, axis=0)
        normals = deque()
        for i in range(len(vectors) - 1):
            unit_normal, valid = unitize(np.cross(vectors[i], 
                                                  vectors[i+1]),
                                         check_valid=True)
            if valid: normals.append(unit_normal)
        normals     = np.array(normals)
        is_coplanar = (np.sum(np.diff(normals, axis=0) ** 2, 
                              axis=1) < TOL_ZERO**2).all()
        if not is_coplanar: return [], []

        # we close the hole polygon by making the last vertex equal the first
        hole       = np.append(hole, hole[0])
        # the new vertex is in the center of the hole
        # this could be bad/stupid if the hole isn't convex
        new_vertex = np.array([np.mean(points, axis=0)])
        # close the hole with triangles from the boundary edge to the 
        # center of the hole 
        # negative indices in faces refer to new vertices,
        # ie -1 = new_vertex[0], -2 = new_vertex[1], etc. 
        new_faces  = np.column_stack((hole[np.arange(len(hole)-1)], 
                                      hole[np.arange(1,len(hole))], 
                                      np.ones(len(hole)-1, dtype=int)*-1))
        return new_faces, new_vertex       
    return [], []
