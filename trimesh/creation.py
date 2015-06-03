'''
Create meshes from primitives, or with operations. 
'''

import numpy as np

from collections import deque

from .base import Trimesh
from .geometry import faces_to_edges
from .grouping import group_rows
from .graph_ops import face_adjacency

from .util import three_dimensionalize

def extrude_polygon(polygon, 
                    height):
    '''
    Turn a shapely.geometry Polygon object and a height (float)
    into a watertight Trimesh object. 
    '''

    def add_vertical_faces(loop):
        loop = three_dimensionalize(loop, return_2D = False)
        
        base = np.array([0, len(loop), 1, 
                         1, len(loop), len(loop)+1])
        new_faces = np.tile(base, (len(loop)-1,  1))
        new_faces += (np.arange(len(loop)-1) * 1).reshape((-1,1))
 
        vertices.append(np.vstack((loop, loop+[0,0,height])))
        faces.append(new_faces.reshape((-1,3)))
 
    vertices, faces = triangulate_polygon(polygon)
    vertices        = three_dimensionalize(vertices, return_2D = False)

    # delaunay appears to deliver consistent winding
    faces    = deque([faces, faces.copy()[:,::-1]])
    vertices = deque([vertices, (vertices.copy() + [0.0, 0, height])])
    
    add_vertical_faces(polygon.exterior.coords)
    for interior in polygon.interiors:
        add_vertical_faces(interior.coords)

    mesh = Trimesh(*append_faces(vertices, faces), process=True)
    mesh.fix_normals()
    return mesh

def triangulate_polygon(polygon):
    '''
    Given a shapely polygon, create a triangulation using meshpy.triangle
    '''
    import meshpy.triangle as triangle

    def round_trip(start, length):
        tiled = np.tile(np.arange(start, start+length).reshape((-1,1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1,2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start):
        vertices.append(np.array(boundary.coords)[:-1])
        facets.append(round_trip(start, len(vertices[-1])))

    vertices = deque()
    facets   = deque()
    start    = len(polygon.exterior.coords) -1

    add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        add_boundary(interior, start)
        start += len(interior.coords) - 1

    vertices    = np.vstack(vertices)
    facets      = np.vstack(facets)
    hole_starts = [i.coords[0] for i in polygon.interiors]

    info = triangle.MeshInfo()
    info.set_points(vertices)
    info.set_facets(facets)
    info.set_holes(hole_starts)
  
    mesh = triangle.build(info)
  
    mesh_vertices = np.array(mesh.points)
    mesh_faces    = np.array(mesh.elements)

    return mesh_vertices, mesh_faces

def append_faces(vertices_seq, faces_seq): 
    '''
    Given a sequence of zero- indexed faces and vertices,
    combine them into a single (n,3) list of faces and (m,3) vertices
    '''
    vertices_len = np.array([len(i) for i in vertices_seq])
    face_offset  = np.append(0, np.cumsum(vertices_len)[:-1])
    
    for offset, faces in zip(face_offset, faces_seq):
        faces += offset

    vertices = np.vstack(vertices_seq)
    faces    = np.vstack(faces_seq)

    return vertices, faces

