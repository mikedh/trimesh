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
    # create a 2D triangulation of the shapely polygon
    vertices, faces = triangulate_polygon(polygon)
    
    # stack the (n,3) faces into (3*n, 2) edges
    edges        = faces_to_edges(faces, sort=True)
    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # shapely polygon, we need to find it again
    edges_unique = group_rows(edges, require_count = 1)

    # (n, 2, 2) set of line segments (positions, not references)
    boundary = vertices[edges[edges_unique]]

    # we are creating two vertical  triangles for every 2D line segment
    # on the boundary of the 2D triangulation
    vertical = np.tile(boundary.reshape((-1,2)), 2).reshape((-1,2))
    vertical = np.column_stack((vertical, 
                                np.tile([0,height,0,height], len(boundary))))
    vertical_faces  = np.tile([0,1,2,2,1,3], (len(boundary), 1))
    vertical_faces += np.arange(len(boundary)).reshape((-1,1)) * 4
    vertical_faces  = vertical_faces.reshape((-1,3))

    # stack the (n,2) vertices with zeros to make them (n, 3)
    vertices_3D  = three_dimensionalize(vertices, return_2D = False)
    
    # a sequence of zero- indexed faces, which will then be appended
    # with offsets to create the final mesh
    faces_seq    = [faces, faces.copy()[:,::-1], vertical_faces]
    vertices_seq = [vertices_3D, (vertices_3D.copy() + [0.0, 0, height]), vertical]

    mesh = Trimesh(*append_faces(vertices_seq, faces_seq), process=True)
    # the winding and normals of our mesh are arbitrary, although now we are
    # watertight so we can traverse the mesh and fix winding and normals 
    mesh.fix_normals()
    return mesh

def triangulate_polygon(polygon):
    '''
    Given a shapely polygon, create a triangulation using meshpy.triangle
    '''
    import meshpy.triangle as triangle

    def round_trip(start, length):
        '''
        Given a start index and length, create a series of (n, 2) edges which
        create a closed traversal. 

        Example:
        start, length = 0, 3
        returns:  [(0,1), (1,2), (2,0)]
        '''
        tiled = np.tile(np.arange(start, start+length).reshape((-1,1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1,2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start):
        vertices.append(np.array(boundary.coords)[:-1])
        facets.append(round_trip(start, len(vertices[-1])))

    #sequence of (n,2) points in space
    vertices = deque()
    #sequence of (n,2) indices of vertices
    facets   = deque()
    # shapely polygons have a duplicate first and last vertex
    start    = len(polygon.exterior.coords) - 1

    add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        add_boundary(interior, start)
        start += len(interior.coords) - 1

    # create clean (n,2) float array of vertices
    # and (m, 2) int array of facets
    # by stacking the sequence of (p,2) arrays
    vertices    = np.vstack(vertices)
    facets      = np.vstack(facets)
    # hole starts in meshpy lingo are a (h, 2) list of (x,y) points
    # which are on the boundary of a hole
    hole_starts = [i.coords[0] for i in polygon.interiors]

    info = triangle.MeshInfo()
    info.set_points(vertices)
    info.set_facets(facets)
    info.set_holes(hole_starts)
    # setting the minimum angle to 30 degrees produces noticeable nicer
    # meshes than if left unset
    mesh = triangle.build(info, min_angle=30)
  
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

