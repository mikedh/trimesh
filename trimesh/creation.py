'''
Create meshes from primitives, or with operations. 
'''

import numpy as np

from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import cKDTree, Delaunay
from collections import deque
from copy import deepcopy

from .base import Trimesh
from .constants import *
from .util import grid_arange_2D, three_dimensionalize

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
 
    vertices, faces = delaunay_polygon(polygon)
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

def nodes_grid(polygon, grid_count=10.0):
    '''
    For a shapely polygon, generate a set of vertices.

    The strategy used in this function is to take all of the vertices 
    from the boundary (interior and exterior), and stack those points 
    with points taken on a grid inside the area. 

    This is anecdotally the same thing solidworks does, and it is nice 
    because it is easy while eliminating excessively long thin triangles.

    Arguments
    ---------
    polygon: shapely.geometry Polygon object

    Returns
    ---------
    nodes: (n,2) vertices
    '''
    boundary = np.array(polygon.exterior.coords)
    if len(polygon.interiors) != 0:
        interiors = np.vstack([i.coords for i in polygon.interiors])
        boundary  = np.vstack((boundary, interiors))
    
    bounds     = np.reshape(polygon.bounds, (2,2))
    radius     = np.ptp(bounds,axis=0).min() / grid_count
    bounds[1] += radius

    # create a (n, 2) grid from the bounds of the polygon
    grid      = grid_arange_2D(bounds, radius)
    # see if the points on the grid are contained by the polygon
    contained = np.array([polygon.contains(Point(i)) for i in grid])
    # remove any point on the grid close to a boundary point 
    # or not inside polygon 
    grid_cull = remove_close_set(boundary, grid[contained], radius)
    # the nodes (AKA vertices) of the mesh are the boundary points
    # and the culled grid points 
    nodes     = np.vstack((boundary, grid_cull))

    return nodes

def delaunay_polygon(polygon):
    '''
    Calculates Delaunay trianglation of a polygon, 
    discarding triangles which are not inside the polygon
    
    If points is not specified, it is calculated as the exterior of
    the polygon and a number of points randomly sampled from
    inside the polygon
    '''
    points      = nodes_grid(polygon)
    faces       = Delaunay(points).simplices.copy()
    test_points = points[faces].mean(axis=1)
    intersects  = np.array([polygon.intersects(Point(i)) for i in test_points])
    faces       = faces[intersects]
    return points, faces

def remove_close(points, radius):
    '''
    Given an (n, m) set of points where n=(2|3) return a list of points
    where no point is closer than radius
    '''
    tree     = cKDTree(points)
    consumed = np.zeros(len(points), dtype=np.bool)
    unique   = np.zeros(len(points), dtype=np.bool)
    for i in xrange(len(points)):
        if consumed[i]: continue
        neighbors = tree.query_ball_point(points[i], r=radius)
        consumed[neighbors] = True
        unique[i]           = True
    return points[unique]

def remove_close_set(points_fixed, points_reduce, radius):
    '''
    Given two sets of points and a radius, return a set of points
    that is the subset of points_reduce where no point is within 
    radius of any point in points_fixed
    '''
    tree_fixed  = cKDTree(points_fixed)
    tree_reduce = cKDTree(points_reduce)
    reduce_duplicates = tree_fixed.query_ball_tree(tree_reduce, r = radius)
    reduce_duplicates = np.unique(np.hstack(reduce_duplicates).astype(int))
    reduce_mask = np.ones(len(points_reduce), dtype=np.bool)
    reduce_mask[reduce_duplicates] = False
    points_clean = points_reduce[reduce_mask]
    return points_clean
