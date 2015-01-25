import numpy as np
from shapely.geometry import Point, Polygon, LineString
import time

from .base import Trimesh

from copy import deepcopy

from scipy.spatial import cKDTree
from collections import deque
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

try:    
    import occmodel as occ
    from geotools import Transform
except: 
    log.error('Failed to import occmodel!')

def solid_to_mesh(solid):
    occ_mesh = solid.createMesh()
    occ_mesh.optimize()
    faces    = np.array(list(occ_mesh.triangles)).reshape((-1,3)).astype(int)
    vertices = np.array(list(occ_mesh.vertices)).reshape((-1,3)).astype(float)
    mesh     = Trimesh(vertices=vertices, faces=faces)
    if not mesh.is_watertight(): 
        raise NameError('Mesh returned from openCASCADE isn\'t watertight!')
    return mesh

def three_dimensionalize(points):
    shape = np.shape(points)
    if len(shape) != 2: raise NameError('Points must be 2D array!')
    if shape[1] == 2: 
        points = np.column_stack((points, np.zeros(len(points))))
    elif shape[1] != 3: raise NameError('Points must be (n,2) or (n,3)!')
    return points

def extrude_shapely(polygon, 
                    height, 
                    height_pad   = [0,0], 
                    contour_pad  = 0,
                    return_solid = False):

    shell = np.array(polygon.exterior.coords)
    holes = [np.array(i.coords) for i in polygon.interiors]    
    mesh  = extrude_points(shell        = shell, 
                           holes        = holes, 
                           height       = height, 
                           height_pad   = height_pad,
                           return_solid = return_solid)
    return mesh

def extrude_points(shell, 
                   vectors    = None,
                   height     = None,
                   holes      = None,
                   return_solid = False,
                   height_pad = [0,0]):

    if vectors is None: 
        pad_signed = np.sign(height) * np.array(height_pad)
        vectors    = np.array([[0.0, 0.0, -pad_signed[0]], 
                               [0.0, 0.0,  pad_signed[1] + height]])
    else:
        vectors = np.array(vectors)

    shell_3D      = three_dimensionalize(shell)
    shell_3D[:,2] = vectors[0][2]

    shell_face  = occ.Face().createPolygonal(shell_3D)
    shell_solid = occ.Solid().extrude(shell_face, *vectors)

    if holes is None:
        if return_solid: return shell_solid
        else:            return solid_to_mesh(shell_solid)
 
    pad_vector  = np.diff(vectors, axis=0)[0]
    cut_vectors = vectors + [-pad_vector, pad_vector]

    for hole in holes:
        cut_face  = occ.Face().createPolygonal(three_dimensionalize(hole))
        cut_solid = occ.Solid().extrude(cut_face, *cut_vectors)
        shell_solid.cut(cut_solid)

    if return_solid: return shell_solid
    else:            return solid_to_mesh(shell_solid)

def remove_close(points, radius):
    '''
    Given an (n, m) set of points where n=(2|3) return a list points
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
    
def sample_polygon(polygon, count=None):
    '''
    Randomly sample inside a polygon using rejection sampling,
    returning count number of points
    '''
    if count is None:
        count = len(polygon.exterior.coords) * 2
    bounds   = np.reshape(polygon.bounds, (2,2))
    box_size = np.ptp(bounds, axis=0)
    samples  = deque()
    while len(samples) < count:
        test_point = (np.random.random(2) * box_size) + bounds[0]
        if polygon.intersects(Point(test_point)):
            samples.append(test_point)
    return np.array(samples)
    
def nodes_random(polygon, radius=None, return_index=False):
    points_fixed = np.array(polygon.exterior.coords)
    polygon_idx  = [[0,len(points_fixed)]]
    if len(polygon.interiors) != 0:
        points_interior = np.vstack([i.coords for i in polygon.interiors])
        points_fixed    = np.vstack((points_fixed, points_interior))
  
    if radius is None:
        radius = np.ptp(np.reshape(polygon.bounds, (2,2)),axis=0).min() / 15.0

    # sample random points from inside the polygon with re
    points_random = sample_polygon(polygon)
    points_random = remove_close(points_random, radius)
    points_random = remove_close_set(points_fixed, points_random, radius)

    nodes = np.vstack((points_fixed, points_random))
    return nodes

def delaunay_polygon(polygon, points=None):
    '''
    Calculates Delaunay trianglation of a polygon, 
    discarding triangles which are not inside the polygon
    
    If points is not specified, it is calculated as the exterior of
    the polygon and a number of points randomly sampled from
    inside the polygon
    '''
    if points is None:
        points = nodes_random(polygon)
    faces       = Delaunay(points).simplices.copy()
    test_points = points[faces].mean(axis=1)
    intersects  = np.array([polygon.intersects(Point(i)) for i in test_points])
    faces       = faces[intersects]
    return points, faces
    
if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    level = logging.DEBUG
    handler_stream.setLevel(level)
    for logger in [log, logging.getLogger('trimesh')]:
        logger.addHandler(handler_stream)
        logger.setLevel(level)

    from scipy.spatial import Delaunay
    from shapely.geometry import MultiPoint
    polygon = Point([0,0]).buffer(1).union(Point([.95,0]).buffer(1)).difference(Point([0,0]).buffer(.5))

    points, faces = delaunay_polygon(polygon)
    
    import matplotlib.pyplot as plt
    plt.plot(*polygon.exterior.xy)
    plt.triplot(*points.T, triangles=faces)
    plt.show()
    

    a = Point([0,0]).buffer(1)
    m = extrude_shapely(a, height=1, height_pad=[-.1,.1])
