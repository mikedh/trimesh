from shapely.geometry import Polygon, LineString
from rtree import Rtree
import numpy as np
import networkx as nx
from collections import deque

from ..geometry import unitize
from .util      import transformation_2D
from .constants import *

def polygons_enclosure_tree(polygons):
    '''
    Given a list of shapely polygons, which are the root (aka outermost)
    polygons, and which represent the holes which penetrate the root
    curve. We do this by creating an R-tree for rough collision detection,
    and then do polygon queries for a final result
    '''
    tree = Rtree()
    for i, polygon in enumerate(polygons):
        tree.insert(i, polygon.bounds)

    count = len(polygons)
    g     = nx.DiGraph()
    g.add_nodes_from(np.arange(count))
    for i in range(count):
        if polygons[i] is None: continue
        #we first query for bounding box intersections from the R-tree
        for j in tree.intersection(polygons[i].bounds):
            if (i==j): continue
            #we then do a more accurate polygon in polygon test to generate
            #the enclosure tree information
            if   polygons[i].contains(polygons[j]): g.add_edge(i,j)
            elif polygons[j].contains(polygons[i]): g.add_edge(j,i)
    roots          = [n for n, deg in list(g.in_degree().items()) if deg==0]
    return roots, g
    
def polygons_obb(polygons):
    '''
    Given a list of polygons, return a list of the minimum area rectangles, and transforms 
    to align the polygons into that minimum area rectangle
    '''
    rectangles = [None] * len(polygons)
    transforms = [None] * len(polygons)
    for i, polygon in enumerate(polygons):
        rectangles[i], transforms[i] = polygon_obb(polygon)
    return np.array(rectangles), np.array(transforms)

def polygon_obb(polygon):
    '''
    Find the oriented bounding box of a Shapely polygon. 

    The OBB is always aligned with an edge of the convex hull of the polygon.
    There is a clever linear time method of finding this called rotating calipers:

    http://cgm.cs.mcgill.ca/~orm/maer.html
    https://stackoverflow.com/questions/13542855/python-help-to-implement-an-algorithm-to-find-the-minimum-area-rectangle-for-gi
    
    http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
    
    https://stackoverflow.com/questions/13542855/python-help-to-implement-an-algorithm-to-find-the-minimum-area-rectangle-for-gi
    
    http://web.cs.swarthmore.edu/~adanner/cs97/s08/pdf/calipers.pdf
   
    This is the less clever n^2 way, because the hull makes n smallish
   
    Arguments
    -------------
    polygons: Shapely polygon

    Returns
    -------------
    size:                 (2) list of edge lengths
    transformation matriz: (3,3) transformation matrix, which will move input
                           polygon from its original position to the origin
                           rotated so fits in axis aligned OBB of size
    '''
    rectangle    = None
    transform    = np.eye(3)
    hull         = np.column_stack(polygon.convex_hull.exterior.xy)
    min_area     = np.inf
    edge_vectors = unitize(np.diff(hull, axis=0))
    perp_vectors = np.fliplr(edge_vectors) * [-1,1]
    for edge_vector, perp_vector in zip(edge_vectors, perp_vectors):
        widths      = np.dot(hull, edge_vector)
        heights     = np.dot(hull, perp_vector)
        rectangle   = np.array([np.ptp(widths), np.ptp(heights)])
        area        = np.prod(rectangle)
        if area < min_area:
            min_area = area
            min_rect = rectangle
            theta    = np.arctan2(*edge_vector[::-1])
            offset   = -np.array([np.min(widths), np.min(heights)])
    rectangle = min_rect
    transform = transformation_2D(offset, theta)
    return rectangle.tolist(), transform.tolist()
    
def transform_polygon(polygon, transform, plot=False):
    '''
    Transform a single shapely polygon, returning a vertex list
    '''
    vertices = np.column_stack(polygon.boundary.xy)
    vertices = np.dot(transform, np.column_stack((vertices, 
                                                  np.ones(len(vertices)))).T)[0:2,:]
    if plot: plt.plot(*vertices)
    return vertices.T
    
def transform_polygons(polygons, transforms, plot=False):
    '''
    Transform a list of Shapely polygons, returning vertex lists. 
    '''
    if plot: plt.axes().set_aspect('equal', 'datalim')
    paths = [None] * len(polygons)
    for i in range(len(polygons)):
        paths[i] = transform_polygon(polygons[i], transforms[i], plot=plot)
    return paths

def rasterize_polygon(polygon, pitch, angle=0, return_points=False):
    '''
    Given a shapely polygon, find the raster representation at a given angle
    relative to the oriented bounding box
    
    Arguments
    ----------
    polygon: shapely polygon
    pitch:   what is the edge length of a pixel
    angle:   what angle to rotate the polygon before rasterization, 
             relative to the oriented bounding box
    
    Returns
    ----------
    rasterized: (n,m) boolean array where filled areas are true
    transform:  (3,3) transformation matrix to move polygon to be covered by
                rasterized representation starting at (0,0)

    '''
    
    rectangle, transform = polygon_obb(polygon)
    transform            = np.dot(transform, transformation_2D(theta=angle))
    vertices             = transform_polygon(polygon, transform)
   
    # after rotating, we want to move the polygon back to the first quadrant
    transform[0:2,2] -= np.min(vertices, axis=0)
    vertices         -= np.min(vertices, axis=0)

    p      = Polygon(vertices)
    bounds = np.reshape(p.bounds, (2,2))
    offset = bounds[0]
    shape  = np.ceil(np.ptp(bounds, axis=0)/pitch).astype(int)
    grid   = np.zeros(shape, dtype=np.bool)

    def fill(ranges):
        ranges  = (np.array(ranges) - offset[0]) / pitch
        x_index = np.array([np.floor(ranges[0]), np.ceil(ranges[1])]).astype(int)
        if np.any(x_index < 0): return
        grid[x_index[0]:x_index[1], y_index] = True
        if (y_index > 0): grid[x_index[0]:x_index[1], y_index-1] = True

    def handler_multi(geometries):
        for geometry in geometries:
            handlers[geometry.__class__.__name__](geometry) 

    def handler_line(line):
        fill(line.xy[0])

    def handler_null(data):
        pass

    handlers = {'GeometryCollection' : handler_multi,
                'MultiLineString'    : handler_multi,
                'MultiPoint'         : handler_multi,
                'LineString'         : handler_line,
                'Point'              : handler_null}
    
    x_extents =  bounds[:,0] + [-pitch, pitch]
    tic = time_function()
    for y_index in range(grid.shape[1]):
        y    = offset[1] + y_index*pitch
        test = LineString(np.column_stack((x_extents, [y,y])))
        hits = p.intersection(test)
        handlers[hits.__class__.__name__](hits)
    toc = time_function()
   
    log.info('Rasterized polygon into %s grid in %f seconds.', str(shape), toc-tic)


    return grid, transform


def grid_polygon(polygon, pitch):
    '''
    Given a shapely polygon, find the raster representation at a given angle
    relative to the oriented bounding box
    
    Arguments
    ----------
    polygon: shapely polygon
    pitch:   what is the edge length of a pixel

    Returns
    ----------
    points:   (n,3) points
    '''
    p      = polygon
    print p.bounds
    bounds = np.reshape(p.bounds, (2,2))
    box    = np.diff(bounds, axis=1)

    def fill(ranges):
        
        for group in np.reshape(ranges, (-1,2)):
            x_members = np.arange(*group, step=pitch)
            y_members = np.ones(len(x_members))*y
            
            points.extend(np.column_stack((x_members, y_members)))
            
    def handler_multi(geometries):
        for geometry in geometries:
            handlers[geometry.__class__.__name__](geometry) 

    def handler_line(line):
        fill(line.xy[0])

    def handler_null(data):
        pass

    handlers = {'GeometryCollection' : handler_multi,
                'MultiLineString'    : handler_multi,
                'MultiPoint'         : handler_multi,
                'LineString'         : handler_line,
                'Point'              : handler_null}
    
    x_extents =  bounds[:,0] + [-pitch, pitch]
    points    = deque()
    for y in np.arange(bounds[0][1], bounds[1][1], pitch):
        test = LineString(np.column_stack((x_extents, [y,y])))
        hits = p.intersection(test)
        handlers[hits.__class__.__name__](hits)
    return np.array(points)
    
def plot_raster(raster, pitch, offset=[0,0]):
    '''
    Plot a raster representation. 

    raster: (n,m) array of booleans, representing filled/empty area
    pitch:  the edge length of a box from raster, in cartesian space
    offset: offset in cartesian space to the lower left corner of the raster grid
    '''
    import matplotlib.pyplot as plt
    plt.axes().set_aspect('equal', 'datalim')
    filled = (np.column_stack(np.nonzero(raster)) * pitch) + offset
    for location in filled:
        plt.gca().add_patch(plt.Rectangle(location, 
                                          pitch, 
                                          pitch, 
                                          facecolor="grey"))
    
def is_ccw(points):
    '''https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    
    '''
    xd = np.diff(points[:,0])
    yd = np.sum(np.column_stack((points[:,1], 
                                 points[:,1])).reshape(-1)[1:-1].reshape((-1,2)), axis=1)
    area = np.sum(xd*yd)*.5
    return area < 0
