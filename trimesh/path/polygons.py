from shapely.geometry import Polygon, Point, LineString
from rtree import Rtree
import numpy as np
import networkx as nx
from collections import deque

from ..points    import unitize
from ..util      import transformation_2D, is_sequence, is_ccw
from .constants  import *

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
    
def polygon_obb(polygon):
    '''
    Find the oriented bounding box of a Shapely polygon. 

    The OBB is always aligned with an edge of the convex hull of the polygon.
   
    Arguments
    -------------
    polygons: shapely.geometry.Polygon or list of Polygons

    Returns
    -------------
    size:                  (2) or (len(polygons), 2) list of edge lengths
    transformation matriz: (3,3) or (len(polygons, 3,3) transformation matrix
                           which will move input polygon from its original position 
                           to the first quadrant where the AABB is the OBB
    '''
    if is_sequence(polygon): return _polygons_obb(polygon)
    else:                    return _polygon_obb(polygon)

def _polygons_obb(polygons):
    '''
    Find the OBBs for a list of shapely.geometry.Polygons
    '''
    rectangles = [None] * len(polygons)
    transforms = [None] * len(polygons)
    for i, polygon in enumerate(polygons):
        rectangles[i], transforms[i] = polygon_obb(polygon)
    return np.array(rectangles), np.array(transforms)

def _polygon_obb(polygon):
    '''
    Find the OBB for a single shapely.geometry.Polygon
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
        x_index = np.array([np.floor(ranges[0]), 
                            np.ceil(ranges[1])]).astype(int)
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
    
    x_extents = bounds[:,0] + [-pitch, pitch]
 
    for y_index in range(grid.shape[1]):
        y    = offset[1] + y_index*pitch
        test = LineString(np.column_stack((x_extents, [y,y])))
        hits = p.intersection(test)
        handlers[hits.__class__.__name__](hits)

    log.info('Rasterized polygon into %s grid', str(shape))
    return grid, transform
    
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
    
def resample_loop(points, count):
    '''
    Given a loop of (n,d) points, resample them such that the
    distance traversed along the loop is constant in between each 
    of the resampled points.

    Arguments
    ----------
    points:   (n,d) sequence of points in space
    count:    number of evenly spaced points to find

    Returns
    ----------
    resampled: (count,d) set of points evenly spaced on the perimeter
    '''
    points     = np.array(points)
    # find the direction of each segment
    vectors    = np.diff(points, axis=0)
    norms      = np.linalg.norm(vectors, axis=1)
    unit_vec   = vectors/norms.reshape((-1,1))
    perimeter  = norms.sum()
    # cumulative sum of section length increasing 
    cum_norm   = np.cumsum(norms)
    # we discard the last sample so searchsorted doesn't
    # return indices outside the range
    samples    = np.linspace(0, perimeter, count+1)[:-1]
    # return the indices in cum_norm that each sample would
    # need to be inserted at to maintain the sorted property
    positions  = np.searchsorted(cum_norm, samples)
    offsets    = np.append(0, cum_norm)[positions]

    # the distance past the reference vertex we need to travel
    projection = samples - offsets
    # find out which dirction we need to project
    direction  = unit_vec[positions]
    # find out which vertex we're offset from
    origin     = points[positions]
    
    # just the parametric equation for a line
    resampled = origin + (direction*projection.reshape((-1,1)))
    
    return resampled

def medial_axis(polygon, resolution=.01, clip=[10,1000]):
    '''
    Given a shapely polygon, find the approximate medial axis based
    on a voronoi diagram of evenly spaced points on the boundary of the polygon.

    Arguments
    ----------
    polygon:    a shapely.geometry.Polygon 
    resolution: target distance between each sample on the polygon boundary
    clip:       [minimum number of samples, maximum number of samples]
                specifying a very fine resolution can cause the sample count to
                explode, so clip specifies a minimum and maximum number of samples
                to use per boundary region. To not clip, this can be specified as:
                [0, np.inf]

    Returns
    ----------
    lines:     (n,2,2) set of line segments
    '''
    def add_boundary(boundary):
        # add a polygon.exterior or polygon.interior to
        # the deque after resampling based on our resolution
        count     = boundary.length / resolution
        count     = int(np.clip(count, *clip))
        points.append(resample_loop(boundary.coords, count))

    # do the import here to avoid it in general use and fail immediatly
    # if we don't have scipy.spatial available
    from scipy.spatial import Voronoi

    # create a sequence of [(n,2)] points
    points = deque()
    add_boundary(polygon.exterior)
    for interior in polygon.interiors:
        add_boundary(interior)

    # create the voronoi diagram, after vertically stacking the points
    # deque from a sequnce into a clean (m,2) array
    voronoi   = Voronoi(np.vstack(points))
    # which voronoi vertices are contained inside the original polygon
    contained = np.array([polygon.contains(Point(i)) for i in voronoi.vertices])
    ridge     = np.reshape(voronoi.ridge_vertices, -1)
    # for the medial axis, we only want to include vertices that are inside
    # the original polygon, and are greater than zero
    # negative indices indicate a vornoi vertex outside the diagram
    test      = np.logical_and(contained[ridge], 
                               ridge >= 0).reshape((-1,2)).all(axis=1)
    ridge     = ridge.reshape((-1,2))[test]
    # index into lines, which are (n,2,2)
    lines     = voronoi.vertices[ridge]
    return lines


