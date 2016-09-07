from shapely.geometry import Polygon, Point, LineString
from rtree            import Rtree
from collections      import deque

import numpy as np
import networkx as nx

from .. import bounds

from ..geometry   import medial_axis as _medial_axis
from ..constants  import tol_path as tol
from ..constants  import log
from ..transformations import transform_points, planar_matrix
from ..util       import is_sequence
from .traversal   import resample_path

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
    roots = [n for n, deg in list(g.in_degree().items()) if deg==0]
    return roots, g

def polygons_obb(polygons):
    '''
    Find the OBBs for a list of shapely.geometry.Polygons
    '''
    rectangles = [None] * len(polygons)
    transforms = [None] * len(polygons)
    for i, p in enumerate(polygons):
        transforms[i], rectangles[i] = polygon_obb(p)
    return np.array(transforms), np.array(rectangles)
    
def polygon_obb(polygon):
    '''
    Find the oriented bounding box of a Shapely polygon. 

    The OBB is always aligned with an edge of the convex hull of the polygon.
   
    Arguments
    -------------
    polygons: shapely.geometry.Polygon

    Returns
    -------------
    transform: (3,3) float, transformation matrix
               which will move input polygon from its original position 
               to the first quadrant where the AABB is the OBB
    extents:   (2,) float, extents of transformed polygon
    '''
    points = np.asanyarray(polygon.exterior.coords)
    return bounds.oriented_bounds_2D(points)

def transform_polygon(polygon, transform, plot=False):
    if is_sequence(polygon):
        result = [transform_polygon(p,t) for p,t in zip(polygon, transform)]
    else:
        shell = transform_points(np.array(polygon.exterior.coords), transform)
        holes = [transform_points(np.array(i.coords), transform) for i in polygon.interiors]
        result = Polygon(shell=shell, holes=holes)
    if plot: 
        plot_polygon(result)
    return result

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
    transform            = np.dot(transform, planar_matrix(origin=[0,0], theta=angle))
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
    
def plot_polygon(polygon, show=True):
    import matplotlib.pyplot as plt

    def plot_single(single):
        plt.plot(*single.exterior.xy, color='b')
        for interior in single.interiors:
            plt.plot(*interior.xy, color='r')

    plt.axes().set_aspect('equal', 'datalim')
    if is_sequence(polygon):
        [plot_single(i) for i in polygon]
    else:
        plot_single(polygon)
    if show: plt.show()

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

def resample_boundaries(polygon, resolution, clip=None):
    def resample_boundary(boundary):
        # add a polygon.exterior or polygon.interior to
        # the deque after resampling based on our resolution
        count = boundary.length / resolution
        count = int(np.clip(count, *clip))
        return resample_path(boundary.coords, count=count)
    if clip is None: 
        clip = [8,200]
    # create a sequence of [(n,2)] points
    result = {'shell' : resample_boundary(polygon.exterior),
              'holes' : deque()}
    for interior in polygon.interiors:
        result['holes'].append(resample_boundary(interior))
    result['holes'] = np.array(result['holes'])
    return result

def stack_boundaries(boundaries):
    if len(boundaries['holes']) == 0:
        return boundaries['shell']
    result = np.vstack((boundaries['shell'],
                       np.vstack(boundaries['holes'])))
    return result

def medial_axis(polygon, resolution=.01, clip=None):
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
    def contains(points):
        return np.array([polygon.contains(Point(i)) for i in points])

    boundary = resample_boundaries(polygon=polygon, 
                                  resolution=resolution, 
                                  clip=clip)
    boundary = stack_boundaries(boundary)

    return _medial_axis(samples = boundary,
                        contains = contains)
 
class InversePolygon:
    '''
    Create an inverse polygon. 

    The primary use case is that given a point inside a polygon,
    you want to find the minimum distance to the boundary of the polygon.
    '''
    def __init__(self, polygon):
        _DIST_BUFFER = .05    

        # create a box around the polygon
        bounds   = (np.array(polygon.bounds)) 
        bounds  += (_DIST_BUFFER*np.array([-1,-1,1,1]))
        coord_ext = bounds[np.array([2,1,2,3,0,3,0,1,2,1])].reshape((-1,2))
        # set the interior of the box to the exterior of the polygon
        coord_int = [np.array(polygon.exterior.coords)]
        
        # a box with an exterior- shaped hole in it
        exterior  = Polygon(shell = coord_ext,
                            holes = coord_int)
        # make exterior polygons out of all of the interiors
        interiors = [Polygon(i.coords) for i in polygon.interiors]
        
        # save these polygons to a flat list
        self._polygons = np.append(exterior, interiors)

    def distances(self, point):
        '''
        Find the minimum distances from a point to the exterior and interiors

        Arguments
        ---------
        point: (2) list or shapely.geometry.Point

        Returns
        ---------
        distances: (n) list of floats
        '''
        distances = [i.distance(Point(point)) for i in self._polygons]
        return distances

    def distance(self, point):
        '''
        Find the minimum distance from a point to the boundary of the polygon. 

        Arguments
        ---------
        point: (2) list or shapely.geometry.Point

        Returns
        ---------
        distance: float
        '''
        distance = np.min(self.distances(point))
        return distance

def polygon_hash(polygon):
    '''
    An approximate hash of a a shapely Polygon object.

    Arguments
    ---------
    polygon: shapely.geometry.Polygon object
    
    Returns
    ---------
    hash: (5) length list of hash representing input polygon
    '''
    result = [len(polygon.interiors),
              polygon.convex_hull.area,
              polygon.convex_hull.length,
              polygon.area, 
              polygon.length]
    return result


def random_polygon(segments=8, radius=1.0):
    '''
    Generate a random polygon with a maximum number of sides and approximate radius.

    Arguments
    ---------
    segments: int, the maximum number of sides the random polygon will have
    radius:   float, the approximate radius of the polygon desired

    Returns
    ---------
    polygon: shapely.geometry.Polygon object with random exterior, and no interiors. 
    '''
    angles = np.sort(np.cumsum(np.random.random(segments)*np.pi*2) % (np.pi*2))
    radii  = np.random.random(segments)*radius
    points = np.column_stack((np.cos(angles), np.sin(angles)))*radii.reshape((-1,1))
    points = np.vstack((points, points[0]))
    polygon = Polygon(points).buffer(0.0)
    if is_sequence(polygon):
        return polygon[0]
    return polygon

def polygon_scale(polygon):
    box = np.abs(np.diff(np.reshape(polygon, (2,2)), axis=0))
    scale = box.max()
    return scale
   
def path_to_polygon(path, scale=None):
    try: 
        polygon = Polygon(path)
    except ValueError:
        return None
    return repair_invalid(polygon, scale)
  
def repair_invalid(polygon, scale=None):
    '''
    Given a shapely.geometry.Polygon, attempt to return a 
    valid version of the polygon. If one can't be found, return None
        
    '''
    # if the polygon is already valid, return immediately
    if is_sequence(polygon):
        pass
    elif polygon.is_valid:
        return polygon

    # basic repair involves buffering the polygon outwards
    # this will fix a subset of problems. 
    basic = polygon.buffer(tol.zero)

    if basic.area < tol.zero:
        return None

    if basic.is_valid:
        log.debug('Recovered invalid polygon through zero buffering')
        return basic

    if scale is None:
        scale = polygon_scale(polygon)
        
    buffered   = basic.buffer(tol.buffer * scale)
    unbuffered = buffered.buffer(-tol.buffer * scale)

    if unbuffered.is_valid and not is_sequence(unbuffered):
        log.debug('Recovered invalid polygon through double buffering')
        return unbuffered

    log.warn('Unable to recover polygon! Returning None!')
    return None
