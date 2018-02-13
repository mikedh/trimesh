import numpy as np
import networkx as nx

from shapely.geometry import Polygon, Point, LineString
from rtree import Rtree
from collections import deque

from .. import bounds
from .. import graph

from ..geometry import medial_axis as _medial_axis
from ..constants import tol_path as tol
from ..constants import log
from ..transformations import transform_points, planar_matrix
from ..util import is_sequence
from .traversal import resample_path


def enclosure_tree(polygons):
    """
    Given a list of shapely polygons, which are the root (aka outermost)
    polygons, and which represent the holes which penetrate the root
    curve. We do this by creating an R-tree for rough collision detection,
    and then do polygon queries for a final result

    Parameters
    -----------
    polygons: (n,) list of shapely.geometry.Polygon objects

    Returns
    -----------
    roots: (m,) int, index of polygons which are root
    contains:  networkx.DiGraph, edges indicate a polygon
               contained by another polygon
    """
    tree = Rtree()
    for i, polygon in enumerate(polygons):
        tree.insert(i, polygon.bounds)
    count = len(polygons)
    contains = nx.DiGraph()
    contains.add_nodes_from(np.arange(count))
    for i in range(count):
        if polygons[i] is None:
            continue
        # we first query for bounding box intersections from the R-tree
        for j in tree.intersection(polygons[i].bounds):
            if (i == j):
                continue
            # we then do a more accurate polygon in polygon test to generate
            # the enclosure tree information
            if polygons[i].contains(polygons[j]):
                contains.add_edge(i, j)
            elif polygons[j].contains(polygons[i]):
                contains.add_edge(j, i)
    roots = [n for n, deg in dict(contains.in_degree()).items() if deg == 0]
    return roots, contains


def edges_to_polygons(edges, vertices):
    """
    Given an edge list of indices and associated vertices
    representing lines, generate a list of polygons.

    Parameters
    -----------
    edges: (n,2) int, indexes of vertices which represent lines
    vertices: (m,2) float, vertex positions

    Returns
    ----------
    polygons: (p,) list of shapely.geometry.Polygon objects
    """

    # create closed polygon objects
    polygons = []

    # loop through a sequence of ordered traversals
    for dfs in graph.dfs_traversals(edges):
        # try to recover polygons before they are more complicated
        try:
            polygons.append(repair_invalid(Polygon(vertices[dfs])))
        except ValueError:
            continue

    # if there is only one polygon, just return it
    if len(polygons) == 1:
        return polygons

    # find which polygons contain which other polygons
    roots, tree = enclosure_tree(polygons)

    # generate list of polygons with proper interiors
    complete = []
    for root in roots:
        interior = list(tree[root].keys())
        shell = polygons[root].exterior.coords
        holes = [polygons[i].exterior.coords for i in interior]
        complete.append(Polygon(shell=shell,
                                holes=holes))
    return complete


def polygons_obb(polygons):
    """
    Find the OBBs for a list of shapely.geometry.Polygons
    """
    rectangles = [None] * len(polygons)
    transforms = [None] * len(polygons)
    for i, p in enumerate(polygons):
        transforms[i], rectangles[i] = polygon_obb(p)
    return np.array(transforms), np.array(rectangles)


def polygon_obb(polygon):
    """
    Find the oriented bounding box of a Shapely polygon.

    The OBB is always aligned with an edge of the convex hull of the polygon.

    Parameters
    -------------
    polygons: shapely.geometry.Polygon

    Returns
    -------------
    transform: (3,3) float, transformation matrix
               which will move input polygon from its original position
               to the first quadrant where the AABB is the OBB
    extents:   (2,) float, extents of transformed polygon
    """
    if hasattr(polygon, 'exterior'):
        points = np.asanyarray(polygon.exterior.coords)
    elif isinstance(polygon, np.ndarray):
        points = polygon
    else:
        raise ValueError('polygon or points must be provided')
    return bounds.oriented_bounds_2D(points)


def transform_polygon(polygon, transform, plot=False):
    if is_sequence(polygon):
        result = [transform_polygon(p, t) for p, t in zip(polygon, transform)]
    else:
        shell = transform_points(np.array(polygon.exterior.coords), transform)
        holes = [transform_points(np.array(i.coords), transform)
                 for i in polygon.interiors]
        result = Polygon(shell=shell, holes=holes)
    if plot:
        plot_polygon(result)
    return result


def rasterize_polygon(polygon, pitch):
    """
    Given a shapely polygon, find the raster representation at a given angle
    relative to the oriented bounding box

    Parameters
    ----------
    polygon: shapely polygon
    pitch:   what is the edge length of a pixel

    Returns
    ----------
    offset:      (2,) float, where the origin of the raster array is located
    grid:        (n,m) bool, where filled areas are True
    grid_points: (p,2) float, points in space
    """

    bounds = np.reshape(polygon.bounds, (2, 2))
    offset = bounds[0]
    shape = np.ceil(np.ptp(bounds, axis=0) / pitch).astype(int)
    grid = np.zeros(shape, dtype=np.bool)

    def fill(ranges):
        ranges = (np.array(ranges) - offset[0]) / pitch
        x_index = np.array([np.floor(ranges[0]),
                            np.ceil(ranges[1])]).astype(int)
        if np.any(x_index < 0):
            return
        grid[x_index[0]:x_index[1], y_index] = True
        if (y_index > 0):
            grid[x_index[0]:x_index[1], y_index - 1] = True

    def handler_multi(geometries):
        for geometry in geometries:
            handlers[geometry.__class__.__name__](geometry)

    def handler_line(line):
        fill(line.xy[0])

    def handler_null(data):
        pass

    handlers = {'GeometryCollection': handler_multi,
                'MultiLineString': handler_multi,
                'MultiPoint': handler_multi,
                'LineString': handler_line,
                'Point': handler_null}

    x_extents = bounds[:, 0] + [-pitch, pitch]
    for y_index in range(grid.shape[1]):
        y = offset[1] + y_index * pitch
        test = LineString(np.column_stack((x_extents, [y, y])))
        hits = polygon.intersection(test)
        handlers[hits.__class__.__name__](hits)

    grid_points = ((np.transpose(np.nonzero(grid)).astype(
        np.float64) * pitch) + offset + (pitch / 2.0))

    return offset, grid, grid_points


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
    if show:
        plt.show()


def plot_raster(raster, pitch, offset=[0, 0]):
    """
    Plot a raster representation.

    raster: (n,m) array of booleans, representing filled/empty area
    pitch:  the edge length of a box from raster, in cartesian space
    offset: offset in cartesian space to the lower left corner of the raster grid
    """
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
        clip = [8, 200]
    # create a sequence of [(n,2)] points
    result = {'shell': resample_boundary(polygon.exterior),
              'holes': deque()}
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
    """
    Given a shapely polygon, find the approximate medial axis based
    on a voronoi diagram of evenly spaced points on the boundary of the polygon.

    Parameters
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
    """
    def contains(points):
        return np.array([polygon.contains(Point(i)) for i in points])

    boundary = resample_boundaries(polygon=polygon,
                                   resolution=resolution,
                                   clip=clip)
    boundary = stack_boundaries(boundary)

    return _medial_axis(samples=boundary,
                        contains=contains)


class InversePolygon:
    """
    Create an inverse polygon.

    The primary use case is that given a point inside a polygon,
    you want to find the minimum distance to the boundary of the polygon.
    """

    def __init__(self, polygon):
        _DIST_BUFFER = .05

        # create a box around the polygon
        bounds = (np.array(polygon.bounds))
        bounds += (_DIST_BUFFER * np.array([-1, -1, 1, 1]))
        coord_ext = bounds[
            np.array([2, 1, 2, 3, 0, 3, 0, 1, 2, 1])].reshape((-1, 2))
        # set the interior of the box to the exterior of the polygon
        coord_int = [np.array(polygon.exterior.coords)]

        # a box with an exterior- shaped hole in it
        exterior = Polygon(shell=coord_ext,
                           holes=coord_int)
        # make exterior polygons out of all of the interiors
        interiors = [Polygon(i.coords) for i in polygon.interiors]

        # save these polygons to a flat list
        self._polygons = np.append(exterior, interiors)

    def distances(self, point):
        """
        Find the minimum distances from a point to the exterior and interiors

        Parameters
        ---------
        point: (2) list or shapely.geometry.Point

        Returns
        ---------
        distances: (n) list of floats
        """
        distances = [i.distance(Point(point)) for i in self._polygons]
        return distances

    def distance(self, point):
        """
        Find the minimum distance from a point to the boundary of the polygon.

        Parameters
        ---------
        point: (2) list or shapely.geometry.Point

        Returns
        ---------
        distance: float
        """
        distance = np.min(self.distances(point))
        return distance


def polygon_hash(polygon):
    """
    An approximate hash of a a shapely Polygon object.

    Parameters
    ---------
    polygon: shapely.geometry.Polygon object

    Returns
    ---------
    hash: (5) length list of hash representing input polygon
    """
    result = [len(polygon.interiors),
              polygon.convex_hull.area,
              polygon.convex_hull.length,
              polygon.area,
              polygon.length]
    return result


def random_polygon(segments=8, radius=1.0):
    """
    Generate a random polygon with a maximum number of sides and approximate radius.

    Parameters
    ---------
    segments: int, the maximum number of sides the random polygon will have
    radius:   float, the approximate radius of the polygon desired

    Returns
    ---------
    polygon: shapely.geometry.Polygon object with random exterior, and no interiors.
    """
    angles = np.sort(np.cumsum(np.random.random(
        segments) * np.pi * 2) % (np.pi * 2))
    radii = np.random.random(segments) * radius
    points = np.column_stack(
        (np.cos(angles), np.sin(angles))) * radii.reshape((-1, 1))
    points = np.vstack((points, points[0]))
    polygon = Polygon(points).buffer(0.0)
    if is_sequence(polygon):
        return polygon[0]
    return polygon


def polygon_scale(polygon):
    """
    For a Polygon object, return the diagonal length of the AABB.

    Parameters
    ------------
    polygon: shapely.geometry.Polygon object

    Returns
    ------------
    scale: float, length of AABB diagonal
    """
    extents = np.reshape(polygon.bounds, (2, 2)).ptp(axis=0)
    scale = (extents ** 2).sum() ** .5

    return scale


def paths_to_polygons(paths, scale=None):
    """
    Given a sequence of connected points turn them into
    valid shapely Polygon objects.

    Parameters
    -----------
    paths: (n,) sequence, of (m,2) float, closed paths
    scale: float, scale of drawing

    Returns
    -----------
    polys: (p,) list of shapely.geometry.Polygons
    valid: (n,) bool, whether input path was valid:
                        valid.sum() == p
    """
    polygons = []
    valid = np.zeros(len(paths), dtype=np.bool)
    for i, path in enumerate(paths):
        if len(path) < 4:
            # since the first and last vertices are identical in
            # a closed loop a 4 vertex path is the minimum for
            # non-zero area
            continue
        try:
            polygons.append(repair_invalid(Polygon(path), scale))
            valid[i] = True
        except ValueError:
            # raised if a polygon is unrecoverable
            continue
    polygons = np.array(polygons)
    return polygons, valid


def repair_invalid(polygon, scale=None, rtol=.5):
    """
    Given a shapely.geometry.Polygon, attempt to return a
    valid version of the polygon.

    Parameters
    -----------
    polygon: shapely.geometry.Polygon object
    rtol:    float, how close does a perimeter have to be
    scale:   float, or None

    Returns
    ----------
    repaired: shapely.geometry.Polygon object

    Raises
    ----------
    ValueError: if polygon can't be repaired
    """
    if hasattr(polygon, 'is_valid') and polygon.is_valid:
        return polygon

    # basic repair involves buffering the polygon outwards
    # this will fix a subset of problems.
    basic = polygon.buffer(tol.zero)
    # if it returned multiple polygons check the largest
    if is_sequence(basic):
        basic = basic[np.argmax([i.area for i in basic])]

    # check perimeter of result agains original perimeter
    if basic.is_valid and np.isclose(basic.length,
                                     polygon.length,
                                     rtol=rtol):
        return basic

    if scale is None:
        distance = tol.buffer * polygon_scale(polygon)
    else:
        distance = tol.buffer * scale

    # if there are no interiors, we can work with just the exterior
    # ring, which is often more reliable
    if len(polygon.interiors) == 0:
        # try buffering the exterior of the polygon
        # the interior will be offset by -tol.buffer
        rings = polygon.exterior.buffer(distance).interiors
        if len(rings) == 1:
            # reconstruct a single polygon from the interior ring
            recon = Polygon(shell=rings[0]).buffer(distance)
            # check perimeter of result agains original perimeter
            if recon.is_valid and np.isclose(recon.length,
                                             polygon.length,
                                             rtol=rtol):
                return recon

    # buffer and unbuffer the whole polygon
    buffered = polygon.buffer(distance).buffer(-distance)
    # if it returned multiple polygons check the largest
    if is_sequence(buffered):
        buffered = buffered[np.argmax([i.area for i in buffered])]
    # check perimeter of result agains original perimeter
    if buffered.is_valid and np.isclose(buffered.length,
                                        polygon.length,
                                        rtol=rtol):
        log.debug('Recovered invalid polygon through double buffering')
        return buffered

    raise ValueError('unable to recover polygon!')
