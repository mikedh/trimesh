import numpy as np
import networkx as nx

from shapely.geometry import Polygon, Point
from shapely import vectorized

from rtree import Rtree
from collections import deque

from .. import util
from .. import bounds
from .. import graph

from ..constants import tol_path as tol
from ..constants import log
from ..transformations import transform_points

from .traversal import resample_path


def enclosure_tree(polygons):
    """
    Given a list of shapely polygons with only exteriors,
    find which curves represent the exterior shell or root curve
    and which represent holes which penetrate the exterior.

    This is done with an R-tree for rough overlap detection,
    and then exact polygon queries for a final result.

    Parameters
    -----------
    polygons : (n,) shapely.geometry.Polygon
       Polygons which only have exteriors and may overlap

    Returns
    -----------
    roots : (m,) int
        Index of polygons which are root
    contains : networkx.DiGraph
       Edges indicate a polygon is
       contained by another polygon
    """
    tree = Rtree()
    # nodes are indexes in polygons
    contains = nx.DiGraph()
    for i, polygon in enumerate(polygons):
        # if a polygon is None it means creation
        # failed due to weird geometry so ignore it
        if polygon is None:
            continue
        # insert polygon bounds into rtree
        tree.insert(i, polygon.bounds)
        # make sure every valid polygon has a node
        contains.add_node(i)

    # loop through every polygon
    for i, polygon in enumerate(polygons):
        # if polygon creation failed ignore it
        if polygon is None:
            continue
        # we first query for bounding box intersections from the R-tree
        for j in tree.intersection(polygon.bounds):
            # if we are checking a polygon against itself continue
            if (i == j):
                continue
            # do a more accurate polygon in polygon test
            # for the enclosure tree information
            if polygons[i].contains(polygons[j]):
                contains.add_edge(i, j)
            elif polygons[j].contains(polygons[i]):
                contains.add_edge(j, i)

    # a root or exterior curve has an even number of parents
    # wrap in dict call to avoid networkx view
    degree = dict(contains.in_degree())

    # convert keys and values to numpy arrays
    indexes = np.array(list(degree.keys()))
    degrees = np.array(list(degree.values()))

    # roots are curves with an even inward degree (parent count)
    roots = indexes[(degrees % 2) == 0]

    # if there are multiple nested polygons split the graph
    # so the contains logic returns the individual polygons
    if len(degrees) > 0 and degrees.max() > 1:
        # collect new edges for graph
        edges = []
        # find edges of subgraph for each root and children
        for root in roots:
            children = indexes[degrees == degree[root] + 1]
            edges.extend(contains.subgraph(np.append(children, root)).edges())
        # stack edges into new directed graph
        contains = nx.from_edgelist(edges, nx.DiGraph())
        # if roots have no children add them anyway
        contains.add_nodes_from(roots)

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
    for dfs in graph.traversals(edges, mode='dfs'):
        try:
            # try to recover polygons before they are more complicated
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


def transform_polygon(polygon, matrix):
    """
    Transform a polygon by a a 2D homogenous transform.

    Parameters
    -------------
    polygon : shapely.geometry.Polygon
                 2D polygon to be transformed.
    matrix  : (3, 3) float
                 2D homogenous transformation.

    Returns
    --------------
    result : shapely.geometry.Polygon
                 Polygon transformed by matrix.

    """
    matrix = np.asanyarray(matrix, dtype=np.float64)

    if util.is_sequence(polygon):
        result = [transform_polygon(p, t)
                  for p, t in zip(polygon, matrix)]
        return result
    # transform the outer shell
    shell = transform_points(np.array(polygon.exterior.coords),
                             matrix)[:, :2]
    # transform the interiors
    holes = [transform_points(np.array(i.coords),
                              matrix)[:, :2]
             for i in polygon.interiors]
    # create a new polygon with the result
    result = Polygon(shell=shell, holes=holes)
    return result


def plot_polygon(polygon, show=True):
    """
    Plot a shapely polygon using matplotlib.

    Parameters
    ------------
    polygon: shapely.geometry.Polygon object
    show:    bool, if True will display immediately
    """
    import matplotlib.pyplot as plt

    def plot_single(single):
        plt.plot(*single.exterior.xy, color='b')
        for interior in single.interiors:
            plt.plot(*interior.xy, color='r')
    # make aspect ratio non- stupid
    plt.axes().set_aspect('equal', 'datalim')
    if util.is_sequence(polygon):
        [plot_single(i) for i in polygon]
    else:
        plot_single(polygon)

    if show:
        plt.show()


def resample_boundaries(polygon, resolution, clip=None):
    """
    Return a version of a polygon with boundaries resampled
    to a specified resolution.

    Parameters
    -------------
    polygon:    shapely.geometry.Polygon object
    resolution: float, desired distance between points on boundary
    clip:       (2,) int, upper and lower bounds to clip
                number of samples to (to avoid exploding counts)

    Returns
    ------------
    kwargs: dict, keyword args for a Polygon(**kwargs)
    """
    def resample_boundary(boundary):
        # add a polygon.exterior or polygon.interior to
        # the deque after resampling based on our resolution
        count = boundary.length / resolution
        count = int(np.clip(count, *clip))
        return resample_path(boundary.coords, count=count)
    if clip is None:
        clip = [8, 200]
    # create a sequence of [(n,2)] points
    kwargs = {'shell': resample_boundary(polygon.exterior),
              'holes': deque()}
    for interior in polygon.interiors:
        kwargs['holes'].append(resample_boundary(interior))
    kwargs['holes'] = np.array(kwargs['holes'])
    return kwargs


def stack_boundaries(boundaries):
    """
    Stack the boundaries of a polygon into a single
    (n, 2) list of vertices.

    Parameters
    ------------
    boundaries: dict, with keys 'shell', 'holes'

    Returns
    ------------
    stacked: (n, 2) float, list of vertices
    """
    if len(boundaries['holes']) == 0:
        return boundaries['shell']
    result = np.vstack((boundaries['shell'],
                        np.vstack(boundaries['holes'])))
    return result


def medial_axis(polygon,
                resolution=None,
                clip=None):
    """
    Given a shapely polygon, find the approximate medial axis
    using a voronoi diagram of evenly spaced points on the
    boundary of the polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The source geometry
    resolution : float
        Distance between each sample on the polygon boundary
    clip : None, or (2,) float
        Clip the lower and upper bound of sample count to:
        [minimum number of samples, maximum number of samples]
        specifying a very fine resolution can cause the sample count to
        explode, so clip specifies a minimum and maximum number of samples
        to use per boundary region. To not clip, this can be specified as:
        [0, np.inf]

    Returns
    ----------
    medial : Path2D object
    """
    from scipy.spatial import Voronoi
    from .path import Path2D
    from .io.misc import edges_to_path

    if resolution is None:
        resolution = .01

    # get evenly spaced points on the polygons boundaries
    samples = resample_boundaries(polygon=polygon,
                                  resolution=resolution,
                                  clip=clip)
    # stack the boundary into a (m,2) float array
    samples = stack_boundaries(samples)
    # create the voronoi diagram on 2D points
    voronoi = Voronoi(samples)
    # which voronoi vertices are contained inside the polygon
    contains = vectorized.contains(polygon, *voronoi.vertices.T)
    # ridge vertices of -1 are outside, make sure they are False
    contains = np.append(contains, False)
    # make sure ridge vertices is numpy array
    ridge = np.asanyarray(voronoi.ridge_vertices, dtype=np.int64)
    # only take ridges where every vertex is contained
    edges = ridge[contains[ridge].all(axis=1)]
    # line objects from edges
    medial = Path2D(**edges_to_path(
        edges=edges,
        vertices=voronoi.vertices))

    return medial


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
    Return a vector containing values representitive of
    a particular polygon.

    Parameters
    ---------
    polygon : shapely.geometry.Polygon
      Input geometry

    Returns
    ---------
    hashed: (6), float
      Representitive values representing input polygon
    """
    result = np.array(
        [len(polygon.interiors),
         polygon.convex_hull.area,
         polygon.convex_hull.length,
         polygon.area,
         polygon.length,
         polygon.exterior.length],
        dtype=np.float64)
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
    if util.is_sequence(polygon):
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
    paths : (n,) sequence
        Of (m,2) float, closed paths
    scale: float
        Approximate scale of drawing for precision

    Returns
    -----------
    polys: (p,) list
        shapely.geometry.Polygon
        None
    """
    polygons = [None] * len(paths)
    for i, path in enumerate(paths):
        if len(path) < 4:
            # since the first and last vertices are identical in
            # a closed loop a 4 vertex path is the minimum for
            # non-zero area
            continue
        try:
            polygons[i] = repair_invalid(Polygon(path), scale)
        except ValueError:
            # raised if a polygon is unrecoverable
            continue
        except BaseException:
            log.error('unrecoverable polygon', exc_info=True)
    polygons = np.array(polygons)
    return polygons


def sample(polygon, count, factor=1.5, max_iter=10):
    """
    Use rejection sampling to generate random points inside a
    polygon.

    Parameters
    -----------
    polygon : shapely.geometry.Polygon
                Polygon that will contain points
    count   : int
                Number of points to return
    factor  : float
                How many points to test per loop
                IE, count * factor
    max_iter : int,
                Maximum number of intersection loops
                to run, total points sampled is
                count * factor * max_iter

    Returns
    -----------
    hit : (n, 2) float
           Random points inside polygon
           where n <= count
    """
    bounds = np.reshape(polygon.bounds, (2, 2))
    extents = bounds.ptp(axis=0)

    hit = []
    hit_count = 0
    per_loop = int(count * factor)

    for i in range(max_iter):
        # generate points inside polygons AABB
        points = np.random.random((per_loop, 2))
        points = (points * extents) + bounds[0]

        # do the point in polygon test and append resulting hits
        mask = vectorized.contains(polygon, *points.T)
        hit.append(points[mask])

        # keep track of how many points we've collected
        hit_count += len(hit[-1])

        # if we have enough points exit the loop
        if hit_count > count:
            break

    # stack the hits into an (n,2) array and truncate
    hit = np.vstack(hit)[:count]

    return hit


def repair_invalid(polygon, scale=None, rtol=.5):
    """
    Given a shapely.geometry.Polygon, attempt to return a
    valid version of the polygon through buffering tricks.

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
    if util.is_sequence(basic):
        basic = basic[np.argmax([i.area for i in basic])]

    # check perimeter of result against original perimeter
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
            # check perimeter of result against original perimeter
            if recon.is_valid and np.isclose(recon.length,
                                             polygon.length,
                                             rtol=rtol):
                return recon

    # buffer and unbuffer the whole polygon
    buffered = polygon.buffer(distance).buffer(-distance)
    # if it returned multiple polygons check the largest
    if util.is_sequence(buffered):
        buffered = buffered[np.argmax([i.area for i in buffered])]
    # check perimeter of result against original perimeter
    if buffered.is_valid and np.isclose(buffered.length,
                                        polygon.length,
                                        rtol=rtol):
        log.debug('Recovered invalid polygon through double buffering')
        return buffered

    raise ValueError('unable to recover polygon!')
