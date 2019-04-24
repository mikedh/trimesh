"""
points.py
-------------

Functions dealing with (n, d) points.
"""
import copy
import warnings

import numpy as np

from .constants import tol
from .geometry import plane_transform
from .parent import Geometry

from . import util
from . import caching
from . import grouping
from . import transformations


def point_plane_distance(points,
                         plane_normal,
                         plane_origin=[0.0, 0.0, 0.0]):
    """
    The minimum perpendicular distance of a point to a plane.

    Parameters
    -----------
    points:       (n, 3) float, points in space
    plane_normal: (3,) float, normal vector
    plane_origin: (3,) float, plane origin in space

    Returns
    ------------
    distances:     (n,) float, distance from point to plane
    """
    points = np.asanyarray(points, dtype=np.float64)
    w = points - plane_origin
    distances = np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal)
    return distances


def major_axis(points):
    """
    Returns an approximate vector representing the major axis of points

    Parameters
    -------------
    points: (n, dimension) float, points in space

    Returns
    -------------
    axis: (dimension,) float, vector along approximate major axis
    """
    U, S, V = np.linalg.svd(points)
    axis = util.unitize(np.dot(S, V))
    return axis


def plane_fit(points):
    """
    Given a set of points, find an origin and normal using SVD.

    Parameters
    ---------
    points : (n,3) float
        Points in 3D space

    Returns
    ---------
    C : (3,) float
        Point on the plane
    N : (3,) float
        Normal vector of plane
    """
    # make sure input is numpy array
    points = np.asanyarray(points, dtype=np.float64)
    # make the plane origin the mean of the points
    C = points.mean(axis=0)
    # points offset by the plane origin
    x = points - C
    # create a (3, 3) matrix
    M = np.dot(x.T, x)
    # run SVD
    N = np.linalg.svd(M)[0][:, -1]

    return C, N


def radial_sort(points,
                origin,
                normal):
    """
    Sorts a set of points radially (by angle) around an
    origin/normal.

    Parameters
    --------------
    points: (n,3) float, points in space
    origin: (3,)  float, origin to sort around
    normal: (3,)  float, vector to sort around

    Returns
    --------------
    ordered: (n,3) flot, re- ordered points in space
    """

    # create two axis perpendicular to each other and the normal,
    # and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)

    # calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)

    # return the points sorted by angle
    return points[[np.argsort(angles)]]


def project_to_plane(points,
                     plane_normal=[0, 0, 1],
                     plane_origin=[0, 0, 0],
                     transform=None,
                     return_transform=False,
                     return_planar=True):
    """
    Projects a set of (n,3) points onto a plane.

    Parameters
    ---------
    points:           (n,3) array of points
    plane_normal:     (3) normal vector of plane
    plane_origin:     (3) point on plane
    transform:        None or (4,4) matrix. If specified, normal/origin are ignored
    return_transform: bool, if true returns the (4,4) matrix used to project points
                      onto a plane
    return_planar:    bool, if True, returns (n,2) points. If False, returns
                      (n,3), where the Z column consists of zeros
    """

    if np.all(np.abs(plane_normal) < tol.zero):
        raise NameError('Normal must be nonzero!')

    if transform is None:
        transform = plane_transform(plane_origin, plane_normal)

    transformed = transformations.transform_points(points, transform)
    transformed = transformed[:, 0:(3 - int(return_planar))]

    if return_transform:
        polygon_to_3D = np.linalg.inv(transform)
        return transformed, polygon_to_3D
    return transformed


def remove_close(points, radius):
    """
    Given an (n, m) set of points where n=(2|3) return a list of points
    where no point is closer than radius.

    Parameters
    ------------
    points : (n, dimension) float
      Points in space
    radius : float
      Minimum radius between result points

    Returns
    ------------
    culled : (m, dimension) float
      Points in space
    mask : (n,) bool
      Which points from the original set were returned
    """
    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(points)
    consumed = np.zeros(len(points), dtype=np.bool)
    unique = np.zeros(len(points), dtype=np.bool)
    for i in range(len(points)):
        if consumed[i]:
            continue
        neighbors = tree.query_ball_point(points[i], r=radius)
        consumed[neighbors] = True
        unique[i] = True

    return points[unique], unique


def k_means(points, k, **kwargs):
    """
    Find k centroids that attempt to minimize the k- means problem:
    https://en.wikipedia.org/wiki/Metric_k-center

    Parameters
    ----------
    points:  (n, d) float
        Points in a space
    k : int
         Number of centroids to compute
    **kwargs : dict
        Passed directly to scipy.cluster.vq.kmeans

    Returns
    ----------
    centroids : (k, d) float
         Points in some space
    labels: (n) int
        Indexes for which points belong to which centroid
    """
    from scipy.cluster.vq import kmeans
    from scipy.spatial import cKDTree

    points = np.asanyarray(points, dtype=np.float64)
    points_std = points.std(axis=0)
    whitened = points / points_std
    centroids_whitened, distortion = kmeans(whitened, k, **kwargs)
    centroids = centroids_whitened * points_std

    # find which centroid each point is closest to
    tree = cKDTree(centroids)
    labels = tree.query(points, k=1)[1]

    return centroids, labels


def tsp(points, start=0):
    """
    Find an ordering of points where each is visited and
    the next point is the closest in euclidean distance,
    and if there are multiple points with equal distance
    go to an arbitrary one.

    Assumes every point is visitable from every other point,
    i.e. the travelling salesman problem on a fully connected
    graph. It is not a MINIMUM traversal; rather it is a
    "not totally goofy traversal, quickly." On random points
    this traversal is often ~20x shorter than random ordering.

    Parameters
    ---------------
    points : (n, dimension) float
      ND points in space
    start : int
      The index of points we should start at

    Returns
    ---------------
    traversal : (n,) int
      Ordered traversal visiting every point
    distances : (n - 1,) float
      The euclidean distance between points in traversal
    """
    # points should be float
    points = np.asanyarray(points, dtype=np.float64)

    if len(points.shape) != 2:
        raise ValueError('points must be (n, dimension)!')

    # start should be an index
    start = int(start)

    # a mask of unvisited points by index
    unvisited = np.ones(len(points), dtype=np.bool)
    unvisited[start] = False

    # traversal of points by index
    traversal = np.zeros(len(points), dtype=np.int64) - 1
    traversal[0] = start
    # list of distances
    distances = np.zeros(len(points) - 1, dtype=np.float64)
    # a mask of indexes in order
    index_mask = np.arange(len(points), dtype=np.int64)

    # in the loop we want to call distances.sum(axis=1)
    # a lot and it's actually kind of slow for "reasons"
    # dot products with ones is equivalent and ~2x faster
    sum_ones = np.ones(points.shape[1])

    # loop through all points
    for i in range(len(points) - 1):
        # which point are we currently on
        current = points[traversal[i]]

        # do NlogN distance query
        # use dot instead of .sum(axis=1) or np.linalg.norm
        # as it is faster, also don't square root here
        dist = np.dot((points[unvisited] - current) ** 2,
                      sum_ones)

        # minimum distance index
        min_index = dist.argmin()
        # successor is closest unvisited point
        successor = index_mask[unvisited][min_index]
        # update the mask
        unvisited[successor] = False
        # store the index to the traversal
        traversal[i + 1] = successor
        # store the distance
        distances[i] = dist[min_index]

    # we were comparing distance^2 so take square root
    distances **= 0.5

    return traversal, distances


def plot_points(points, show=True):
    """
    Plot an (n,3) list of points using matplotlib

    Parameters
    -------------
    points : (n, 3) float
      Points in space
    show : bool
      If False, will not show until plt.show() is called
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA

    points = np.asanyarray(points, dtype=np.float64)

    if len(points.shape) != 2:
        raise ValueError('Points must be (n, 2|3)!')

    if points.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T)
    elif points.shape[1] == 2:
        plt.scatter(*points.T)
    else:
        raise ValueError('points not 2D/3D: {}'.format(
            points.shape))

    if show:
        plt.show()


class PointCloud(Geometry):
    """
    Hold 3D points in an object which can be visualized
    in a scene.
    """

    def __init__(self, vertices, colors=None, color=None):
        self._data = caching.DataStore()
        self._cache = caching.Cache(self._data.md5)
        self.metadata = {}

        # load vertices
        self.vertices = vertices

        if colors is not None:
            self.colors = colors
        elif color is not None:
            msg = (
                "'PointCloud(color=__)' is deprecated and will be " +
                "removed in versions released after 6/1/2019. " +
                "Use 'PointCloud(colors=__)' instead")
            warnings.warn(msg, DeprecationWarning)
            self.colors = color

    def __setitem__(self, *args, **kwargs):
        return self.vertices.__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.vertices.__getitem__(*args, **kwargs)

    @property
    def shape(self):
        """
        Get the shape of the pointcloud

        Returns
        ----------
        shape : (2,) int
          Shape of vertex array
        """
        return self.vertices.shape

    @property
    def is_empty(self):
        """
        Are there any vertices defined or not.

        Returns
        ----------
        empty : bool
          True if no vertices defined
        """
        return len(self.vertices) == 0

    def copy(self):
        """
        Safely get a copy of the current point cloud.

        Copied objects will have emptied caches to avoid memory
        issues and so may be slow on initial operations until
        caches are regenerated.

        Current object will *not* have its cache cleared.

        Returns
        ---------
        copied : trimesh.PointCloud
          Copy of current point cloud
        """
        copied = PointCloud(vertices=None)

        # copy vertex and face data
        copied._data.data = copy.deepcopy(self._data.data)
        # get metadata
        copied.metadata = copy.deepcopy(self.metadata)

        # make sure cache is set from here
        copied._cache.clear()

        return copied

    def md5(self):
        """
        Get an MD5 hash of the current vertices.

        Returns
        ----------
        md5 : str
          Hash of self.vertices
        """
        return self._data.md5()

    def merge_vertices(self):
        """
        Merge vertices closer than tol.merge (default: 1e-8)
        """
        # run unique rows
        unique, inverse = grouping.unique_rows(self.vertices)

        # apply unique mask to vertices
        self.vertices = self.vertices[unique]

        # apply unique mask to colors
        if (self.colors is not None and
                len(self.colors) == len(inverse)):
            self.colors = self.colors[unique]

    def apply_transform(self, transform):
        """
        Apply a homogenous transformation to the PointCloud
        object in- place.

        Parameters
        --------------
        transform : (4, 4) float
          Homogenous transformation to apply to PointCloud
        """
        self.vertices = transformations.transform_points(self.vertices,
                                                         matrix=transform)

    @property
    def bounds(self):
        """
        The axis aligned bounds of the PointCloud

        Returns
        ------------
        bounds : (2, 3) float
          Miniumum, Maximum verteex
        """
        return np.array([self.vertices.min(axis=0),
                         self.vertices.max(axis=0)])

    @property
    def extents(self):
        """
        The size of the axis aligned bounds

        Returns
        ------------
        extents : (3,) float
          Edge length of axis aligned bounding box
        """
        return self.bounds.ptp(axis=0)

    @property
    def centroid(self):
        """
        The mean vertex position

        Returns
        ------------
        centroid : (3,) float
          Mean vertex position
        """
        return self.vertices.mean(axis=0)

    @property
    def vertices(self):
        """
        Vertices of the PointCloud

        Returns
        ------------
        vertices : (n, 3) float
          Points in the PointCloud
        """
        return self._data['vertices']

    @vertices.setter
    def vertices(self, data):
        if data is None:
            self._data['vertices'] = None
        else:
            # we want to copy data for new object
            data = np.array(data, dtype=np.float64, copy=True)
            if not util.is_shape(data, (-1, 3)):
                raise ValueError('Point clouds must be (n, 3)!')
            self._data['vertices'] = data

    @property
    def colors(self):
        """
        Stored per- point color

        Returns
        ----------
        colors : (len(self.vertices), 4) np.uint8
          Per- point RGBA color
        """
        return self._data['colors']

    @colors.setter
    def colors(self, data):
        data = np.asanyarray(data)
        if data.shape == (4,):
            data = np.tile(data, (len(self.vertices), 1))
        self._data['colors'] = data

    @caching.cache_decorator
    def convex_hull(self):
        """
        A convex hull of every point.

        Returns
        -------------
        convex_hull : trimesh.Trimesh
          A watertight mesh of the hull of the points
        """
        from . import convex
        return convex.convex_hull(self.vertices)

    def scene(self):
        """
        A scene containing just the PointCloud

        Returns
        ----------
        scene : trimesh.Scene
          Scene object containing this PointCloud
        """
        from .scene.scene import Scene
        return Scene(self)

    def show(self, **kwargs):
        """
        Open a viewer window displaying the current PointCloud
        """
        self.scene().show(**kwargs)
