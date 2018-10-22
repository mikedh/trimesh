"""
path.py

A library designed to work with vector paths.
"""

import numpy as np
import networkx as nx

import copy
import collections

from shapely.geometry import Polygon
from scipy.spatial import cKDTree as KDTree

from ..points import plane_fit
from ..geometry import plane_transform

from ..constants import log
from ..constants import tol_path as tol

from .util import concatenate

from .. import util
from .. import units
from .. import caching
from .. import grouping
from .. import transformations

from . import raster
from . import simplify
from . import entities
from . import polygons
from . import segments
from . import traversal

from .io.export import export_path

try:
    # try running shapely speedups
    # these mostly speed up object instantiation
    from shapely import speedups
    if speedups.available:
        speedups.enable()
except BaseException:
    pass


class Path(object):
    """
    A Path object consists of:

    vertices: (n,[2|3]) coordinates, stored in self.vertices

    entities: geometric primitives (aka Lines, Arcs, etc.)
              that reference indexes in self.vertices
    """

    def __init__(self,
                 entities=[],
                 vertices=[],
                 metadata=None,
                 process=True):
        """
        Instantiate a path object.

        Parameters
        -----------
        entities : (m,) trimesh.path.entities.Entity
          Contains geometric entities
        vertices : (n, dimension) float
          The vertices referenced by entities
        metadata : dict
          Any metadata about the path
        process :  bool
          Run simple cleanup or not
        """

        self.entities = np.array(entities)
        self.vertices = vertices
        self.metadata = dict()

        if metadata.__class__.__name__ == 'dict':
            self.metadata.update(metadata)

        self._cache = caching.Cache(id_function=self.crc)

        if process:
            # literally nothing will work if vertices aren't
            # merged properly
            self.merge_vertices()

    def process(self):
        """
        Apply basic cleaning functions to the Path object, in- place.
        """
        log.debug('Processing drawing')
        with self._cache:
            for func in self._process_functions():
                func()
        return self

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, values):
        self._vertices = caching.tracked_array(values)

    @property
    def layers(self):
        """
        If entities have a layer defined, return it.

        Returns
        ---------
        layers: (len(entities), ) list of str
        """
        layer = [None] * len(self.entities)
        for i, e in enumerate(self.entities):
            if hasattr(e, 'layer'):
                layer[i] = str(e.layer)
        return layer

    def crc(self):
        """
        A CRC of the current vertices and entities.

        Returns
        ------------
        crc: int, CRC of entity points and vertices
        """
        # first CRC the points in every entity
        target = caching.crc32(bytes().join(
            e._bytes()
            for e in self.entities))
        # add the CRC for the vertices
        target += self.vertices.crc()
        return target

    def md5(self):
        """
        An MD5 hash of the current vertices and entities.

        Returns
        ------------
        md5: str, two appended MD5 hashes
        """

        target = util.md5_object(bytes().join(
            e._bytes()
            for e in self.entities))
        target += self.vertices.md5()
        return target

    @caching.cache_decorator
    def paths(self):
        """
        Sequence of closed paths, encoded by entity index.

        Returns
        ---------
        paths: (n,) sequence of (*,) int referencing self.entities
        """
        paths = traversal.closed_paths(self.entities,
                                       self.vertices)
        return paths

    @caching.cache_decorator
    def dangling(self):
        """
        List of entities that aren't included in a closed path

        Returns
        ----------
        dangling: (n,) int, index of self.entities
        """
        if len(self.paths) == 0:
            return np.arange(len(self.entities))
        else:
            included = np.hstack(self.paths)
        dangling = np.setdiff1d(np.arange(len(self.entities)),
                                included)
        return dangling

    @caching.cache_decorator
    def kdtree(self):
        """
        A KDTree object holding the vertices of the path.

        Returns
        ----------
        kdtree: scipy.spatial.cKDTree object holding self.vertices
        """

        kdtree = KDTree(self.vertices.view(np.ndarray))
        return kdtree

    @property
    def scale(self):
        """
        What is a representitive number that reflects the magnitude
        of the world holding the paths, for numerical comparisons.

        Returns
        ----------
        scale : float
            Approximate size of the world holding this path
        """
        # use vertices peak-peak rather than exact extents
        scale = float((self.vertices.ptp(axis=0) ** 2).sum() ** .5)
        return scale

    @caching.cache_decorator
    def bounds(self):
        """
        Return the axis aligned bounding box of the current path.

        Returns
        ----------
        bounds: (2, dimension) float, (min, max) coordinates
        """
        # get the exact bounds of each entity
        # some entities (aka 3- point Arc) have bounds that can't
        # be generated from just bound box of vertices
        points = np.array([e.bounds(self.vertices)
                           for e in self.entities],
                          dtype=np.float64)
        # flatten bound extrema into (n, dimension) array
        points = points.reshape((-1, self.vertices.shape[1]))
        # get the max and min of all bounds
        bounds = np.array([points.min(axis=0),
                           points.max(axis=0)],
                          dtype=np.float64)

        return bounds

    @property
    def extents(self):
        """
        The size of the axis aligned bounding box.

        Returns
        ---------
        extents: (dimension,) float, edge length of AABB
        """
        return self.bounds.ptp(axis=0)

    @property
    def units(self):
        """
        If there are units defined in self.metadata return them.

        Returns
        -----------
        units: str, current unit system
        """
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @units.setter
    def units(self, units):
        self.metadata['units'] = units

    def convert_units(self, desired, guess=False):
        """
        Convert the units of the current drawing in place.

        Parameters
        -----------
        desired: str, unit system to convert to
        guess:   bool, if True will attempt to guess units
        """
        units._convert_units(self,
                             desired=desired,
                             guess=guess)

    def explode(self):
        """
        Turn every multi- segment entity into single segment entities, in- place
        """
        new_entities = collections.deque()
        for entity in self.entities:
            new_entities.extend(entity.explode())
        self.entities = np.array(new_entities)

    def fill_gaps(self, max_distance=np.inf):
        """
        Find vertices with degree 1 and try to connect them to other
        vertices of degree 1, in place.

        Parameters
        ----------
        max_distance: float, connect vertices up to this distance.
                      Default is infinity, but something like path.scale/100
                      may make more sense.
        """

        broken = np.array(
            [k for k, v in dict(self.vertex_graph.degree()).items() if v == 1])
        if len(broken) < 2:
            return

        distance, node = KDTree(self.vertices[broken]).query(
            self.vertices[broken], k=2)

        edges = broken[node]
        ok = np.logical_and(distance[:, 1] < max_distance, [
                            not self.vertex_graph.has_edge(*i) for i in edges])

        self.entities = np.append(self.entities,
                                  [entities.Line(i) for i in edges[ok]])

    @property
    def is_closed(self):
        """
        Are all entities connected to other entities.

        Returns
        -----------
        closed: every entity is connected at its ends
        """
        closed = all(i == 2 for i in dict(self.vertex_graph.degree()).values())
        return closed

    @caching.cache_decorator
    def vertex_graph(self):
        """
        Return a networkx.Graph object for the entity connectiviy

        graph: networkx.Graph object, holding vertex indexes
        """
        graph, closed = traversal.vertex_graph(self.entities)
        return graph

    def apply_transform(self, transform):
        """
        Apply a transformation matrix to the current path in- place

        Parameters
        -----------
        transform: (d+1, d+1) float, homogenous transformation
                                    matrix for (n, d) vertices
        """
        dimension = self.vertices.shape[1]
        transform = np.asanyarray(transform, dtype=np.float64)

        if transform.shape != (dimension + 1, dimension + 1):
            raise ValueError('transform is incorrect shape!')
        elif np.abs(transform - np.eye(dimension + 1)).max() < 1e-8:
            # if we've been passed an identity matrix do nothing
            return

        # make sure cache is up to date
        self._cache.verify()
        # new cache to transfer items
        cache = {}
        # apply transform to discretized paths
        if 'discrete' in self._cache.cache:
            cache['discrete'] = np.array([
                transformations.transform_points(
                    d, matrix=transform)
                for d in self.discrete])

        # things we can just straight up copy
        # as they are topological not geometric
        for key in ['root',
                    'paths',
                    'path_valid',
                    'dangling',
                    'vertex_graph',
                    'enclosure',
                    'enclosure_shell',
                    'enclosure_directed']:
            # if they're in cache save them from the purge
            if key in self._cache.cache:
                cache[key] = self._cache.cache[key]

        # transform vertices in place
        self.vertices = transformations.transform_points(
            self.vertices,
            matrix=transform)
        # explicitly clear the cache
        self._cache.clear()
        self._cache.id_set()

        # populate the things we wangled
        self._cache.cache.update(cache)

    def apply_scale(self, scale):
        """
        Apply a transformation matrix to the current path in- place

        Parameters
        -----------
        transform: (dimension + 1, dimension + 1) float, homogenous
                   transformation matrix
        """
        dimension = self.vertices.shape[1]
        matrix = np.eye(dimension + 1)
        matrix[:dimension, :dimension] *= float(scale)
        self.apply_transform(matrix)

    def apply_translation(self, offset):
        """
        Apply a transformation matrix to the current path in- place

        Parameters
        -----------
        transform: (dimension + 1, dimension + 1) float, homogenous
                   transformation matrix
        """
        # work on 2D and 3D paths
        dimension = self.vertices.shape[1]
        # make sure offset is correct length and type
        offset = np.array(
            offset, dtype=np.float64).reshape(dimension)
        # create a homogenous transform
        matrix = np.eye(dimension + 1)
        # apply the offset
        matrix[:dimension, dimension] = offset

        self.apply_transform(matrix)

    def apply_layer(self, name):
        """
        Apply a layer name to every entity in the path.

        Parameters
        ------------
        name: str to apply to each entity
        """
        for e in self.entities:
            e.layer = name

    def rezero(self):
        """
        Translate so that every vertex is positive in the current
        mesh is positive.

        Returns
        -----------
        matrix: (dimension + 1, dimension + 1) float,
                    homogenous transformation
                    that was applied to the current Path object.
        """
        dimension = self.vertices.shape[1]
        matrix = np.eye(dimension + 1)
        matrix[:dimension, dimension] = -self.vertices.min(axis=0)
        self.apply_transform(matrix)
        return matrix

    def merge_vertices(self):
        """
        Merges vertices which are identical and replace references.

        Alters
        -----------
        self.entities: entity.points re- referenced
        self.vertices: duplicates removed
        """
        digits = util.decimal_to_digits(tol.merge * self.scale,
                                        min_digits=1)
        unique, inverse = grouping.unique_rows(self.vertices,
                                               digits=digits)
        self.vertices = self.vertices[unique]
        for entity in self.entities:
            # don't screw up control- point- knot relationship
            if type(entity).__name__ in 'BSpline Bezier':
                entity.points = inverse[entity.points]
                continue
            # if we merged duplicate vertices, the entity may contain
            # multiple references to the same vertex
            entity.points = grouping.merge_runs(inverse[entity.points])

    def replace_vertex_references(self, mask):
        """
        Replace the vertex index references in every entity.

        Parameters
        ------------
        mask: (len(self.vertices), ) int, contains new vertex indexes

        Alters
        ------------
        entity.points in self.entities: replaced by mask[entity.points]
        """
        for entity in self.entities:
            entity.points = mask[entity.points]

    def remove_entities(self, entity_ids):
        """
        Remove entities by index.

        Parameters
        -----------
        entity_ids: (n,) int, indexes of self.entities to remove
        """
        if len(entity_ids) == 0:
            return
        kept = np.setdiff1d(np.arange(len(self.entities)), entity_ids)
        self.entities = np.array(self.entities)[kept]

    def remove_invalid(self):
        """
        Remove entities which declare themselves invalid

        Alters
        ----------
        self.entities: shortened
        """
        valid = np.array([i.is_valid for i in self.entities],
                         dtype=np.bool)
        self.entities = self.entities[valid]

    def remove_duplicate_entities(self):
        """
        Remove entities that are duplicated

        Alters
        -------
        self.entities: length same or shorter
        """
        entity_hashes = np.array([hash(i) for i in self.entities])
        unique, inverse = grouping.unique_rows(entity_hashes)
        if len(unique) != len(self.entities):
            self.entities = np.array(self.entities)[unique]

    @caching.cache_decorator
    def referenced_vertices(self):
        """
        Which vertices are referenced by an entity.

        Returns
        -----------
        referenced_vertices: (n,) int, indexes of self.vertices
        """
        referenced = np.hstack([e.points for e in self.entities])
        referenced = np.unique(referenced.astype(np.int64))

        return referenced

    def remove_unreferenced_vertices(self):
        """
        Removes all vertices which aren't used by an entity.

        Alters
        ---------
        self.vertices: reordered and shortened
        self.entities: entity.points references updated
        """

        unique = self.referenced_vertices

        mask = np.ones(len(self.vertices), dtype=np.int64) * -1
        mask[unique] = np.arange(len(unique), dtype=np.int64)

        self.replace_vertex_references(mask=mask)
        self.vertices = self.vertices[unique]

    def discretize_path(self, path):
        """
        Given a list of entities, return a list of connected points.

        Parameters
        -----------
        path: (n,) int, indexes of self.entities

        Returns
        -----------
        discrete: (m, dimension)
        """
        discrete = traversal.discretize_path(self.entities,
                                             self.vertices,
                                             path,
                                             scale=self.scale)
        return discrete

    @caching.cache_decorator
    def discrete(self):
        """
        A sequence of connected vertices in space, corresponding to
        self.paths.

        Returns
        ---------
        discrete : (len(self.paths),)
            A sequence of (m*, dimension) float
        """
        discrete = np.array([self.discretize_path(i)
                             for i in self.paths])
        return discrete

    def export(self,
               file_obj=None,
               file_type=None,
               **kwargs):
        """
        Export the path to a file object or return data.

        Parameters
        ---------------
        file_obj : None, str, or file object
          File object or string to export to
        file_type : None or str
          Type of file: dxf, dict, svg

        Returns
        ---------------
        exported : bytes or str
          Exported as specified type
        """
        return export_path(self,
                           file_type=file_type,
                           file_obj=file_obj,
                           **kwargs)

    def to_dict(self):
        export_dict = self.export(file_type='dict')
        return export_dict

    def copy(self):
        """
        Get a copy of the current mesh

        Returns
        ---------
        copied: Path object, copy of self
        """

        metadata = {}
        # grab all the keys into a list so if something is added
        # in another thread it probably doesn't stomp on our loop
        for key in list(self.metadata.keys()):
            try:
                metadata[key] = copy.deepcopy(self.metadata[key])
            except RuntimeError:
                # multiple threads
                log.warning('key {} changed during copy'.format(key))

        # copy the core data
        copied = type(self)(entities=copy.deepcopy(self.entities),
                            vertices=copy.deepcopy(self.vertices),
                            metadata=metadata)

        cache = {}
        # try to copy the cache over to the new object
        try:
            # save dict keys before doing slow iteration
            keys = list(self._cache.cache.keys())
            # run through each key and copy into new cache
            for k in keys:
                cache[k] = copy.deepcopy(self._cache.cache[k])
        except RuntimeError:
            # if we have multiple threads this may error and is NBD
            log.debug('unable to copy cache')
        except BaseException:
            # catch and log errors we weren't expecting
            log.error('unable to copy cache', exc_info=True)
        copied._cache.cache = cache
        copied._cache.id_set()

        return copied

    def scene(self):
        """
        Get a scene object containing the current Path3D object.

        Returns
        --------
        scene: trimesh.scene.Scene object containing current path
        """
        from ..scene import Scene
        scene = Scene(self)
        return scene

    def __add__(self, other):
        """
        Concatenate two Path objects by appending vertices and
        reindexing point references.

        Parameters
        -----------
        other: Path object

        Returns
        -----------
        concat: Path object, appended from self and other
        """
        concat = concatenate([self, other])
        return concat


class Path3D(Path):

    def _process_functions(self):
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices,
                self.generate_closed_paths,
                self.generate_discrete]

    def to_planar(self,
                  to_2D=None,
                  normal=None,
                  check=True):
        """
        Check to see if current vectors are all coplanar.

        If they are, return a Path2D and a transform which will
        transform the 2D representation back into 3 dimensions

        Parameters
        -----------
        to_2D: (4,4) float
            Homogenous transformation matrix to apply,
            If not passed a plane will be fitted to vertices.
        normal: (3,) float, or None
           Approximate normal of direction of plane
           If to_2D is not specified sign
           will be applied to fit plane normal
        check:  bool
            If True: Raise a ValueError if
            points aren't coplanar

        Returns
        -----------
        planar : trimesh.path.Path2D
                   Current path transformed onto plane
        to_3D :  (4,4) float
                   Homeogenous transformation to move planar
                   back into 3D space
        """
        # which vertices are actually referenced
        referenced = self.referenced_vertices

        # no explicit transform passed
        if to_2D is None:
            # fit a plane to our vertices
            C, N = plane_fit(self.vertices[referenced])
            # apply the normal sign hint
            if normal is not None:
                normal = np.asanyarray(normal, dtype=np.float64)
                if normal.shape == (3,):
                    N *= np.sign(np.dot(N, normal))
                    N = normal
                else:
                    log.warning(
                        "passed normal not used: {}".format(
                            normal.shape))
            # create a transform from fit plane to XY
            to_2D = plane_transform(origin=C,
                                    normal=N)

        # make sure we've extracted a transform
        to_2D = np.asanyarray(to_2D, dtype=np.float64)
        if to_2D.shape != (4, 4):
            raise ValueError('unable to create transform!')

        # transform all vertices to 2D plane
        flat = transformations.transform_points(self.vertices,
                                                to_2D)

        # Z values of vertices which are referenced
        heights = flat[referenced][:, 2]
        # points are not on a plane because Z varies
        if heights.ptp() > tol.planar:
            # since Z is inconsistent set height to zero
            height = 0.0
            if check:
                raise ValueError('points are not flat!')
        else:
            # if the points were planar store the height
            height = heights.mean()

        # the transform from 2D to 3D
        to_3D = np.linalg.inv(to_2D)

        # if the transform didn't move the path to
        # exactly Z=0 adjust it so the returned transform does
        if np.abs(height) > tol.planar:
            # adjust to_3D transform by height
            adjust = transformations.translation_matrix(
                [0, 0, height])
            # apply the height adjustment to_3D
            to_3D = np.dot(to_3D, adjust)

            # do a check on to_3D
            # flat[:,2] = 0
            # a = transformations.transform_points(flat[referenced],
            #                                     to_3D)
            # assert np.allclose(self.vertices[referenced], a)

        # copy metadata to new object
        metadata = copy.deepcopy(self.metadata)
        # store transform we used to move it onto the plane
        metadata['to_3D'] = to_3D

        # create the Path2D with the same entities
        # and XY values of vertices projected onto the plane
        planar = Path2D(entities=copy.deepcopy(self.entities),
                        vertices=flat[:, :2],
                        metadata=metadata)

        return planar, to_3D

    def show(self, **kwargs):
        """
        Show the current Path3D object.
        """
        scene = self.scene()
        return scene.show(**kwargs)

    def plot_discrete(self, show=False):
        """
        Plot closed curves

        Parameters
        ------------
        show : bool
           If False will not execute matplotlib.pyplot.show
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for discrete in self.discrete:
            axis.plot(*discrete.T)
        if show:
            plt.show()

    def plot_entities(self, show=False):
        """
        Plot discrete version of entities without regards
        for connectivity.

        Parameters
        -------------
        show : bool
           If False will not execute matplotlib.pyplot.show
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for entity in self.entities:
            vertices = entity.discrete(self.vertices)
            axis.plot(*vertices.T)
        if show:
            plt.show()


class Path2D(Path):

    def show(self):
        """
        Plot the current Path2D object using matplotlib.
        """
        if self.is_closed:
            self.plot_discrete(show=True)
        else:
            self.plot_entities(show=True)

    def _process_functions(self):
        """
        Return a list of functions to clean up a Path2D
        """
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices]

    def apply_obb(self):
        """
        Transform the current path so that its OBB is axis aligned
        and OBB center is at the origin.
        """
        if len(self.root) == 1:
            matrix, bounds = polygons.polygon_obb(
                self.polygons_closed[self.root[0]])
            self.apply_transform(matrix)
            return matrix
        else:
            raise ValueError('Not implemented for multibody geometry')

    def rasterize(self,
                  pitch,
                  origin,
                  resolution=None,
                  fill=True,
                  width=None,
                  **kwargs):
        """
        Rasterize a Path2D object into a boolean image ("mode 1").

        Parameters
        ------------
        pitch:      float, length in model space of a pixel edge
        origin:     (2,) float, origin position in model space
        resolution: (2,) int, resolution in pixel space
        fill:       bool, if True will return closed regions as filled
        width:      int, if not None will draw outline this wide (pixels)

        Returns
        ------------
        raster: PIL.Image object, mode 1
        """
        image = raster.rasterize(self,
                                 pitch=pitch,
                                 origin=origin,
                                 resolution=resolution,
                                 fill=fill,
                                 width=width)
        return image

    def sample(self, count, **kwargs):
        """
        Use rejection sampling to generate random points inside a
        polygon.

        Parameters
        -----------
        count   : int
                    Number of points to return
                    If there are multiple bodies, there will
                    be up to count * bodies points returned
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
        """

        poly = self.polygons_full
        if len(poly) == 0:
            samples = np.array([])
        elif len(poly) == 1:
            samples = polygons.sample(poly[0], count=count, **kwargs)
        else:
            samples = util.vstack_empty([
                polygons.sample(i, count=count, **kwargs)
                for i in poly])

        return samples

    @property
    def body_count(self):
        return len(self.root)

    def to_3D(self, transform=None):
        """
        Convert 2D path to 3D path on the XY plane.

        Parameters
        -------------
        transform : (4, 4) float
            If passed, will transform vertices.
            If not passed and 'to_3D' is in metadata
            that transform will be used.

        Returns
        -----------
        path_3D: Path3D version of current path
        """
        # if there is a stored 'to_3D' transform in metadata use it
        if transform is None and 'to_3D' in self.metadata:
            transform = self.metadata['to_3D']

        # copy vertices and stack with zeros from (n, 2) to (n, 3)
        vertices = np.column_stack((copy.deepcopy(self.vertices),
                                    np.zeros(len(self.vertices))))
        if transform is not None:
            vertices = transformations.transform_points(vertices,
                                                        transform)
        # make sure everything is deep copied
        path_3D = Path3D(entities=copy.deepcopy(self.entities),
                         vertices=vertices,
                         metadata=copy.deepcopy(self.metadata))
        return path_3D

    @caching.cache_decorator
    def polygons_closed(self):
        """
        Cycles in the vertex graph, as shapely.geometry.Polygons.
        These are polygon objects for every closed circuit, with no notion
        of whether a polygon is a hole or an area. Every polygon in this
        list will have an exterior, but NO interiors.

        Returns
        ---------
        polygons_closed: (n,) list of shapely.geometry.Polygon objects
        """
        # will attempt to recover invalid garbage geometry
        # and will be None if geometry is unrecoverable
        polys = polygons.paths_to_polygons(self.discrete)
        return polys

    @caching.cache_decorator
    def polygons_full(self):
        """
        A list of shapely.geometry.Polygon objects with interiors created
        by checking which closed polygons enclose which other polygons.

        Returns
        ---------
        full : (len(self.root),) shapely.geometry.Polygon
            Polygons containing interiors
        """
        # pre- allocate the list to avoid indexing problems
        full = [None] * len(self.root)
        # store the graph to avoid cache thrashing
        enclosure = self.enclosure_directed
        # store closed polygons to avoid cache hits
        closed = self.polygons_closed

        # loop through root curves
        for i, root in enumerate(self.root):
            # a list of multiple Polygon objects that
            # are fully contained by the root curve
            children = [closed[child]
                        for child in enclosure[root].keys()]
            # all polygons_closed are CCW, so for interiors reverse them
            holes = [np.array(p.exterior.coords)[::-1] for p in children]
            # a single Polygon object
            shell = closed[root].exterior
            # create a polygon with interiors
            full[i] = polygons.repair_invalid(Polygon(shell=shell,
                                                      holes=holes))
        # so we can use advanced indexing
        full = np.array(full)

        return full

    @caching.cache_decorator
    def area(self):
        """
        Return the area of the polygons interior.

        Returns
        ---------
        area: float, total area of polygons minus interiors
        """
        area = sum(i.area for i in self.polygons_full)
        return area

    @caching.cache_decorator
    def length(self):
        """
        The total discretized length of every entity.

        Returns
        --------
        length: float, summed length of every entity
        """
        length = float(sum(i.length(self.vertices) for i in self.entities))
        return length

    def extrude(self, height, **kwargs):
        """
        Extrude the current 2D path into a 3D mesh.

        Parameters
        ----------
        height: float, how far to extrude the profile
        kwargs: passed directly to meshpy.triangle.build:
                triangle.build(mesh_info,
                               verbose=False,
                               refinement_func=None,
                               attributes=False,
                               volume_constraints=True,
                               max_volume=None,
                               allow_boundary_steiner=True,
                               allow_volume_steiner=True,
                               quality_meshing=True,
                               generate_edges=None,
                               generate_faces=False,
                               min_angle=None)
        Returns
        --------
        mesh: trimesh object representing extruded polygon
        """
        from ..primitives import Extrusion
        result = [Extrusion(polygon=i, height=height, **kwargs)
                  for i in self.polygons_full]
        if len(result) == 1:
            return result[0]
        return result

    def medial_axis(self, resolution=None, clip=None):
        """
        Find the approximate medial axis based
        on a voronoi diagram of evenly spaced points on the
        boundary of the polygon.

        Parameters
        ----------
        resolution: target distance between each sample on the polygon boundary
        clip:       [minimum number of samples, maximum number of samples]
                    specifying a very fine resolution can cause the sample count to
                    explode, so clip specifies a minimum and maximum number of samples
                    to use per boundary region. To not clip, this can be specified as:
                    [0, np.inf]

        Returns
        ----------
        medial : Path2D object
        """
        if resolution is None:
            resolution = self.scale / 1000.0

        medials = [polygons.medial_axis(i, resolution, clip)
                   for i in self.polygons_full]
        medials = np.sum(medials)
        return medials

    def connected_paths(self, path_id, include_self=False):
        """
        Given an index of self.paths, find other paths which overlap with
        that path.

        Parameters
        -----------
        path_id:      int, index of self.paths
        include_self: bool, should the result include path_id or not

        Returns
        -----------
        path_ids: (n,) int, indexes of self.paths that overlap input path_id
        """
        if len(self.root) == 1:
            path_ids = np.arange(len(self.polygons_closed))
        else:
            path_ids = list(nx.node_connected_component(self.enclosure,
                                                        path_id))
        if include_self:
            return np.array(path_ids)
        return np.setdiff1d(path_ids, [path_id])

    def simplify(self, **kwargs):
        """
        Return a version of the current path with colinear segments
        merged, and circles entities replacing segmented circular paths.

        Returns
        ---------
        simplified: Path2D object
        """
        return simplify.simplify_basic(self, **kwargs)

    def simplify_spline(self, path_indexes=None, smooth=.0002):
        """
        Convert paths into b-splines.

        Parameters
        -----------
        path_indexes: (n) int list of indexes for self.paths
        smooth:       float, how much the spline should smooth the curve

        Returns
        ------------
        simplified: Path2D object
        """
        return simplify.simplify_spline(self,
                                        path_indexes=path_indexes,
                                        smooth=smooth)

    def split(self):
        """
        If the current Path2D consists of n 'root' curves,
        split them into a list of n Path2D objects

        Returns
        ----------
        split: (n,) list of Path2D objects
        """
        return traversal.split(self)

    def plot_discrete(self, show=False):
        """
        Plot the closed curves of the path.
        """
        import matplotlib.pyplot as plt
        axis = plt.axes()
        axis.set_aspect('equal', 'datalim')

        for i, points in enumerate(self.discrete):
            color = ['g', 'k'][i in self.root]
            axis.plot(*points.T, color=color)
        if show:
            plt.show()
        return axis

    def plot_entities(self, show=False, color=None):
        """
        Plot the entities of the path, with no notion of topology
        """
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        eformat = {'Line0': {'color': 'g', 'linewidth': 1},
                   'Line1': {'color': 'y', 'linewidth': 1},
                   'Arc0': {'color': 'r', 'linewidth': 1},
                   'Arc1': {'color': 'b', 'linewidth': 1},
                   'Bezier0': {'color': 'k', 'linewidth': 1},
                   'BSpline0': {'color': 'm', 'linewidth': 1},
                   'BSpline1': {'color': 'm', 'linewidth': 1}}
        for entity in self.entities:
            discrete = entity.discrete(self.vertices)
            e_key = entity.__class__.__name__ + str(int(entity.closed))
            fmt = eformat[e_key]
            if color is not None:
                # passed color will override other optons
                fmt['color'] = color
            elif hasattr(entity, 'color'):
                # if entity has specified color use it
                fmt['color'] = entity.color
            plt.plot(*discrete.T, **fmt)
        if show:
            plt.show()

    @property
    def identifier(self):
        """
        A unique identifier for the path.

        Returns
        ---------
        identifier: (5,) float, unique identifier
        """
        if len(self.polygons_full) != 1:
            raise TypeError('Identifier only valid for single body')
        return polygons.polygon_hash(self.polygons_full[0])

    @caching.cache_decorator
    def identifier_md5(self):
        """
        Return an MD5 of the identifier
        """
        as_int = (self.identifier * 1e4).astype(np.int64)
        hashed = util.md5_object(as_int.tostring(order='C'))
        return hashed

    @property
    def path_valid(self):
        """
        Returns
        ----------
        path_valid: (n,) bool, indexes of self.paths self.polygons_closed
                         which are valid polygons
        """
        valid = [i is not None for i in self.polygons_closed]
        valid = np.array(valid, dtype=np.bool)
        return valid

    @caching.cache_decorator
    def root(self):
        """
        Which indexes of self.paths/self.polygons_closed are root curves.
        Also known as 'shell' or 'exterior.

        Returns
        ---------
        root: (n,) int, list of indexes
        """
        populate = self.enclosure_directed
        return self._cache['root']

    @caching.cache_decorator
    def enclosure(self):
        """
        Networkx Graph object of polygon enclosure.
        """
        with self._cache:
            undirected = self.enclosure_directed.to_undirected()
        return undirected

    @caching.cache_decorator
    def enclosure_directed(self):
        """
        Networkx DiGraph of polygon enclosure
        """
        root, enclosure = polygons.enclosure_tree(self.polygons_closed)
        self._cache['root'] = root
        return enclosure

    @caching.cache_decorator
    def enclosure_shell(self):
        """
        A dictionary of path indexes which are 'shell' paths, and values
        of 'hole' paths.

        Returns
        ----------
        corresponding: dict, {index of self.paths of shell : [indexes of holes]}
        """
        pairs = [(r, self.connected_paths(r, include_self=False))
                 for r in self.root]
        # OrderedDict to maintain corresponding order
        corresponding = collections.OrderedDict(pairs)
        return corresponding
