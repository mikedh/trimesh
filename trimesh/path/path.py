'''
path.py

A library designed to work with vector paths.
'''

import numpy as np
import networkx as nx

from shapely.geometry import Polygon
from scipy.spatial import cKDTree as KDTree
from copy import deepcopy
from collections import deque

from .simplify  import simplify_path, points_to_spline_entity
from .polygons  import polygons_enclosure_tree, medial_axis, polygon_hash, path_to_polygon, polygon_obb
from .traversal import vertex_graph, closed_paths, discretize_path
from .io.export import export_path
from ..points    import plane_fit
from ..geometry  import plane_transform
from ..grouping  import unique_rows
from ..units     import _set_units
from ..util      import decimal_to_digits
from ..constants import log
from ..constants import tol_path as tol



from .. import transformations

from .. import util

class Path(object):
    '''
    A Path object consists of two things:
    vertices: (n,[2|3]) coordinates, stored in self.vertices
    entities: geometric primitives (lines, arcs, and circles)
              that reference indices in self.vertices
    '''
    def __init__(self, 
                 entities = [], 
                 vertices = [],
                 metadata = None,
                 process  = True):
        '''
        entities:
            Objects which contain things like keypoints, as 
            references to self.vertices
        vertices:
            (n, (2|3)) list of vertices
        '''
        self.entities = np.array(entities)
        self.vertices = vertices
        self.metadata = dict()

        if metadata.__class__.__name__ == 'dict':
            self.metadata.update(metadata)

        self._cache = util.Cache(id_function = self.md5)

        if process:
            # literally nothing will work if vertices aren't merged properly
            self.merge_vertices()

    def process(self):
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
        self._vertices = util.tracked_array(values)

    def md5(self):
        result  = self.vertices.md5()
        result += str(len(self.entities))
        return result

    @util.cache_decorator
    def paths(self):
        paths = closed_paths(self.entities, self.vertices)
        return paths

    @util.cache_decorator
    def kdtree(self):
        kdtree = KDTree(self.vertices.view(np.ndarray))
        return kdtree

    @property
    def scale(self):
        scale = self.extents.max()
        return scale
        
    @util.cache_decorator
    def bounds(self):
        return np.vstack((np.min(self.vertices, axis=0),
                          np.max(self.vertices, axis=0)))
    @property
    def extents(self):
        return np.diff(self.bounds, axis=0)[0]

    @property
    def units(self):
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @property
    def is_closed(self):
        return all(i == 2 for i in self.vertex_graph.degree().values())

    @util.cache_decorator
    def vertex_graph(self):
        graph, closed = vertex_graph(self.entities)
        return graph
   
    @units.setter
    def units(self, units):
        self.metadata['units'] = units
            
    def convert_units(self, desired, guess=False):
        _set_units(self, desired, guess)

    def apply_transform(self, transform):
        self.vertices = transformations.transform_points(self.vertices, 
                                                         transform)

    def rezero(self):
        self.vertices -= self.vertices.min(axis=0)
        
    def merge_vertices(self):
        '''
        Merges vertices which are identical and replaces references
        '''
        digits = decimal_to_digits(tol.merge * self.scale, min_digits=1)
        unique, inverse = unique_rows(self.vertices, digits=digits)
        self.vertices = self.vertices[unique]
        for entity in self.entities: 
            entity.points = inverse[entity.points]

    def replace_vertex_references(self, replacement_dict):
        for entity in self.entities: entity.rereference(replacement_dict)

    def remove_entities(self, entity_ids):
        '''
        Remove entities by their index.
        '''
        if len(entity_ids) == 0: return
        kept = np.setdiff1d(np.arange(len(self.entities)), entity_ids)
        self.entities = np.array(self.entities)[kept]

    def remove_invalid(self):
        valid = np.array([i.is_valid for i in self.entities], dtype=np.bool)
        self.entities = self.entities[valid]

    def remove_duplicate_entities(self):
        entity_hashes   = np.array([i.hash for i in self.entities])
        unique, inverse = unique_rows(entity_hashes)
        if len(unique) != len(self.entities):
            self.entities = np.array(self.entities)[unique]
            
    def referenced_vertices(self):
        referenced = deque()
        for entity in self.entities: 
            referenced.extend(entity.points)
        return np.array(referenced)
    
    def remove_unreferenced_vertices(self):
        '''
        Removes all vertices which aren't used by an entity
        Reindexes vertices from zero, and replaces references
        '''
        referenced       = self.referenced_vertices()
        unique_ref       = np.int_(np.unique(referenced))
        replacement_dict = dict()
        replacement_dict.update(np.column_stack((unique_ref, 
                                                 np.arange(len(unique_ref)))))
        self.replace_vertex_references(replacement_dict)
        self.vertices = self.vertices[[unique_ref]] 
        
    def discretize_path(self, path):
        '''
        Return a (n, dimension) list of vertices. 
        Samples arcs/curves to be line segments
        '''
        discrete = discretize_path(self.entities, 
                                   self.vertices, 
                                   path, 
                                   scale = self.scale)
        return discrete
        
    def paths_to_splines(self, path_indexes=None, smooth=.0002):
        '''
        Convert paths into b-splines.
        
        Arguments
        -----------
        path_indexes: (n) int list of indexes for self.paths 
        smooth:       float, how much the spline should smooth the curve
        '''
        if path_indexes is None:
            path_indexes = np.arange(len(self.paths))
        entities_keep = np.ones(len(self.entities), dtype=np.bool)
        new_vertices = deque()
        new_entities = deque()
        for i in path_indexes:
            path     = self.paths[i]
            discrete = self.discrete[i]
            entity, vertices = points_to_spline_entity(discrete)
            entity.points += len(self.vertices) + len(new_vertices)
            new_vertices.extend(vertices)
            new_entities.append(entity)
            entities_keep[path] = False
        self.entities = np.append(self.entities[entities_keep], 
                                  new_entities)
        self.vertices = np.vstack((self.vertices, 
                                   np.array(new_vertices)))

    def export(self, file_obj=None, file_type='dict'):
        return export_path(self, 
                           file_type = file_type,
                           file_obj  = file_obj)

    def to_dict(self):
        export_dict = self.export(file_type='dict')
        return export_dict

    def copy(self):
        return deepcopy(self)

    def show(self):
        if self.is_closed:
            self.plot_discrete(show=True)
        else:
            self.plot_entities(show=True)

    def __add__(self, other):
        new_entities = deepcopy(other.entities)
        for entity in new_entities:
            entity.points += len(self.vertices)
        new_entities = np.append(deepcopy(self.entities), new_entities)
 
        new_vertices = np.vstack((self.vertices, other.vertices))
        new_meta     = deepcopy(self.metadata)
        new_meta.update(other.metadata)

        new_path = self.__class__(entities = new_entities,
                                  vertices = new_vertices,
                                  metadata = new_meta)
        return new_path
   
class Path3D(Path):
    def _process_functions(self): 
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices,
                self.generate_closed_paths,
                self.generate_discrete]
        
    @util.cache_decorator
    def discrete(self):
        discrete = list(map(self.discretize_path, self.paths))
        return discrete

    def to_planar(self, to_2D=None, normal=None, check=True):
        '''
        Check to see if current vectors are all coplanar.
        
        If they are, return a Path2D and a transform which will 
        transform the 2D representation back into 3 dimensions
        '''
        if to_2D is None:
            C, N = plane_fit(self.vertices)
            if normal is not None:
                N *= np.sign(np.dot(N, normal))
                N = normal
            to_2D = plane_transform(C,N)
 
        flat = transformations.transform_points(self.vertices, to_2D)
        
        if check and np.any(np.std(flat[:,2]) > tol.planar):
            log.error('points have z with deviation %f', np.std(flat[:,2]))
            raise NameError('Points aren\'t planar!')
            
        vector = Path2D(entities = deepcopy(self.entities), 
                        vertices = flat[:,0:2])
        to_3D  = np.linalg.inv(to_2D)

        return vector, to_3D

    def scene(self):
        '''
        Get a scene object containing the current Path3D object.

        Returns
        --------
        scene: trimesh.scene.Scene object containing current path
        '''
        from ..scene import Scene
        scene = Scene(self)
        return scene

    def show(self):
        '''
        Show the current Path3D object.
        '''
        self.scene().show()
    
    def plot_discrete(self, show=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig  = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for discrete in self.discrete:
            axis.plot(*discrete.T)
        if show: plt.show()

    def plot_entities(self, show=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig  = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for entity in self.entities:
            vertices = entity.discrete(self.vertices)
            axis.plot(*vertices.T)        
        if show: plt.show()

class Path2D(Path):
    def _process_functions(self): 
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices]

    def apply_obb(self):
        if len(self.root) == 1:
            bounds, T = polygon_obb(self.polygons_closed[self.root[0]])
            self.transform(T)
        else:
            raise ValueError('Not implemented for multibody geometry')

    @property
    def body_count(self):
        return len(self.root)

    def to_3D(self):
        '''
        Convert 2D path to 3D path on the XY plane.

        Returns
        -----------
        path_3D: Path3D version of current path
        '''
        path_3D = Path3D(entities = deepcopy(self.entities), 
                         vertices = np.column_stack((self.vertices,
                                                     np.zeros(len(self.vertices)))))
        return path_3D

    @util.cache_decorator
    def polygons_full(self):
        result = [None] * len(self.root)
        for index, root in enumerate(self.root):
            hole_index = self.connected_paths(root, include_self=False)
            holes = [p.exterior.coords for p in self.polygons_closed[hole_index]]
            shell = self.polygons_closed[root].exterior.coords
            result[index] = Polygon(shell  = shell,
                                    holes  = holes)
        return result

    @util.cache_decorator
    def area(self):
        '''
        Return the area of the polygons interior
        '''
        area = np.sum([i.area for i in self.polygons_full])
        return area
        
    def extrude(self, height, **kwargs):
        '''
        Extrude the current 2D path into a 3D mesh. 

        Arguments
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
        '''
        from ..primitives import Extrusion
        result = [Extrusion(polygon=i, height=height, **kwargs) for i in self.polygons_full]
        if len(result) == 1: 
            return result[0]
        return result

    def medial_axis(self, resolution=None, clip=None):

        '''
        Find the approximate medial axis based
        on a voronoi diagram of evenly spaced points on the boundary of the polygon.

        Arguments
        ----------
        resolution: target distance between each sample on the polygon boundary
        clip:       [minimum number of samples, maximum number of samples]
                    specifying a very fine resolution can cause the sample count to
                    explode, so clip specifies a minimum and maximum number of samples
                    to use per boundary region. To not clip, this can be specified as:
                    [0, np.inf]

        Returns
        ----------
        medial:     Path2D object
        '''
        if 'medial' in self._cache:
            return self._cache.get('medial')

        if resolution is None:
            resolution = self.scale / 1000.0

        medials = [medial_axis(i, resolution, clip) for i in self.polygons_full]
        medials = np.sum(medials)
        return self._cache.set(key = 'medial',
                               value = medials)
        
    def connected_paths(self, path_id, include_self = False):
        if len(self.root) == 1:
            path_ids = np.arange(len(self.polygons_closed))
        else:
            path_ids = list(nx.node_connected_component(self.enclosure, path_id))
        if include_self: 
            return np.array(path_ids)
        return np.setdiff1d(path_ids, [path_id])
        
    def simplify(self):
        self._cache.clear()
        simplify_path(self)

    def split(self):
        '''
        If the current Path2D consists of n 'root' curves,
        split them into a list of n Path2D objects
        '''
        if self.root is None or len(self.root) == 0:
            split = []
        elif len(self.root) == 1:
            split = [deepcopy(self)]
        else:
            split = [None] * len(self.root)
            for i, root in enumerate(self.root):
                connected = self.connected_paths(root, include_self=True)
                new_root     = np.nonzero(connected == root)[0]
                new_entities = deque()
                new_paths    = deque()
                new_metadata = {'split_2D' : i}
                new_metadata.update(self.metadata)


                for path in self.paths[connected]:
                    new_paths.append(np.arange(len(path)) + len(new_entities))
                    new_entities.extend(path)
                new_entities = np.array(new_entities)
                # prevents the copying from nuking our cache
                with self._cache:
                    split[i] = Path2D(entities = deepcopy(self.entities[new_entities]),
                                      vertices = deepcopy(self.vertices))
                    split[i]._cache.update({'paths'          : np.array(new_paths),
                                            'polygons_closed': self.polygons_closed[connected],
                                            'root'           : new_root})
        [i._cache.id_set() for i in split]
        self._cache.id_set()
        return np.array(split)

    def plot_discrete(self, show=False, transform=None, axes=None):
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        def plot_transformed(vertices, color='g'):
            if transform is None: 
                if axes is None:
                    plt.plot(*vertices.T, color=color)
                else:
                    axes.plot(*vertices.T, color=color)
            else:
                transformed = transformations.transform_points(vertices, transform)
                plt.plot(*transformed.T, color=color)
        for i, polygon in enumerate(self.polygons_closed):
            color = ['g','k'][i in self.root]
            plot_transformed(np.array(polygon.boundary.coords), color=color)
        if show: plt.show()

    def plot_entities(self, show=False):
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        eformat = {'Line0'  : {'color'  :'g', 'linewidth':1},
                   'Line1'  : {'color'  :'y', 'linewidth':1},
                   'Arc0'   : {'color'  :'r', 'linewidth':1}, 
                   'Arc1'   : {'color'  :'b', 'linewidth':1},
                   'Bezier0': {'color'  :'k', 'linewidth':1},
                   'BSpline0': {'color' :'m', 'linewidth':1},
                   'BSpline1': {'color' :'m', 'linewidth':1}}
        for entity in self.entities:
            discrete = entity.discrete(self.vertices)
            e_key    = entity.__class__.__name__ + str(int(entity.closed))
            plt.plot(discrete[:,0], 
                     discrete[:,1], 
                     **eformat[e_key])
        if show: plt.show()

    @property
    def identifier(self):
        if len(self.polygons_full) != 1: 
            raise TypeError('Identifier only valid for single body')
        return polygon_hash(self.polygons_full[0])


    @property
    def polygons_valid(self):
        exists = self.polygons_closed
        return self._cache.get('polygons_valid')

    @property
    def discrete(self):
        if not 'discrete' in self._cache:
            test = self.polygons_closed
        return self._cache['discrete']

    @property
    def polygons_closed(self):
        if 'polygons_closed' in self._cache: 
            return self._cache.get('polygons_closed')

        def reverse_path(path):
            for entity in self.entities[path]: 
                entity.reverse()
            return path[::-1]
        
        with self._cache:
            discretized = [None] * len(self.paths)
            polygons = [None]  * len(self.paths)
            valid    = [False] * len(self.paths)
            for i, path in enumerate(self.paths):
                discrete = discretize_path(self.entities, 
                                           self.vertices, 
                                           path, 
                                           scale = self.scale)
                candidate = path_to_polygon(discrete, scale=self.scale)
                if candidate is None: 
                    continue
                if type(candidate).__name__ == 'MultiPolygon':
                    area_ok = np.array([i.area for i in candidate]) > tol.zero
                    if area_ok.sum() == 1:
                        candidate = candidate[np.nonzero(area_ok)[0][0]]
                    else:
                        continue
                if not candidate.exterior.is_ccw:
                    log.debug('Clockwise polygon detected, correcting!')
                    self.paths[i] = reverse_path(path)
                    candidate = Polygon(np.array(candidate.exterior.coords)[::-1])
                polygons[i] = candidate
                valid[i]    = True
                discretized[i] = discrete
            valid = np.array(valid, dtype=np.bool)
            polygons = np.array(polygons)[valid]
            discretized = np.array(discretized)

        self._cache.set('discrete',        discretized)
        self._cache.set('polygons_valid',  valid)
        self._cache.set('polygons_closed', polygons)
        return polygons

    @property
    def root(self):
        if 'root' in self._cache:
            return self._cache.get('root')
        with self._cache:
            root, enclosure = polygons_enclosure_tree(self.polygons_closed)
        self._cache.set('enclosure_directed', enclosure)
        return self._cache.set('root', root)

    @property
    def enclosure(self):
        if 'enclosure' in self._cache:
            return self._cache.get('enclosure')
        with self._cache:
            undirected = self.enclosure_directed.to_undirected()
        return self._cache.set('enclosure', undirected)
        
    @util.cache_decorator
    def enclosure_directed(self):
        root, enclosure = polygons_enclosure_tree(self.polygons_closed)
        self._cache.set('root', root)
        return enclosure
