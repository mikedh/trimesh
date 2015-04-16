'''
vector_path

A library designed to work with vector paths, 
or continuous paths in space
'''
import numpy as np
import networkx as nx
import json

from shapely.geometry import Polygon, Point
from copy import deepcopy

from collections import deque
from .polygons  import polygons_enclosure_tree, is_ccw
from .constants import *
from ..geometry import plane_fit, plane_transform, transform_points
from ..grouping import unique_rows

class Path:
    '''
    Three tiers of objects:
    Vertices: coordinates, stored in self.vertices
    Entities: objects with certain functions defined, that reference self.vertices
    Paths   : lists of entity indices, that references self.entities
    '''
    def __init__(self, 
                 entities = [], 
                 vertices = [],
                 metadata = None):
        '''
        entities:
            Objects which contain things like keypoints, as 
            references to self.vertices
        vertices:
            (n, (2|3)) list of vertices
        '''
        self.entities = np.array(entities)
        self.vertices = np.array(vertices)

        if metadata.__class__.__name__ == 'dict':
            self.metadata = metadata
        else:
            self.metadata = dict()

        self._cache = {}

    def _cache_verify(self):
        ok = 'entity_count' in self._cache
        ok = ok and (len(self.entities) == self._cache['entity_count'])
        if not ok: 
            self._cache = {'entity_count': len(self.entities)}
            self.process()

    def _cache_get(self, key):
        self._cache_verify()
        if key in self._cache: 
            return self._cache[key]
        return None

    def _cache_put(self, key, value):
        self._cache_verify()
        self._cache[key] = value

    @property
    def paths(self):
        return self._cache_get('paths')

    @property
    def polygons(self):
        return self._cache_get('polygons')

    @property
    def root(self):
        return self._cache_get('root')

    @property
    def enclosure(self):
        return self._cache_get('enclosure')

    @property
    def discrete(self):
        return self._cache_get('discrete')

    def scale(self):
        return np.max(np.ptp(self.vertices, axis=0))
 
    def bounds(self):
        return np.vstack((np.min(self.vertices, axis=0),
                          np.max(self.vertices, axis=0)))

    def box_size(self):
        return np.diff(self.bounds, axis=0)[0]
 
    def area(self):
        sum_area = 0.0
        for path_index in self.paths:
            sign      = ((path_index in self.root)*2) - 1
            sum_area += (sign * self.polygons[path_index].area)
        return sum_area
        
    def transform(self, transform):
        self.vertices = transform_points(self.vertices, transform)

    def rezero(self):
        self.vertices -= self.vertices.min(axis=0)
        
    def merge_vertices(self):
        '''
        Merges vertices which are identical and replaces references
        '''
        unique, inverse = unique_rows(self.vertices, return_inverse=True)
        self.vertices = self.vertices[unique]
        for entity in self.entities: 
            entity.points = inverse[entity.points]

    def replace_vertex_references(self, replacement_dict):
        for entity in self.entities: entity.rereference(replacement_dict)

    def remove_entities(self, entity_ids, reindex_paths = False):
        if len(entity_ids) == 0: return
        kept = np.setdiff1d(np.arange(len(self.entities)), entity_ids)
        if reindex_paths:
            reindex       = np.arange(len(self.entities))
            reindex[kept] = np.arange(len(kept))
            self.paths    = [reindex[p] for p in self.paths]
        self.entities = np.array(self.entities)[kept]

    def remove_duplicate_entities(self):
        entity_hashes = np.array([i.hash() for i in self.entities])
        unique        = unique_rows(entity_hashes)
        if len(unique) != len(self.entities):
            self.entities = np.array(self.entities)[unique]

    def vertex_graph(self, return_closed=False):
        return vertex_graph(self.entities, return_closed)

    def generate_closed_paths(self):
        '''
        Paths are lists of entity indices.
        We first generate vertex paths using graph cycle algorithms, 
        and then convert them to entity paths using 
        a frankly worrying number of loops and conditionals...
        
        This will also change the ordering of entity.points in place, so that
        a path may be traversed without having to reverse the entity
        '''
        paths = generate_closed_paths(self.entities, self.vertices)
        self._cache_put('paths', paths)

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
        discrete = discretize_path(self.entities, self.vertices, path)
        return discrete

    def to_dict(self):
        export_entities = [e.todict() for e in self.entities]
        export_object = {'entities' : export_entities, 
                         'vertices' : self.vertices.tolist()}
        return export_object
        
    def process(self):
        tic   = deque([time_function()])
        label = deque()
        for process_function in self.process_functions():
            process_function()  
            tic.append(time_function())
            label.append(process_function.__name__)
        log.debug('%s processed in %f seconds',
                  self.__class__.__name__,
                  tic[-1] - tic[0])
        log.debug('%s', str(np.column_stack((label, np.diff(tic)))))
        return self

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
                                  metadata = new_meta,
                                  process  = False)
        return new_path
   
class Path3D(Path):
    def process_functions(self): 
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices,
                self.generate_closed_paths,
                self.generate_discrete]
               
    def generate_discrete(self):
        discrete = list(map(self.discretize_path, self.paths))
        self._cache_put('discrete', discrete)

    def to_planar(self, normal=None, transform=None):
        '''
        Check to see if current vectors are all coplanar.
        
        If they are, return a Path2D and a transform which will 
        transform the 2D representation back into 3 dimensions
        '''
        
        if transform is None:
            C, N = plane_fit(self.vertices)
            if normal is not None:
                N *= np.sign(np.dot(N, normal))
            to_planar = plane_transform(C,N)
        else:
            to_planar = transform

        vertices  = transform_points(self.vertices, to_planar)
        
        if np.any(np.std(vertices[:,2]) > TOL_MERGE):
            raise NameError('Points aren\'t planar!')
            
        vector = Path2D(entities = deepcopy(self.entities), 
                              vertices = vertices)
        to_3D  = np.linalg.inv(to_planar)

        return vector, to_3D

    def show(self, entities=False):
        if entities: self.plot_entities(show=True)
        else:        self.plot_discrete(show=True)

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
            vertices = self.vertices[entity.points]
            axis.plot(*vertices.T)        
        if show: plt.show()

class Path2D(Path):
    def process_functions(self): 
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.generate_closed_paths,
                self.generate_discrete,
                self.generate_enclosure_tree]
               
    @property
    def body_count(self):
        return len(self.root)

    def generate_discrete(self):
        '''
        Turn a vector path consisting of entities of any type into polygons
        Uses shapely.geometry Polygons to populate self.polygons
        '''
        def path_to_polygon(path):
            discrete = discretize_path(self.entities, self.vertices, path)
            return Polygon(discrete)
        polygons = np.array(map(path_to_polygon, self.paths))
        self._cache_put('polygons', polygons)

    def generate_enclosure_tree(self):
        root, enclosure = polygons_enclosure_tree(self.polygons)
        self._cache_put('root',      root)
        self._cache_put('enclosure', enclosure.to_undirected())

    @property
    def polygons_full(self):
        result = self._cache_get('polygons_full')
        if result: return result
        result = [None] * len(self.root)
        for index, root in enumerate(self.root):
            hole_index = self.connected_paths(root, include_self=False)
            holes = [np.array(p.exterior.coords) for p in np.array(self.polygons)[hole_index]]
            shell = np.array(self.polygons[root].exterior.coords)
            result[index] = Polygon(shell  = shell,
                                    holes  = holes)
        self._cache_put('polygons_full', result)
        return result
        
    def connected_paths(self, path_id, include_self = False):
        paths = nx.node_connected_component(self.enclosure, path_id)
        if include_self: 
            return np.array(paths)
        return np.setdiff1d(paths, [path_id])
        
    def split(self):
        result        = [None] * len(self.root)
        paths    = np.array(self.paths)
        polygons = np.array(self.polygons)
        for i, root in enumerate(self.root):
            current_entities = deque()
            new_paths        = deque()
            connected = self.connected_paths(root, include_self=True)
            paths = self.paths[[connected]]
            for path in paths:
                new_paths.append(np.arange(len(path)) + len(current_entities))
                current_entities.extend(path)
            result[i] = Path2D(entities = self.entities[[list(current_entities)]],
                                     vertices = self.vertices,
                                     process  = False)
            result[i].paths    = np.array(new_paths)
            result[i].polygons = self.polygons[[connected]]
            result[i].metadata = self.metadata
            result[i].generate_enclosure_tree()
        return result

    def show(self):
        import matplotlib.pyplot as plt
        self.plot_discrete(show=True)
     
    def plot_discrete(self, show=False, transform=None):
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        def plot_transformed(vertices, color='g'):
            if transform is None: 
                plt.plot(*vertices.T, color=color)
            else:
                transformed = transform_points(vertices, transform)
                plt.plot(*transformed.T, color=color)
        for i, polygon in enumerate(self.polygons):
            color = ['g','r'][i in self.root]
            plot_transformed(np.column_stack(polygon.boundary.xy), color=color)
        if show: plt.show()

    def plot_entities(self, show=False):
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        eformat = {'Line0':   {'color':'g', 'linewidth':1}, 
                   'Arc0':    {'color':'r', 'linewidth':1}, 
                   'Arc1':    {'color':'b', 'linewidth':1}}
        for entity in self.entities:
            discrete = entity.discrete(self.vertices)
            plt.plot(discrete[:,0], 
                     discrete[:,1], 
                     **eformat[entity.__class__.__name__ + str(int(entity.closed))])
        if show: plt.show()

    def identifier(self):
        polygons = self.polygons_full
        if len(polygons) != 1: 
            raise NameError('Identifier only valid for single body')
        return [polygons[0].area, 
                polygons[0].length, 
                polygons[0].__hash__()*1e-3]

def vertex_graph(entities, return_closed=False):
    graph  = nx.Graph()
    closed = deque()
    for index, entity in enumerate(entities):
        if return_closed and entity.closed: 
            closed.append(index)
        else:             
            graph.add_edges_from(entity.nodes(), 
                                 entity_index = index)
    if return_closed:
        return graph, np.array(closed)
    return graph

def generate_closed_paths(entities, vertices):
    '''
    Paths are lists of entity indices.
    We first generate vertex paths using graph cycle algorithms, 
    and then convert them to entity paths using 
    a frankly worrying number of loops and conditionals...

    This will also change the ordering of entity.points in place, so that
    a path may be traversed without having to reverse the entity
    '''
    def entity_direction(a,b):
        if   a[0] == b[0]: return  1
        elif a[0] == b[1]: return -1
        elif a[1] == b[1]: return  1
        elif a[1] == b[0]: return -1
        else: raise NameError('Can\'t determine direction, noncontinous path!')

    graph, closed = vertex_graph(entities, return_closed=True)
    paths = deque()        
    paths.extend(np.reshape(closed, (-1,1)))

    vertex_paths = np.array(nx.cycles.cycle_basis(graph))
    
    #all of the following is for converting vertex paths to entity paths
    for idx, vertex_path in enumerate(vertex_paths):           
        #we have removed all closed entities, so paths MUST have more than 2 vertices
        if len(vertex_path) < 2: continue
        # vertex path contains 'nodes', which are always on the path,
        # regardless of entity type. 
        # this is essentially a very coarsely discretized polygon
        # thus it is valid to compute if the path is counter-clockwise on these
        # vertices relatively cheaply, and reverse the path if it is clockwise
        ccw_dir             = is_ccw(vertices[[np.append(vertex_path, vertex_path[0])]])*2 - 1
        current_entity_path = deque()

        for i in range(len(vertex_path)+1):
            path_pos = np.mod(np.arange(2) + i, len(vertex_path))
            vertex_index = np.array(vertex_path)[[path_pos]]
            entity_index = graph.get_edge_data(*vertex_index)['entity_index']
            if ((len(current_entity_path) == 0) or 
                ((current_entity_path[-1] != entity_index) and 
                 (current_entity_path[0]  != entity_index))):
                entity     = entities[entity_index]
                endpoints  = entity.end_points() 
                direction  = entity_direction(vertex_index, endpoints) * ccw_dir
                current_entity_path.append(entity_index)
                entity.points = entity.points[::direction]
        paths.append(list(current_entity_path)[::ccw_dir])
    paths  = np.array(paths)
    return paths

def discretize_path(entities, vertices, path):
    '''
    Return a (n, dimension) list of vertices. 
    Samples arcs/curves to be line segments
    '''
    pathlen  = len(path)
    if pathlen == 0:  raise NameError('Cannot discretize empty path!')
    if pathlen == 1:  return np.array(entities[path[0]].discrete(vertices))
    discrete = deque()
    for i, entity_id in enumerate(path):
        last    = (i == (pathlen - 1))
        current = entities[entity_id].discrete(vertices)
        slice   = (int(last) * len(current)) + (int(not last) * -1)
        discrete.extend(current[:slice])
    return np.array(discrete)    
