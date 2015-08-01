'''
path.py

A library designed to work with vector paths.
'''

import numpy as np
import networkx as nx

from shapely.geometry import Polygon, Point
from copy import deepcopy
from collections import deque

from .simplify  import simplify
from .polygons  import polygons_enclosure_tree, is_ccw
from .constants import *

from ..points   import plane_fit, transform_points
from ..geometry import plane_transform
from ..grouping import unique_rows
from ..units    import unit_conversion

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
        self.metadata = dict()

        if metadata.__class__.__name__ == 'dict':
            self.metadata.update(metadata)

        self._cache = {}

    @property
    def _cache_ok(self):
        processing = ('processing' in self._cache and 
                      self._cache['processing'])
        entity_ok = ('entity_count' in self._cache and 
                     (len(self.entities) == self._cache['entity_count']))
        ok = processing or entity_ok
        return ok

    def _cache_clear(self):
        self._cache = {}

    def _cache_verify(self):
        if not self._cache_ok:
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
    def polygons_closed(self):
        return self._cache_get('polygons_closed')

    @property
    def root(self):
        return self._cache_get('root')

    @property
    def enclosure(self):
        return self._cache_get('enclosure')

    @property
    def discrete(self):
        return self._cache_get('discrete')

    @property
    def scale(self):
        return np.max(np.ptp(self.vertices, axis=0))

    @property
    def bounds(self):
        return np.vstack((np.min(self.vertices, axis=0),
                          np.max(self.vertices, axis=0)))
    @property
    def box_size(self):
        return np.diff(self.bounds, axis=0)[0]

    @property
    def units(self):
        if 'units' in self.metadata:
                return self.metadata['units']
        else:
            return None

    def set_units(self, desired):
        if self.units is None:
            log.error('Current document doesn\'t have units specified!')
        else:
            conversion = unit_conversion(self.units,
                                         desired)
            self.vertices *= conversion
            self._cache_clear()
        self.metadata['units'] = desired
        
    def transform(self, transform):
        self._cache = {}
        self.vertices = transform_points(self.vertices, transform)

    def rezero(self):
        self._cache = {}
        self.vertices -= self.vertices.min(axis=0)
        
    def merge_vertices(self):
        '''
        Merges vertices which are identical and replaces references
        '''
        unique, inverse = unique_rows(self.vertices, digits=TOL_MERGE_DIGITS)
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

    def remove_duplicate_entities(self):
        entity_hashes   = np.array([i.hash() for i in self.entities])
        unique, inverse = unique_rows(entity_hashes)
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
        export_entities = [e.to_dict() for e in self.entities]
        export_object   = {'entities' : export_entities, 
                           'vertices' : self.vertices.tolist()}
        return export_object
        
    def process(self):
        self._cache['processing'] = True
        tic = time_function()        
        for func in self._process_functions():
            func()
        toc = time_function()
        self._cache['processing']   = False
        self._cache['entity_count'] = len(self.entities)
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
                                  metadata = new_meta)
        return new_path
   
class Path3D(Path):
    def _process_functions(self): 
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.remove_unreferenced_vertices,
                self.generate_closed_paths,
                self.generate_discrete]
               
    def generate_discrete(self):
        discrete = list(map(self.discretize_path, self.paths))
        self._cache_put('discrete', discrete)

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
            to_2D = plane_transform(C,N)
 
        flat = transform_points(self.vertices, to_2D)
        
        if check and np.any(np.std(flat[:,2]) > TOL_PLANAR):
            log.error('points have z with deviation %f', np.std(flat[:,2]))
            raise NameError('Points aren\'t planar!')
            
        vector = Path2D(entities = deepcopy(self.entities), 
                        vertices = flat[:,0:2])
        to_3D  = np.linalg.inv(to_2D)

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
    def _process_functions(self): 
        return [self.merge_vertices,
                self.remove_duplicate_entities,
                self.generate_closed_paths,
                self.generate_discrete,
                self.generate_enclosure_tree]
               
    @property
    def body_count(self):
        return len(self.root)

    @property
    def polygons_full(self):
        cached = self._cache_get('polygons_full')
        if cached:  return cached
        result = [None] * len(self.root)
        for index, root in enumerate(self.root):
            hole_index = self.connected_paths(root, include_self=False)
            holes = [p.exterior.coords for p in self.polygons_closed[hole_index]]
            shell = self.polygons_closed[root].exterior.coords
            result[index] = Polygon(shell  = shell,
                                    holes  = holes)
        self._cache_put('polygons_full', result)
        return result

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
        from ..creation import extrude_polygon
        result = [extrude_polygon(i, height, **kwargs) for i in self.polygons_full]
        if len(result) == 1: 
            return result[0]
        return result

    def generate_discrete(self):
        '''
        Turn a vector path consisting of entities of any type into polygons
        Uses shapely.geometry Polygons to populate self.polygons
        '''
        def path_to_polygon(path):
            discrete = discretize_path(self.entities, self.vertices, path)
            return Polygon(discrete)
        polygons = np.array(list(map(path_to_polygon, self.paths)))
        self._cache_put('polygons_closed', polygons)

    def generate_enclosure_tree(self):
        root, enclosure = polygons_enclosure_tree(self.polygons_closed)
        self._cache_put('root',      root)
        self._cache_put('enclosure', enclosure.to_undirected())

    def connected_paths(self, path_id, include_self = False):
        if len(self.root) == 1:
            path_ids = np.arange(len(self.paths))
        else:
            path_ids = nx.node_connected_component(self.enclosure, path_id)
        if include_self: 
            return np.array(path_ids)
        return np.setdiff1d(path_ids, [path_id])
        
    def simplify(self):
        self._cache = {}
        simplify(self)

    def split(self):
        '''
        If the current Path2D consists of n 'root' curves,
        split them into a list of n Path2D objects
        '''
        if len(self.root) == 1:
            return [deepcopy(self)]
        result   = [None] * len(self.root)
        for i, root in enumerate(self.root):
            connected    = self.connected_paths(root, include_self=True)
            new_root     = np.nonzero(connected == root)[0]
            new_entities = deque()
            new_paths    = deque()
            new_metadata = {'split_2D' : i}
            new_metadata.update(self.metadata)

            for path in self.paths[connected]:
                new_paths.append(np.arange(len(path)) + len(new_entities))
                new_entities.extend(path)
            
            result[i] = Path2D(entities = deepcopy(self.entities[new_entities]),
                               vertices = deepcopy(self.vertices))
            result[i]._cache = {'entity_count' : len(new_entities),
                                'paths'        : np.array(new_paths),
                                'polygons'     : self.polygons_closed[connected],
                                'metadata'     : new_metadata,
                                'root'         : new_root}
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
        for i, polygon in enumerate(self.polygons_closed):
            color = ['g','r'][i in self.root]
            plot_transformed(np.column_stack(polygon.boundary.xy), color=color)
        if show: plt.show()

    def plot_entities(self, show=False):
        import matplotlib.pyplot as plt
        plt.axes().set_aspect('equal', 'datalim')
        eformat = {'Line0'  : {'color':'g', 'linewidth':1}, 
                   'Arc0'   : {'color':'r', 'linewidth':1}, 
                   'Arc1'   : {'color':'b', 'linewidth':1},
                   'Bezier0': {'color':'k', 'linewidth':1}}
        for entity in self.entities:
            discrete = entity.discrete(self.vertices)
            e_key    = entity.__class__.__name__ + str(int(entity.closed))
            plt.plot(discrete[:,0], 
                     discrete[:,1], 
                     **eformat[e_key])
        if show: plt.show()

    def identifier(self):
        polygons = self.polygons_full
        if len(polygons) != 1: 
            raise NameError('Identifier only valid for single body')
        return [polygons[0].area, 
                polygons[0].length, 
                polygons[0].__hash__()*1e-5]

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
