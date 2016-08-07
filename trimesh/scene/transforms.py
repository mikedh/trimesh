import numpy as np
import time

import networkx as nx

from ..transformations import quaternion_matrix, rotation_matrix

from .. import util

class TransformForest:
    def __init__(self, base_frame='world'):
        self.transforms = EnforcedForest()
        self.base_frame = base_frame
        self._paths     = {}
        self._updated   = time.time()

    def update(self, 
               frame_to,
               frame_from = None,
               **kwargs):
        '''
        Update a transform in the tree.

        Arguments
        ---------
        frame_from: hashable object, usually a string (eg 'world').
                    If left as None it will be set to self.base_frame
        frame_to:   hashable object, usually a string (eg 'mesh_0')
        
        Additional kwargs (can be used in combinations)
        --------- 
        matrix:      (4,4) array 
        quaternion:  (4) quaternion
        axis:        (3) array
        angle:       float, radians
        translation: (3) array
        '''
        if frame_from is None:
            frame_from = self.base_frame
        matrix  = kwargs_to_matrix(**kwargs)
        changed = self.transforms.add_edge(frame_from, 
                                           frame_to,
                                           attr_dict = {'matrix' : matrix,
                                                        'time'   : time.time()})
        if changed:
            self._paths = {}
        self._updated = time.time()

    def md5(self):
        '''
        MD5 of transforms.

        Currently only hashing update time.
        '''
        result = util.md5_object(str(int(self._updated * 1000)).encode('utf-8'))
        return result

    def export(self):
        export = nx.to_edgelist(self.transforms)
        for e in export:
            e[2]['matrix'] = np.array(e[2]['matrix']).tolist()
        return export

    def load(self, edgelist):
        for edge in edgelist:
            self.transforms.add_edge(edge[0], edge[1], **edge[2])

    def get(self,
            frame_to,
            frame_from = None):
        '''
        Get the transform from one frame to another, assuming they are connected
        in the transform tree. 

        If the frames are not connected a NetworkXNoPath error will be raised.

        Arguments
        ---------
        frame_from: hashable object, usually a string (eg 'world').
                    If left as None it will be set to self.base_frame
        frame_to:   hashable object, usually a string (eg 'mesh_0')

        Returns
        ---------
        transform:  (4,4) homogenous transformation matrix
        '''

        if frame_from is None:
            frame_from = self.base_frame
        transform = np.eye(4)
        path = self._get_path(frame_from, frame_to)

        for i in range(len(path) - 1):
            data, direction = self.transforms.get_edge_data_direction(path[i],
                                                                      path[i+1])
            matrix = data['matrix']
            if direction < 0: 
                matrix = np.linalg.inv(matrix)
            transform = np.dot(transform, matrix)
        return transform

    def __getitem__(self, key):
        return self.get(key)
        
    def __setitem__(self, key, value):
        value = np.asanyarray(value)
        if value.shape != (4,4):
            raise ValueError('Matrix must be specified!')
        return self.update(key, matrix=value)
        
    def clear(self):
        self.transforms = EnforcedForest()
        self._paths     = {}
        
    def _get_path(self, 
                  frame_from,
                  frame_to):
        '''
        Find a path between two frames, either from cached paths or
        from the transform graph. 
        
        Arguments
        ---------
        frame_from: a frame key, usually a string 
                    example: 'world'
        frame_to:   a frame key, usually a string 
                    example: 'mesh_0'

        Returns
        ----------
        path: (n) list of frame keys
              example: ['mesh_finger', 'mesh_hand', 'world']
        '''
        key = (frame_from, frame_to)
        if not (key in self._paths):
            path =  self.transforms.shortest_path_undirected(frame_from, 
                                                             frame_to)
            self._paths[key] = path
        return self._paths[key]

class EnforcedForest(nx.DiGraph):    
    def __init__(self, *args, **kwargs):
        self.flags = {'strict'        : False,
                      'assert_forest' : False}
        
        for k, v in self.flags.items():
            if k in kwargs:
                self.flags[k] = bool(kwargs[k])
                kwargs.pop(k, None)

        super(self.__class__, self).__init__(*args, **kwargs)
        # keep a second parallel but undirected copy of the graph
        # all of the networkx methods for turning a directed graph
        # into an undirected graph are quite slow, so we do minor bookkeeping 
        self._undirected = nx.Graph()

    def add_edge(self, u, v, *args, **kwargs):
        changed = False
        if u == v:
            if self.flags['strict']:
                raise ValueError('Edge must be between two unique nodes!')
            return changed
        if self._undirected.has_edge(u, v):
            self.remove_edges_from([[u, v], [v,u]])
        elif len(self.nodes()) > 0:
            try: 
                path = nx.shortest_path(self._undirected, u, v)
                if self.flags['strict']:
                    raise ValueError('Multiple edge path exists between nodes!')
                self.disconnect_path(path)
                changed = True
            except (nx.NetworkXError, nx.NetworkXNoPath):
                pass
        self._undirected.add_edge(u,v)
        super(self.__class__, self).add_edge(u, v, *args, **kwargs)
      
        if self.flags['assert_forest']:
            # this is quite slow but makes very sure structure is correct 
            # so is mainly used for testing
            assert nx.is_forest(nx.Graph(self))
            
        return changed

    def add_edges_from(self, *args, **kwargs):
        raise ValueError('EnforcedTree requires add_edge method to be used!')

    def add_path(self, *args, **kwargs):
        raise ValueError('EnforcedTree requires add_edge method to be used!')

    def remove_edge(self, *args, **kwargs):
        super(self.__class__, self).remove_edge(*args, **kwargs)
        self._undirected.remove_edge(*args, **kwargs)
        
    def remove_edges_from(self, *args, **kwargs):
        super(self.__class__, self).remove_edges_from(*args, **kwargs)
        self._undirected.remove_edges_from(*args, **kwargs)

    def disconnect_path(self, path):
        ebunch = np.array([[path[0], path[1]]])
        ebunch = np.vstack((ebunch, np.fliplr(ebunch)))
        self.remove_edges_from(ebunch)

    def shortest_path_undirected(self, u, v):
        path = nx.shortest_path(self._undirected, u, v)
        return path

    def get_edge_data_direction(self, u, v):
        if self.has_edge(u,v):
            direction = 1
        elif self.has_edge(v,u):
            direction = -1
        else: 
            raise ValueError('Edge doesnt exist!')
        data = self.get_edge_data(*[u,v][::direction])
        return data, direction
        
def path_to_edges(path):
    '''
    Turn an (n) path into a (2(n-1)) set of edges
    '''
    return np.column_stack((path, path)).reshape(-1)[1:-1].reshape((-1,2))

def kwargs_to_matrix(**kwargs):
    '''
    Turn a set of keyword arguments into a transformation matrix. 
    '''
    matrix = np.eye(4)
    if 'matrix' in kwargs:
        # a matrix takes precedence over other options
        matrix = kwargs['matrix']
    elif 'quaternion' in kwargs:
        matrix = quaternion_matrix(kwargs['quaternion'])
    elif ('axis' in kwargs) and ('angle' in kwargs):
        matrix = rotation_matrix(kwargs['angle'],
                                 kwargs['axis'])
    else: 
        raise ValueError('Couldn\'t update transform!')

    if 'translation' in kwargs:
        # translation can be used in conjunction with any of the methods of 
        # specifying transforms. In the case a matrix and translation are passed,
        # we add the translations together rather than picking one. 
        matrix[0:3,3] += kwargs['translation']
    return matrix
        
