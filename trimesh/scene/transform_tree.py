import numpy as np
import time

import networkx as nx
from networkx import DiGraph, shortest_path, NetworkXNoPath, to_edgelist

from ..transformations import quaternion_matrix, rotation_matrix
from ..constants       import TransformError


class EnforcedForest(nx.DiGraph):    
    def __init__(self, *args, **kwargs):
        self.flags = {'strict' : False,
                      'assert_forest' : False}
        for k, v in self.flags.items():
            if k in kwargs:
                self.flags[k] = kwargs[k]
                kwargs.pop(k, None)

        super(self.__class__, self).__init__(*args, **kwargs)
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
                self.remove_path(path)
                changed = True
            except (nx.NetworkXError, nx.NetworkXNoPath):
                pass

        self._undirected.add_edge(u,v)
        super(self.__class__, self).add_edge(u, v, *args, **kwargs)
      
        if self.flags['assert_forest']:
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

    def remove_path(self, path):
        ebunch = np.array([[path[0], path[1]]])
        ebunch = np.vstack((ebunch, np.fliplr(ebunch)))
        self.remove_edges_from(ebunch)

class TransformTree:
    def __init__(self, base_frame='world'):
        self.transforms = EnforcedForest()
        self._paths     = {}
        self.base_frame = base_frame

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
        quaternion:  (4) quatenion
        axis:        (3) array
        angle:       float, radians
        translation: (3) array
        '''
        if frame_from is None:
            frame_from = self.base_frame

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

        changed = self.transforms.add_edge(frame_from, 
                                           frame_to,
                                           attr_dict = {'matrix' : matrix,
                                                        'times'  : time.time()})
        if changed:
            self._paths = {}

    def export(self):
        export = to_edgelist(self.transforms)
        for e in export:
            e[2]['matrix'] = np.array(e[2]['matrix']).tolist()
        return export

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

        tic = time.time()
        if frame_from is None:
            frame_from = self.base_frame
        transform = np.eye(4)
        path, inverted = self._get_path(frame_from, frame_to)
        for i in range(len(path) - 1):
            matrix = self.transforms.get_edge_data(path[i], 
                                                   path[i+1])['matrix']
            transform = np.dot(transform, matrix)
        if inverted:
            transform = np.linalg.inv(transform)
        return transform

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
        inverted: boolean flag, whether the path is traversing stored
                  matrices forwards or backwards. 
        '''
        try: 
            return self._paths[(frame_from, frame_to)]
        except KeyError:
            return self._generate_path(frame_from, frame_to)

    def _generate_path(self, 
                       frame_from, 
                       frame_to):
        '''
        Generate a path between two frames.
        
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
        inverted: boolean flag, whether the path is traversing stored
                  matrices forwards or backwards. 
        '''
        try: 
            path = shortest_path(self.transforms, frame_from, frame_to)
            inverted = False
        except NetworkXNoPath:
            path = shortest_path(self.transforms, frame_to, frame_from)
            inverted = True
        self._paths[(frame_from, frame_to)] = (path, inverted)
        return path, inverted
