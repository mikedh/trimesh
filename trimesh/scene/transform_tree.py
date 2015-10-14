import numpy as np
import time

from networkx import DiGraph, shortest_path, NetworkXNoPath

from ..transformations import quaternion_matrix, rotation_matrix
from ..constants       import TransformError


class TransformTree:
    '''
    A feature complete if not particularly optimized implementation of a transform graph.
    
    Few allowances are made for thread safety, caching, or enforcing graph structure.
    '''

    def __init__(self, base_frame='world'):
        self._transforms = DiGraph()
        self._parents    = {}
        self._paths      = {}
        self._is_changed = False
        self.base_frame  = base_frame

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

        if self._transforms.has_edge(frame_from, frame_to):
            self._transforms.edge[frame_from][frame_to]['matrix'] = matrix
            self._transforms.edge[frame_from][frame_to]['time']   = time.time()
        else:
            # since the connectivity has changed, throw out previously computed
            # paths through the transform graph so queries compute new shortest paths
            # we could only throw out transforms that are connected to the new edge,
            # but this is less bookeeping at the expensive of being slower. 
            self._paths = {}
            self._transforms.add_edge(frame_from, 
                                      frame_to, 
                                      matrix = matrix, 
                                      time   = time.time())
        self._is_changed = True

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
        path, inverted = self._get_path(frame_from, frame_to)
        for i in range(len(path) - 1):
            matrix = self._transforms.get_edge_data(path[i], 
                                                    path[i+1])['matrix']
            transform = np.dot(transform, matrix)
        if inverted:
            transform = np.linalg.inv(transform)
        return transform

    def clear(self):
        self._transforms = DiGraph()
        self._paths      = {}
        
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
            path = shortest_path(self._transforms, frame_from, frame_to)
            inverted = False
        except NetworkXNoPath:
            path = shortest_path(self._transforms, frame_to, frame_from)
            inverted = True
        self._paths[(frame_from, frame_to)] = (path, inverted)
        return path, inverted
