'''
entities.py: basic geometric primitives

Design intent: only store references to vertex indices and pass the vertex
               array back to functions that require it. 
               This keeps all vertices in one external list.
'''

import numpy as np

from .constants import *
from .arc       import discretize_arc, arc_center
from ..points   import unitize

_HASH_LENGTH = 5

class Entity:
    def __init__(self, 
                 points, 
                 closed = False):
        self.points = np.array(points)
        self.closed = closed
        
    def hash(self):
        '''
        Returns a string unique to the entity.
        If two identical entities exist, they can be removed
        by comparing the string returned by this function.
        '''
        hash = np.zeros(_HASH_LENGTH, dtype=np.int)
        hash[-2:] = self._CLASS_ID, int(self.closed)
        hash[0:len(self.points)] = np.sort(self.points)
        return hash
        
    def to_dict(self):
        '''
        Returns a dictionary with all of the information about the entity. 
        '''
        return {'type'  : self.__class__.__name__, 
                'points': self.points.tolist(),
                'closed': self.closed}
                
    def rereference(self, replacement):
        '''
        Given a replacement dictionary, change points to reflect the dictionary.
        eg, if replacement = {0:107}, self.points = [0,1902] becomes [107, 1902]
        '''
        self.points = replace_references(self.points, replacement)
        
    def nodes(self):
        '''
        Returns an (n,2) list of nodes, or vertices on the path.
        Note that this generic class function assumes that all of the reference 
        points are on the path, which is true for lines and three point arcs. 
        If you were to define another class where that wasn't the case
        (for example, the control points of a bezier curve), 
        you would need to implement an entity- specific version of this function.
        
        The purpose of having a list of nodes is so that they can then be added 
        as edges to a graph, so we can use functions to check connectivity, 
        extract paths, etc.  
        
        The slicing on this function is essentially just tiling points
        so the first and last vertices aren't repeated. Example:
        
        self.points = [0,1,2]
        returns:      [[0,1], [1,2]]
        '''
        return np.column_stack((self.points,
                                self.points)).reshape(-1)[1:-1].reshape((-1,2))
                       
    def end_points(self):
        '''
        Returns the first and last points. Also note that if you
        define a new entity class where the first and last vertices
        in self.points aren't the endpoints of the curve you need to 
        implement this function for your class.
        
        self.points = [0,1,2]
        returns:      [0,2]
        '''
        return self.points[[0,-1]]
            
class Arc(Entity):
    _CLASS_ID = 1
    def discrete(self, vertices):
        return discretize_arc(vertices[[self.points]], 
                              close = self.closed)
                              
    def center(self, vertices):
        return arc_center(vertices[[self.points]])
        
    def offset(self, vertices, distance, normal=None):
        new_points = arc_offset(vertices[[self.points]], distance)
        return new_points
        
    def tangents(self, vertices):
        return arc_tangents(vertices[[self.points]])

class Line(Entity):
    _CLASS_ID = 0
    def discrete(self, vertices):
        return vertices[[self.points]]
        
    def offset(self, vertices, distance, normal=[0,0,1]):
        points = vertices[[self.points]]
        line_vec = np.diff(points)
        perp_vec = unitize(np.cross(line_vec, normal))
        new_points = points + perp_vec*distance
        return new_points 
        
    def tangents(self, vertices):
        points   = vertices[[self.points]]
        tangents = np.tile(unitize(np.diff(points, axis=0)), (2,1))
        return tangents
        
        
def is_ccw(points):
    '''https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    
    '''
    xd = np.diff(points[:,0])
    yd = np.sum(np.column_stack((points[:,1], 
                                 points[:,1])).reshape(-1)[1:-1].reshape((-1,2)), axis=1)
    area = np.sum(xd*yd)*.5
    return area < 0
        
def replace_references(data, reference_dict):
    # Replace references in place
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view
