'''
entities.py: basic geometric primitives

Design intent: only store references to vertex indices and pass the vertex
               array back to functions that require it.
               This keeps all vertices in one external list.
'''

import numpy as np

from .arc import discretize_arc, arc_center
from .curve import discretize_bezier, discretize_bspline
from ..util import replace_references

_HASH_LENGTH = 5


class Entity(object):

    def __init__(self,
                 points,
                 closed=None):
        self.points = np.asanyarray(points)
        if closed is not None:
            self.closed = closed

    @property
    def _class_id(self):
        '''
        Return an integer that is unique to the class type.
        Note that this implementation will fail if a class is defined
        that starts with the same letter as an existing class.
        Since this function is called a lot, it is a tradeoff between
        speed and robustness where speed won.
        '''
        return ord(self.__class__.__name__[0])

    @property
    def hash(self):
        '''
        Returns a string unique to the entity.
        If two identical entities exist, they can be removed
        by comparing the string returned by this function.
        '''
        hash = np.zeros(_HASH_LENGTH, dtype=np.int)
        hash[-2:] = self._class_id, int(self.closed)
        points_count = np.min([3, len(self.points)])
        hash[0:points_count] = np.sort(self.points)[-points_count:]
        return hash

    def to_dict(self):
        '''
        Returns a dictionary with all of the information about the entity.
        '''
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'closed': self.closed}

    def rereference(self, replacement):
        '''
        Given a replacement dictionary, change points to reflect the dictionary.
        eg, if replacement = {0:107}, self.points = [0,1902] becomes [107, 1902]
        '''
        self.points = replace_references(self.points, replacement)

    @property
    def closed(self):
        '''
        If the first point is the same as the end point, the entity is closed
        '''
        closed = (len(self.points) > 2 and
                  np.equal(self.points[0], self.points[-1]))
        return closed

    @property
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
                                self.points)).reshape(-1)[1:-1].reshape((-1, 2))

    @property
    def end_points(self):
        '''
        Returns the first and last points. Also note that if you
        define a new entity class where the first and last vertices
        in self.points aren't the endpoints of the curve you need to
        implement this function for your class.

        self.points = [0,1,2]
        returns:      [0,2]
        '''
        return self.points[[0, -1]]

    @property
    def is_valid(self):
        return True

    def reverse(self, direction=-1):
        '''
        Reverse the current entity.
        '''
        self.points = self.points[::direction]


class Line(Entity):
    '''
    A line or poly-line entity
    '''

    def discrete(self, vertices, scale=1.0):
        return vertices[self.points]

    @property
    def is_valid(self):
        valid = np.any((self.points - self.points[0]) != 0)
        return valid


class Arc(Entity):

    @property
    def closed(self):
        if hasattr(self, '_closed'):
            return self._closed
        return False

    @closed.setter
    def closed(self, value):
        self._closed = bool(value)

    def discrete(self, vertices, scale=1.0):
        return discretize_arc(vertices[self.points],
                              close=self.closed,
                              scale=scale)

    def center(self, vertices):
        return arc_center(vertices[self.points])


class Curve(Entity):

    @property
    def _class_id(self):
        return sum([ord(i) for i in self.__class__.__name__])

    @property
    def nodes(self):
        return [[self.points[0],
                 self.points[1]],
                [self.points[1],
                 self.points[-1]]]


class Bezier(Curve):

    def discrete(self, vertices, scale=1.0):
        return discretize_bezier(vertices[self.points], scale=scale)


class BSpline(Curve):

    def __init__(self, points, knots, closed=None):
        self.points = points
        self.knots = knots

    def discrete(self, vertices, count=None, scale=1.0):
        result = discretize_bspline(control=vertices[self.points],
                                    knots=self.knots,
                                    count=count,
                                    scale=scale)
        return result

    def to_dict(self):
        '''
        Returns a dictionary with all of the information about the entity.
        '''
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'knots': self.knots.tolist(),
                'closed': self.closed}
