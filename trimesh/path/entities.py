"""
entities.py
--------------

Basic geometric primitives which only store references to
vertex indices rather than vertices themselves.
"""

import numpy as np

import copy

from .arc import discretize_arc, arc_center
from .curve import discretize_bezier, discretize_bspline

from .. import util

_HASH_LENGTH = 5


class Entity(object):

    def __init__(self,
                 points,
                 closed=None,
                 layer=None,
                 **kwargs):

        self.points = np.asanyarray(points)
        if closed is not None:
            self.closed = closed
        self.layer = layer
        self.kwargs = kwargs

    @property
    def _class_id(self):
        """
        Return an integer that is unique to the class type.
        Note that this implementation will fail if a class is defined
        that starts with the same letter as an existing class.
        Since this function is called a lot, it is a tradeoff between
        speed and robustness where speed won.
        """
        return ord(self.__class__.__name__[0])

    @property
    def hash(self):
        """
        Returns a string unique to the entity.
        If two identical entities exist, they can be removed
        by comparing the string returned by this function.
        """
        hash = np.zeros(_HASH_LENGTH, dtype=np.int)
        hash[-2:] = self._class_id, int(self.closed)
        points_count = np.min([3, len(self.points)])
        hash[0:points_count] = np.sort(self.points)[-points_count:]
        return hash

    def to_dict(self):
        """
        Returns a dictionary with all of the information about the entity.
        """
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'closed': self.closed}

    @property
    def closed(self):
        """
        If the first point is the same as the end point, the entity is closed
        """
        closed = (len(self.points) > 2 and
                  np.equal(self.points[0], self.points[-1]))
        return closed

    @property
    def nodes(self):
        """
        Returns an (n,2) list of nodes, or vertices on the path.
        Note that this generic class function assumes that all of the
        reference points are on the path which is true for lines and
        three point arcs.

        If you were to define another class where that wasn't the case
        (for example, the control points of a bezier curve),
        you would need to implement an entity- specific version of this
        function.

        The purpose of having a list of nodes is so that they can then be
        added as edges to a graph so we can use functions to check
        connectivity, extract paths, etc.

        The slicing on this function is essentially just tiling points
        so the first and last vertices aren't repeated. Example:

        self.points = [0,1,2]
        returns:      [[0,1], [1,2]]
        """
        return np.column_stack((self.points,
                                self.points)).reshape(
                                    -1)[1:-1].reshape((-1, 2))

    @property
    def end_points(self):
        """
        Returns the first and last points. Also note that if you
        define a new entity class where the first and last vertices
        in self.points aren't the endpoints of the curve you need to
        implement this function for your class.

        self.points = [0,1,2]
        returns:      [0,2]
        """
        return self.points[[0, -1]]

    @property
    def is_valid(self):
        """
        Is the current entity valid.
        """
        return True

    def reverse(self, direction=-1):
        """
        Reverse the current entity in place.
        """
        self.points = self.points[::direction]

    def bounds(self, vertices):
        """
        Return the AABB of the current entity.

        Parameters
        -----------
        vertices: (n,dimension) float, vertices in space

        Returns
        -----------
        bounds: (2, dimension) float, (min, max) coordinate of AABB
        """
        bounds = np.array([vertices[self.points].min(axis=0),
                           vertices[self.points].max(axis=0)])
        return bounds

    def length(self, vertices):
        """
        Return the total length of the entity.

        Returns
        ---------
        length: float, total length of entity
        """
        length = ((np.diff(self.discrete(vertices),
                           axis=0)**2).sum(axis=1)**.5).sum()
        return length

    def explode(self):
        """
        Split the entity into multiple entities.
        """
        return [self]

    def copy(self):
        """
        Return a copy of the current entity.
        """
        return copy.deepcopy(self)


class Line(Entity):
    """
    A line or poly-line entity
    """

    def discrete(self, vertices, scale=1.0):
        """
        Discretize into a world- space path.

        Parameters
        ------------
        vertices: (n, dimension) float, points in space
        scale:    float, size of overall scene for numerical comparisons

        Returns
        -------------
        discrete: (m, dimension) float, linear path in space
        """
        discrete = vertices[self.points]
        return discrete

    @property
    def is_valid(self):
        valid = np.any((self.points - self.points[0]) != 0)
        return valid

    def explode(self):
        """
        If the current Line entity consists of multiple lines, break it
        up into n Line entities.

        Returns
        ----------
        exploded: (n,) Line entities
        """
        points = np.column_stack((self.points,
                                  self.points)).ravel()[1:-1].reshape((-1, 2))
        exploded = [Line(i) for i in points]
        return exploded


class Arc(Entity):

    @property
    def closed(self):
        """
        A boolean flag for whether the arc is closed (a circle) or not.

        Returns
        ----------
        closed: bool, if true arc will be a closed circle in space
        """
        if hasattr(self, '_closed'):
            return self._closed
        return False

    @closed.setter
    def closed(self, value):
        self._closed = bool(value)

    def discrete(self, vertices, scale=1.0):
        """
        Discretize the arc entity into line sections.

        Parameters
        ------------
        vertices: (n, dimension) float, points in space
        scale:    float, size of overall scene for numerical comparisons

        Returns
        -------------
        discrete: (m, dimension) float, linear path in space
        """
        discrete = discretize_arc(vertices[self.points],
                                  close=self.closed,
                                  scale=scale)
        return discrete

    def center(self, vertices):
        """
        Return the center information about the arc entitiy.

        Parameters
        -------------
        vertices: (n,dimension) float, vertices in space

        Returns
        -------------
        info: dict, with keys:
                            'radius'
                            'center'
        """
        info = arc_center(vertices[self.points])
        return info

    def bounds(self, vertices):
        """
        Return the AABB of the arc entity.

        Parameters
        -----------
        vertices: (n,dimension) float, vertices in space

        Returns
        -----------
        bounds: (2, dimension) float, (min, max) coordinate of AABB
        """
        if util.is_shape(vertices, (-1, 2)) and self.closed:
            # if we have a closed arc (a circle), we can return the actual bounds
            # this only works in two dimensions, otherwise this would return the
            # AABB of an sphere
            info = self.center(vertices)
            bounds = np.array([info['center'] - info['radius'],
                               info['center'] + info['radius']])
        else:
            # since the AABB of a partial arc is hard, approximate
            # the bounds by just looking at the discrete values
            discrete = self.discrete(vertices)
            bounds = np.array([discrete.min(axis=0),
                               discrete.max(axis=0)])
        return bounds


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

    def discrete(self, vertices, scale=1.0, count=None):
        return discretize_bezier(vertices[self.points],
                                 count=count,
                                 scale=scale)


class BSpline(Curve):

    def __init__(self, points,
                 knots,
                 closed=None,
                 layer=None,
                 **kwargs):
        self.points = points
        self.knots = knots
        self.layer = layer
        self.kwargs = kwargs

    def discrete(self, vertices, count=None, scale=1.0):
        result = discretize_bspline(control=vertices[self.points],
                                    knots=self.knots,
                                    count=count,
                                    scale=scale)
        return result

    def to_dict(self):
        """
        Returns a dictionary with all of the information about the entity.
        """
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'knots': self.knots.tolist(),
                'closed': self.closed}
