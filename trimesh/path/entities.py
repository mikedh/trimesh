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


class Entity(object):

    def __init__(self,
                 points,
                 closed=None,
                 layer=None,
                 **kwargs):

        self.points = np.asanyarray(points, dtype=np.int64)
        if closed is not None:
            self.closed = closed
        self.layer = layer
        self.kwargs = kwargs

    def to_dict(self):
        """
        Returns a dictionary with all of the information
        about the entity.
        """
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'closed': self.closed}

    @property
    def closed(self):
        """
        If the first point is the same as the end point
        the entity is closed
        """
        closed = (len(self.points) > 2 and
                  self.points[0] == self.points[-1])
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

        Returns
        -----------
        valid : bool
          Is the current entity well formed
        """
        return True

    def reverse(self, direction=-1):
        """
        Reverse the current entity in place.
        """
        if direction < 0:
            self._direction = -1
        else:
            self._direction = 1

    def _orient(self, curve):
        """
        Reverse a curve if a flag is set
        """
        if hasattr(self, '_direction') and self._direction < 0:
            return curve[::-1]
        return curve

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

    def __hash__(self):
        """
        Return a hash that represents the current entity.

        Returns
        ----------
        hashed : int
            Hash of current class name, points, and closed
        """
        hashed = hash(self._bytes())
        return hashed

    def _bytes(self):
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return (self.__class__.__name__.encode('utf-8') +
                    self.points.tobytes())
        else:
            return (self.__class__.__name__.encode('utf-8') +
                    self.points[::-1].tobytes())


class Text(Entity):
    """
    Text to annotate a 2D or 3D path.
    """

    def __init__(self,
                 origin,
                 text,
                 height=None,
                 vector=None,
                 normal=None,
                 align=None):
        """
        An entity for text labels.

        Parameters
        --------------
        origin : int
          Index of a single vertex for text origin
        text : str
          The text to label
        height : float or None
          The height of text
        vector : int or None
          An vertex index for which direction text
          is written along unitized: vector - origin
        normal : int or None
          A vertex index for the plane normal:
          vector is along unitized: normal - origin
        align : (2,) str or None
          Where to draw from for [horizontal, vertical]:
              'center', 'left', 'right'
        """
        # where is text placed
        self.origin = origin

        self.vector = vector
        self.normal = normal

        # how high is the text entity
        self.height = height

        # None or (2,) str
        if align is None:
            # if not set make everything centered
            align = ['center', 'center']
        elif util.is_string(align):
            # if only one is passed set for both
            # horizontal and vertical
            align = [align, align]
        elif len(align) != 2:
            # otherwise raise rror
            raise ValueError('align must be (2,) str')

        if any(i not in ['left', 'right', 'center']
               for i in align):
            print('nah')

        self.align = align

        # make sure text is a string
        if hasattr(text, 'decode'):
            self.text = text.decode('utf-8')
        else:
            self.text = str(text)

    @property
    def origin(self):
        """
        The origin point of the text.

        Returns
        -----------
        origin : int
          Index of vertices
        """
        return self.points[0]

    @origin.setter
    def origin(self, value):
        value = int(value)
        if not hasattr(self, 'points') or self.points.ptp() == 0:
            self.points = np.ones(3, dtype=np.int64) * value
        else:
            self.points[0] = value

    @property
    def vector(self):
        """
        A point representing the text direction
        along the vector: vertices[vector] - vertices[origin]

        Returns
        ----------
        vector : int
          Index of vertex
        """
        return self.points[1]

    @vector.setter
    def vector(self, value):
        if value is None:
            return
        self.points[1] = int(value)

    @property
    def normal(self):
        """
        A point representing the plane normal along the
        vector: vertices[normal] - vertices[origin]

        Returns
        ------------
        normal : int
          Index of vertex
        """
        return self.points[2]

    @normal.setter
    def normal(self, value):
        if value is None:
            return
        self.points[2] = int(value)

    def plot(self, vertices, show=False):
        """
        Plot the text using matplotlib.

        Parameters
        --------------
        vertices : (n, 2) float
          Vertices in space
        show : bool
          If True, call plt.show()
        """
        if vertices.shape[1] != 2:
            raise ValueError('only for 2D points!')

        import matplotlib.pyplot as plt

        # get rotation angle in degrees
        angle = np.degrees(self.angle(vertices))

        plt.text(*vertices[self.origin],
                 s=self.text,
                 rotation=angle,
                 ha=self.align[0],
                 va=self.align[1],
                 size=18)

        if show:
            plt.show()

    def angle(self, vertices):
        """
        If Text is 2D, get the rotation angle in radians.

        Parameters
        -----------
        vertices : (n, 2) float
          Vertices in space referenced by self.points

        Returns
        ---------
        angle : float
          Rotation angle in radians
        """

        if vertices.shape[1] != 2:
            raise ValueError('angle only valid for 2D points!')

        # get the vector from origin
        direction = vertices[self.vector] - vertices[self.origin]
        # get the rotation angle in radians
        angle = np.arctan2(*direction[::-1])

        return angle

    def length(self, vertices):
        return 0.0

    def discrete(self, *args, **kwargs):
        return np.array([])

    @property
    def closed(self):
        return False

    @property
    def is_valid(self):
        return True

    @property
    def nodes(self):
        return np.array([])

    @property
    def end_points(self):
        return np.array([])

    def _bytes(self):
        data = b''.join([b'Text',
                         self.points.tobytes(),
                         self.text.encode('utf-8')])
        return data


class Line(Entity):
    """
    A line or poly-line entity
    """

    def discrete(self, vertices, scale=1.0):
        """
        Discretize into a world- space path.

        Parameters
        ------------
        vertices: (n, dimension) float
          Points in space
        scale : float
          Size of overall scene for numerical comparisons

        Returns
        -------------
        discrete: (m, dimension) float
          Path in space composed of line segments
        """
        discrete = self._orient(vertices[self.points])
        return discrete

    @property
    def is_valid(self):
        """
        Is the current entity valid.

        Returns
        -----------
        valid : bool
          Is the current entity well formed
        """
        valid = np.any((self.points - self.points[0]) != 0)
        return valid

    def explode(self):
        """
        If the current Line entity consists of multiple line
        break it up into n Line entities.

        Returns
        ----------
        exploded: (n,) Line entities
        """
        points = np.column_stack((
            self.points,
            self.points)).ravel()[1:-1].reshape((-1, 2))
        exploded = [Line(i) for i in points]
        return exploded

    def _bytes(self):
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return b'Line' + self.points.tobytes()
        else:
            return b'Line' + self.points[::-1].tobytes()


class Arc(Entity):

    @property
    def closed(self):
        """
        A boolean flag for whether the arc is closed (a circle) or not.

        Returns
        ----------
        closed : bool
          If set True, Arc will be a closed circle
        """
        if hasattr(self, '_closed'):
            return self._closed
        return False

    @closed.setter
    def closed(self, value):
        """
        Set the Arc to be closed or not, without
        changing the control points

        Parameters
        ------------
        value : bool
          Should this Arc be a closed circle or not
        """
        self._closed = bool(value)

    @property
    def is_valid(self):
        """
        Is the current Arc entity valid.

        Returns
        -----------
        valid : bool
          Does the current Arc have exactly 3 control points
        """
        return len(np.unique(self.points)) == 3

    def _bytes(self):
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return b'Arc' + bytes(self.closed) + self.points.tobytes()
        else:
            return b'Arc' + bytes(self.closed) + self.points[::-1].tobytes()

    def discrete(self, vertices, scale=1.0):
        """
        Discretize the arc entity into line sections.

        Parameters
        ------------
        vertices : (n, dimension) float
            Points in space
        scale : float
            Size of overall scene for numerical comparisons

        Returns
        -------------
        discrete: (m, dimension) float, linear path in space
        """
        discrete = discretize_arc(vertices[self.points],
                                  close=self.closed,
                                  scale=scale)
        return self._orient(discrete)

    def center(self, vertices):
        """
        Return the center information about the arc entity.

        Parameters
        -------------
        vertices : (n, dimension) float
            Vertices in space

        Returns
        -------------
        info: dict, with keys: 'radius'
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
                               info['center'] + info['radius']],
                              dtype=np.float64)
        else:
            # since the AABB of a partial arc is hard, approximate
            # the bounds by just looking at the discrete values
            discrete = self.discrete(vertices)
            bounds = np.array([discrete.min(axis=0),
                               discrete.max(axis=0)],
                              dtype=np.float64)
        return bounds


class Curve(Entity):
    """
    The parent class for all wild curves in space.
    """
    @property
    def nodes(self):
        # a point midway through the curve
        mid = self.points[len(self.points) // 2]
        return [[self.points[0], mid],
                [mid, self.points[-1]]]


class Bezier(Curve):
    """
    An open or closed Bezier curve
    """

    def discrete(self, vertices, scale=1.0, count=None):
        """
        Discretize the Bezier curve.

        Parameters
        -------------
        vertices : (n, 2) or (n, 3) float
          Points in space
        scale : float
          Scale of overall drawings (for precision)
        count : int
          Number of segments to return

        Returns
        -------------
        discrete : (m, 2) or (m, 3) float
          Curve as line segments
        """
        discrete = discretize_bezier(vertices[self.points],
                                     count=count,
                                     scale=scale)
        return self._orient(discrete)


class BSpline(Curve):
    """
    An open or closed B- Spline.
    """

    def __init__(self, points,
                 knots,
                 closed=None,
                 layer=None,
                 **kwargs):
        self.points = np.asanyarray(points, dtype=np.int64)
        self.knots = np.asanyarray(knots, dtype=np.float64)
        self.layer = layer
        self.kwargs = kwargs

    def discrete(self, vertices, count=None, scale=1.0):
        """
        Discretize the B-Spline curve.

        Parameters
        -------------
        vertices : (n, 2) or (n, 3) float
          Points in space
        scale : float
          Scale of overall drawings (for precision)
        count : int
          Number of segments to return

        Returns
        -------------
        discrete : (m, 2) or (m, 3) float
          Curve as line segments
        """
        discrete = discretize_bspline(
            control=vertices[self.points],
            knots=self.knots,
            count=count,
            scale=scale)
        return self._orient(discrete)

    def _bytes(self):
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return (b'BSpline' +
                    self.knots.tobytes() +
                    self.points.tobytes())
        else:
            return (b'BSpline' +
                    self.knots[::-1].tobytes() +
                    self.points[::-1].tobytes())

    def to_dict(self):
        """
        Returns a dictionary with all of the information
        about the entity.
        """
        return {'type': self.__class__.__name__,
                'points': self.points.tolist(),
                'knots': self.knots.tolist(),
                'closed': self.closed}
