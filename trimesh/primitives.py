"""
primitives.py
----------------

Subclasses of Trimesh objects that are parameterized as primitives.

Useful because you can move boxes and spheres around, and then use
trimesh operations on them at any point.
"""
import numpy as np
import pprint
import copy

from . import util
from . import sample
from . import caching
from . import inertia
from . import creation
from . import triangles
from . import transformations as tf

from .base import Trimesh
from .constants import log, tol


class _Primitive(Trimesh):
    """
    Geometric _Primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    """

    # ignore superclass copy directives
    __copy__ = None
    __deepcopy__ = None

    def __init__(self, *args, **kwargs):
        super(_Primitive, self).__init__(*args, **kwargs)
        self._data.clear()
        self._validate = False

    def __repr__(self):
        return '<trimesh.primitives.{}>'.format(type(self).__name__)

    @property
    def faces(self):
        stored = self._cache['faces']
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        return self._cache['faces']

    @faces.setter
    def faces(self, values):
        log.warning('Primitive faces are immutable! Not setting!')

    @property
    def vertices(self):
        stored = self._cache['vertices']
        if util.is_shape(stored, (-1, 3)):
            return stored

        self._create_mesh()
        return self._cache['vertices']

    @vertices.setter
    def vertices(self, values):
        if values is not None:
            log.warning('Primitive vertices are immutable! Not setting!')

    @property
    def face_normals(self):
        # we need to avoid the logic in the superclass that
        # is specific to the data model prioritizing faces
        stored = self._cache['face_normals']
        if util.is_shape(stored, (-1, 3)):
            return stored
        # just calculate if not stored
        unit, valid = triangles.normals(self.triangles)
        normals = np.zeros((len(valid), 3))
        normals[valid] = unit
        # store and return
        self._cache['face_normals'] = normals
        return normals

    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            log.warning('Primitive face normals are immutable! Not setting!')

    def copy(self, **kwargs):
        """
        Return a copy of the Primitive object.

        Returns
        -------------
        copied : object
          Copy of current primitive
        """
        return copy.deepcopy(self)

    def to_mesh(self):
        """
        Return a copy of the Primitive object as a Trimesh object.
        """
        result = Trimesh(vertices=self.vertices.copy(),
                         faces=self.faces.copy(),
                         face_normals=self.face_normals.copy(),
                         process=False)
        return result

    def apply_transform(self, matrix):
        """
        Apply a transform to the current primitive by
        setting self.transform

        Parameters
        ------------
        matrix: (4,4) float
          Homogeneous transformation
        """
        matrix = np.asanyarray(matrix, order='C', dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Transformation matrix must be (4,4)!')

        if util.allclose(matrix, np.eye(4), 1e-8):
            log.debug('apply_transform received identity matrix')
            return

        new_transform = np.dot(matrix, self.primitive.transform)
        self.primitive.transform = new_transform
        return self

    def _create_mesh(self):
        raise ValueError('Primitive doesn\'t define mesh creation!')


class _PrimitiveAttributes(object):
    """
    Hold the mutable data which defines a primitive.
    """

    def __init__(self, parent, defaults, kwargs):
        self._data = parent._data
        self._defaults = defaults
        self._parent = parent
        self._data.update(defaults)
        self._mutable = True
        for key, value in kwargs.items():
            if key in defaults:
                self._data[key] = util.convert_like(
                    value, defaults[key])
        # if configured as immutable, apply setting after instantiation values
        # are set
        if 'mutable' in kwargs:
            self._mutable = bool(kwargs['mutable'])

    @property
    def __doc__(self):
        # this is generated dynamically as the format operation can be surprisingly
        # slow and if generated in __init__ it is called a lot of times
        # when we didn't really need to generate it

        doc = (
            'Store the attributes of a {name} object.\n\n' +
            'When these values are changed, the mesh geometry will \n' +
            'automatically be updated to reflect the new values.\n\n' +
            'Available properties and their default values are:\n {defaults}' +
            '\n\nExample\n---------------\n' +
            'p = trimesh.primitives.{name}()\n' +
            'p.primitive.radius = 10\n' +
            '\n').format(
            name=self._parent.__class__.__name__,
            defaults=pprint.pformat(
                self._defaults,
                width=-1)[1:-1])
        return doc

    def __getattr__(self, key):
        if key.startswith('_'):
            return super(_PrimitiveAttributes, self).__getattr__(key)
        elif key in self._defaults:
            return util.convert_like(self._data[key], self._defaults[key])
        raise AttributeError(
            "primitive object has no attribute '{}' ".format(key))

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super(_PrimitiveAttributes, self).__setattr__(key, value)
        elif key in self._defaults:
            if self._mutable:
                self._data[key] = util.convert_like(
                    value, self._defaults[key])
            else:
                raise ValueError(
                    'Primitive is configured as immutable! Cannot set attribute!')
        else:
            keys = list(self._defaults.keys())
            raise ValueError(
                'Only default attributes {} can be set!'.format(keys))

    def to_kwargs(self):
        """
        Return a dict with copies of kwargs for the current
        Primitive.

        Returns
        ------------
        kwargs : dict
          Arguments to reconstruct current PrimitiveAttributes
        """
        return {k: copy.deepcopy(self._data[k])
                for k in self._defaults.keys()}

    def __dir__(self):
        result = sorted(dir(type(self)) +
                        list(self._defaults.keys()))
        return result


class Cylinder(_Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Cylinder Primitive, a subclass of Trimesh.

        Parameters
        -------------
        radius : float
          Radius of cylinder
        height : float
          Height of cylinder
        transform : (4, 4) float
          Homogeneous transformation matrix
        sections : int
          Number of facets in circle
        """
        super(Cylinder, self).__init__(*args, **kwargs)

        defaults = {'height': 10.0,
                    'radius': 1.0,
                    'transform': np.eye(4),
                    'sections': 32}
        self.primitive = _PrimitiveAttributes(self,
                                              defaults,
                                              kwargs)

    @caching.cache_decorator
    def volume(self):
        """
        The analytic volume of the cylinder primitive.

        Returns
        ---------
        volume : float
          Volume of the cylinder
        """
        volume = ((np.pi * self.primitive.radius ** 2) *
                  self.primitive.height)
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the cylinder primitive.

        Returns
        ----------
        tensor: (3,3) float, 3D inertia tensor
        """

        tensor = inertia.cylinder_inertia(
            mass=self.volume,
            radius=self.primitive.radius,
            height=self.primitive.height,
            transform=self.primitive.transform)
        return tensor

    @caching.cache_decorator
    def direction(self):
        """
        The direction of the cylinder's axis.

        Returns
        --------
        axis: (3,) float, vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    @property
    def segment(self):
        """
        A line segment which if inflated by cylinder radius
        would represent the cylinder primitive.

        Returns
        -------------
        segment : (2, 3) float
          Points representing a single line segment
        """
        # half the height
        half = self.primitive.height / 2.0
        # apply the transform to the Z- aligned segment
        points = np.dot(
            self.primitive.transform,
            np.transpose([[0, 0, -half, 1], [0, 0, half, 1]])).T[:, :3]
        return points

    def buffer(self, distance):
        """
        Return a cylinder primitive which covers the source cylinder
        by distance: radius is inflated by distance, height by twice
        the distance.

        Parameters
        ------------
        distance : float
          Distance to inflate cylinder radius and height

        Returns
        -------------
        buffered : Cylinder
         Cylinder primitive inflated by distance
        """
        distance = float(distance)

        buffered = Cylinder(
            height=self.primitive.height + distance * 2,
            radius=self.primitive.radius + distance,
            transform=self.primitive.transform.copy())
        return buffered

    def _create_mesh(self):
        log.debug('creating mesh for Cylinder primitive')
        mesh = creation.cylinder(radius=self.primitive.radius,
                                 height=self.primitive.height,
                                 sections=self.primitive.sections,
                                 transform=self.primitive.transform)

        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Capsule(_Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Capsule Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius: float, radius of cylinder
        height: float, height of cylinder
        transform: (4,4) float, transformation matrix
        sections: int, number of facets in circle
        """
        super(Capsule, self).__init__(*args, **kwargs)

        defaults = {'height': 1.0,
                    'radius': 1.0,
                    'transform': np.eye(4),
                    'sections': 32}
        self.primitive = _PrimitiveAttributes(self,
                                              defaults,
                                              kwargs)

    @caching.cache_decorator
    def direction(self):
        """
        The direction of the capsule's axis.

        Returns
        --------
        axis: (3,) float, vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    def _create_mesh(self):
        log.debug('creating mesh for Capsule primitive')

        mesh = creation.capsule(radius=self.primitive.radius,
                                height=self.primitive.height)
        mesh.apply_transform(self.primitive.transform)

        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Sphere(_Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Sphere Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius: float, radius of sphere
        center: (3,) float, center of sphere
        subdivisions: int, number of subdivisions for icosphere. Default is 3
        """

        super(Sphere, self).__init__(*args, **kwargs)

        defaults = {'radius': 1.0,
                    'center': np.zeros(3, dtype=np.float64),
                    'subdivisions': 3}

        self.primitive = _PrimitiveAttributes(self,
                                              defaults,
                                              kwargs)

    def apply_transform(self, matrix):
        """
        Apply a transform to the sphere primitive

        Parameters
        ------------
        matrix: (4,4) float, homogeneous transformation
        """
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('shape must be 4,4')

        center = np.dot(matrix,
                        np.append(self.primitive.center, 1.0))[:3]
        self.primitive.center = center

    @property
    def bounds(self):
        # no docstring so will inherit Trimesh docstring
        # return exact bounds from primitive center and radius (rather than faces)
        # self.extents will also use this information
        bounds = np.array([self.primitive.center - self.primitive.radius,
                           self.primitive.center + self.primitive.radius])
        return bounds

    @property
    def bounding_box_oriented(self):
        # for a sphere the oriented bounding box is the same as the axis aligned
        # bounding box, and a sphere is the absolute slowest case for the OBB calculation
        # as it is a convex surface with a ton of face normals that all need to
        # be checked
        return self.bounding_box

    @caching.cache_decorator
    def area(self):
        """
        Surface area of the current sphere primitive.

        Returns
        --------
        area: float, surface area of the sphere Primitive
        """

        area = 4.0 * np.pi * (self.primitive.radius ** 2)
        return area

    @caching.cache_decorator
    def volume(self):
        """
        Volume of the current sphere primitive.

        Returns
        --------
        volume: float, volume of the sphere Primitive
        """

        volume = (4.0 * np.pi * (self.primitive.radius ** 3)) / 3.0
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the sphere primitive.

        Returns
        ----------
        tensor: (3,3) float, 3D inertia tensor
        """
        tensor = inertia.sphere_inertia(mass=self.volume,
                                        radius=self.primitive.radius)
        return tensor

    def _create_mesh(self):
        log.debug('creating mesh for Sphere primitive')
        unit = creation.icosphere(subdivisions=self.primitive.subdivisions)
        unit.vertices *= self.primitive.radius
        unit.vertices += self.primitive.center

        self._cache['vertices'] = unit.vertices
        self._cache['faces'] = unit.faces
        self._cache['face_normals'] = unit.face_normals


class Box(_Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Box Primitive, a subclass of Trimesh

        Parameters
        ----------
        extents:   (3,)  float, size of box
        transform: (4,4) float, transformation matrix for box center
        """
        super(Box, self).__init__(*args, **kwargs)

        defaults = {'transform': np.eye(4),
                    'extents': np.ones(3)}
        self.primitive = _PrimitiveAttributes(self,
                                              defaults,
                                              kwargs)

    def sample_volume(self, count):
        """
        Return random samples from inside the volume of the box.

        Parameters
        -------------
        count : int
          Number of samples to return

        Returns
        ----------
        samples : (count, 3) float
          Points inside the volume
        """
        samples = sample.volume_rectangular(
            extents=self.primitive.extents,
            count=count,
            transform=self.primitive.transform)
        return samples

    def sample_grid(self, count=None, step=None):
        """
        Return a 3D grid which is contained by the box.
        Samples are either 'step' distance apart, or there are
        'count' samples per box side.

        Parameters
        -----------
        count : int or (3,) int
          If specified samples are spaced with np.linspace
        step : float or (3,) float
          If specified samples are spaced with np.arange

        Returns
        -----------
        grid : (n, 3) float
          Points inside the box
        """

        if (count is not None and
                step is not None):
            raise ValueError('only step OR count can be specified!')

        # create pre- transform bounds from extents
        bounds = np.array([-self.primitive.extents,
                           self.primitive.extents]) * .5

        if step is not None:
            grid = util.grid_arange(bounds, step=step)
        elif count is not None:
            grid = util.grid_linspace(bounds, count=count)
        else:
            raise ValueError('either count or step must be specified!')

        transformed = tf.transform_points(
            grid, matrix=self.primitive.transform)
        return transformed

    @property
    def is_oriented(self):
        """
        Returns whether or not the current box is rotated at all.
        """
        if util.is_shape(self.primitive.transform, (4, 4)):
            return not np.allclose(self.primitive.transform[
                                   0:3, 0:3], np.eye(3))
        else:
            return False

    @caching.cache_decorator
    def volume(self):
        """
        Volume of the box Primitive.

        Returns
        --------
        volume: float, volume of box
        """
        volume = float(np.product(self.primitive.extents))
        return volume

    def _create_mesh(self):
        log.debug('creating mesh for Box primitive')
        box = creation.box(extents=self.primitive.extents,
                           transform=self.primitive.transform)

        self._cache['vertices'] = box.vertices
        self._cache['faces'] = box.faces
        self._cache['face_normals'] = box.face_normals

    def as_outline(self):
        """
        Return a Path3D containing the outline of the box.

        Returns
        -----------
        outline : trimesh.path.Path3D
          Outline of box primitive
        """
        # do the import in function to keep soft dependency
        from .path.creation import box_outline
        # return outline with same size as primitive
        return box_outline(
            extents=self.primitive.extents,
            transform=self.primitive.transform)


class Extrusion(_Primitive):

    def __init__(self, triangle_args=None, *args, **kwargs):
        """
        Create an Extrusion primitive, which
        is a subclass of Trimesh.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
          Polygon to extrude
        transform : (4,4) float
          Transform to apply after extrusion
        height : float
          Height to extrude polygon by
        triangle_args : str
          Arguments to pass to triangle
        """
        # do the import here, fail early if Shapely isn't installed
        from shapely.geometry import Point
        super(Extrusion, self).__init__(*args, **kwargs)
        # save arguments for triangulation
        self.triangle_args = triangle_args
        # set default values
        defaults = {'polygon': Point([0, 0]).buffer(1.0),
                    'transform': np.eye(4),
                    'height': 1.0}
        self.primitive = _PrimitiveAttributes(
            self, defaults, kwargs)

    @caching.cache_decorator
    def area(self):
        """
        The surface area of the primitive extrusion.

        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        area: float
          Surface area of 3D extrusion
        """
        # area of the sides of the extrusion
        area = abs(self.primitive.height *
                   self.primitive.polygon.length)
        # area of the two caps of the extrusion
        area += self.primitive.polygon.area * 2
        return area

    @caching.cache_decorator
    def volume(self):
        """
        The volume of the Extrusion primitive.
        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        volume : float
          Volume of 3D extrusion
        """
        # height may be negative
        volume = abs(self.primitive.polygon.area *
                     self.primitive.height)
        return volume

    @caching.cache_decorator
    def direction(self):
        """
        Based on the extrudes transform what is the
        vector along which the polygon will be extruded.

        Returns
        ---------
        direction : (3,) float
          Unit direction vector
        """
        # only consider rotation and signed height
        direction = np.dot(
            self.primitive.transform[:3, :3],
            [0.0, 0.0, np.sign(self.primitive.height)])
        return direction

    @property
    def origin(self):
        """
        Based on the extrude transform what is the
        origin of the plane it is extruded from.

        Returns
        -----------
        origin : (3,) float
          Origin of extrusion plane
        """
        return self.primitive.transform[:3, 3]

    @caching.cache_decorator
    def bounding_box_oriented(self):
        # no docstring for inheritance
        # calculate OBB using 2D polygon and known axis
        from . import bounds
        # find the 2D bounding box using the polygon
        to_origin, box = bounds.oriented_bounds_2D(
            self.primitive.polygon.exterior.coords)
        #  3D extents
        extents = np.append(box, abs(self.primitive.height))
        # calculate to_3D transform from 2D obb
        rotation_Z = np.linalg.inv(tf.planar_matrix_to_3D(to_origin))
        rotation_Z[2, 3] = self.primitive.height / 2.0
        # combine the 2D OBB transformation with the 2D projection transform
        to_3D = np.dot(self.primitive.transform, rotation_Z)
        obb = Box(transform=to_3D,
                  extents=extents,
                  mutable=False)
        return obb

    def slide(self, distance):
        """
        Alter the transform of the current extrusion to slide it
        along its extrude_direction vector

        Parameters
        -----------
        distance : float
          Distance along self.extrude_direction to move
        """
        distance = float(distance)
        translation = np.eye(4)
        translation[2, 3] = distance
        new_transform = np.dot(self.primitive.transform.copy(),
                               translation.copy())
        self.primitive.transform = new_transform

    def buffer(self, distance, distance_height=None, **kwargs):
        """
        Return a new Extrusion object which is expanded in profile
        and in height by a specified distance.

        Parameters
        --------------
        distance : float
          Distance to buffer polygon
        distance_height : float
          Distance to buffer above and below extrusion
        kwargs : dict
          Passed to Extrusion constructor

        Returns
        ----------
        buffered : primitives.Extrusion
          Extrusion object with new values
        """
        distance = float(distance)
        # if not specified use same distance for everything
        if distance_height is None:
            distance_height = distance

        # start with current height
        height = self.primitive.height
        # if current height is negative offset by negative amount
        height += np.sign(height) * 2.0 * distance_height

        # create a new extrusion with a buffered polygon
        # use type(self) vs Extrusion to handle subclasses
        buffered = type(self)(
            transform=self.primitive.transform.copy(),
            polygon=self.primitive.polygon.buffer(distance),
            height=height,
            **kwargs)

        # slide the stock along the axis
        buffered.slide(-np.sign(height) * distance_height)

        return buffered

    def to_dict(self):
        """
        """
        kwargs = {'polygon': self.primitive.polygon.wkt,
                  'transform': self.primitive.transform.tolist(),
                  'height': float(self.primitive.height)}
        return kwargs

    def _create_mesh(self):
        log.debug('creating mesh for Extrusion primitive')
        # extrude the polygon along Z
        mesh = creation.extrude_polygon(
            polygon=self.primitive.polygon,
            height=self.primitive.height,
            transform=self.primitive.transform,
            triangle_args=self.triangle_args)

        # check volume here in unit tests
        if tol.strict and mesh.volume < 0.0:
            raise ValueError('matrix inverted mesh!')

        # cache mesh geometry in the primitive
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
