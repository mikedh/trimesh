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
from . import transformations

from .base import Trimesh
from .constants import log


class _Primitive(Trimesh):
    """
    Geometric _Primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    """

    def __init__(self, *args, **kwargs):
        super(_Primitive, self).__init__(*args, **kwargs)
        self._data.clear()
        self._validate = False

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
        stored = self._cache['face_normals']
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        return self._cache['face_normals']

    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            log.warning('Primitive face normals are immutable! Not setting!')

    def copy(self):
        """
        Return a copy of the Primitive object.
        """
        result = copy.deepcopy(self)
        result._cache.clear()
        return result

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
        Apply a transform to the current primitive (sets self.transform)

        Parameters
        -----------
        matrix: (4,4) float, homogenous transformation
        """
        matrix = np.asanyarray(matrix, order='C', dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Transformation matrix must be (4,4)!')

        if np.allclose(matrix, np.eye(4)):
            log.debug('apply_tranform received identity matrix')
            return

        new_transform = np.dot(matrix, self.primitive.transform)

        self.primitive.transform = new_transform

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
                self._data[key] = util.convert_like(value, defaults[key])
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
        if '_' in key:
            return super(_PrimitiveAttributes, self).__getattr__(key)
        elif key in self._defaults:
            return util.convert_like(self._data[key], self._defaults[key])
        return super(_PrimitiveAttributes, self).__getattr__(key)

    def __setattr__(self, key, value):
        if '_' in key:
            return super(_PrimitiveAttributes, self).__setattr__(key, value)
        elif key in self._defaults:
            if self._mutable:
                self._data[key] = util.convert_like(value,
                                                    self._defaults[key])
            else:
                raise ValueError(
                    'Primitive is configured as immutable! Cannot set attribute!')
        else:
            keys = list(self._defaults.keys())
            raise ValueError(
                'Only default attributes {} can be set!'.format(keys))

    def __dir__(self):
        result = sorted(dir(type(self)) +
                        list(self._defaults.keys()))
        return result


class Cylinder(_Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create a Cylinder Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius: float, radius of cylinder
        height: float, height of cylinder
        transform: (4,4) float, transformation matrix
        sections: int, number of facets in circle
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
        volume: float, volume of the cylinder
        """
        volume = (np.pi * self.primitive.radius ** 2) * self.primitive.height
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the cylinder primitive.

        Returns
        ----------
        tensor: (3,3) float, 3D inertia tensor
        """

        tensor = inertia.cylinder_inertia(mass=self.volume,
                                          radius=self.primitive.radius,
                                          height=self.primitive.height,
                                          transform=self.primitive.transform)
        return tensor

    @property
    def direction(self):
        """
        The direction of the cylinder's axis.

        Returns
        --------
        axis: (3,) float, vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    def _create_mesh(self):
        log.info('Creating cylinder mesh with r=%f, h=%f and %d sections',
                 self.primitive.radius,
                 self.primitive.height,
                 self.primitive.sections)

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

    @property
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
        log.info('Creating capsule mesh with r=%f, h=%f and %d sections',
                 self.primitive.radius,
                 self.primitive.height,
                 self.primitive.sections)

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
        matrix: (4,4) float, homogenous transformation
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
        count: int, number of samples to return

        Returns
        ----------
        samples: (count,3) float, points inside the volume
        """
        samples = sample.volume_rectangular(extents=self.primitive.extents,
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
        count: int   or (3,) int,   if specified samples are spaced with np.linspace
        step:  float or (3,) float, if specified samples are spaced with np.arange

        Returns
        -----------
        grid: (n,3) float, points inside the box
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

        transformed = transformations.transform_points(
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
        log.debug('Creating mesh for box Primitive')
        box = creation.box(extents=self.primitive.extents,
                           transform=self.primitive.transform)

        self._cache['vertices'] = box.vertices
        self._cache['faces'] = box.faces
        self._cache['face_normals'] = box.face_normals


class Extrusion(_Primitive):

    def __init__(self, *args, **kwargs):
        """
        Create an Extrusion Primitive, a subclass of Trimesh

        Parameters
        ----------
        polygon:   shapely.geometry.Polygon, polygon to extrude
        transform: (4,4) float, transform to apply after extrusion
        height:    float, height to extrude polygon by
        """
        super(Extrusion, self).__init__(*args, **kwargs)

        # do the import here, fail early if Shapely isn't installed
        from shapely.geometry import Point

        defaults = {'polygon': Point([0, 0]).buffer(1.0),
                    'transform': np.eye(4),
                    'height': 1.0}

        self.primitive = _PrimitiveAttributes(self,
                                              defaults,
                                              kwargs)

    @caching.cache_decorator
    def area(self):
        """
        The surface area of the primitive extrusion.

        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        area: float, surface area of 3D extrusion
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
        The volume of the primitive extrusion.

        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        volume: float, volume of 3D extrusion
        """
        volume = abs(self.primitive.polygon.area *
                     self.primitive.height)
        return volume

    @property
    def direction(self):
        """
        Based on the extrudes transform, what is the vector along
        which the polygon will be extruded

        Returns
        ---------
        direction: (3,) float vector. If self.primitive.transform is an
                   identity matrix this will be [0.0, 0.0, 1.0]
        """
        direction = np.dot(self.primitive.transform[:3, :3],
                           [0.0, 0.0, np.sign(self.primitive.height)])
        return direction

    def slide(self, distance):
        """
        Alter the transform of the current extrusion to slide it
        along its extrude_direction vector

        Parameters
        -----------
        distance: float, distance along self.extrude_direction to move
        """
        distance = float(distance)
        translation = np.eye(4)
        translation[2, 3] = distance
        new_transform = np.dot(self.primitive.transform.copy(),
                               translation.copy())
        self.primitive.transform = new_transform

    def buffer(self, distance):
        """
        Return a new Extrusion object which is expanded in profile and
        in height by a specified distance.

        Returns
        ----------
        buffered: Extrusion object
        """
        distance = float(distance)

        # start with current height
        height = self.primitive.height
        # if current height is negative offset by negative amount
        height += np.sign(height) * 2.0 * distance

        buffered = Extrusion(
            transform=self.primitive.transform.copy(),
            polygon=self.primitive.polygon.buffer(distance),
            height=height)

        # slide the stock along the axis
        buffered.slide(-np.sign(height) * distance)

        return buffered

    def _create_mesh(self):
        log.debug('Creating mesh for extrude Primitive')
        mesh = creation.extrude_polygon(self.primitive.polygon,
                                        self.primitive.height)
        mesh.apply_transform(self.primitive.transform)

        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
