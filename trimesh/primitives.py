import numpy as np
import pprint

from . import util
from . import creation
from . import sample

from .base import Trimesh
from .constants import log


class _Primitive(Trimesh):
    '''
    Geometric _Primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    '''

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
        '''
        Return a copy of the Primitive object as a trimesh.
        '''
        result = Trimesh(vertices=self.vertices.copy(),
                         faces=self.faces.copy(),
                         face_normals=self.face_normals.copy())
        return result

    def _create_mesh(self):
        raise ValueError('Primitive doesn\'t define mesh creation!')


class _PrimitiveAttributes(object):

    def __init__(self, data, defaults, parent, kwargs):
        self._data = data
        self._defaults = defaults
        self._data.update(defaults)
        self._mutable = True
        for key, value in kwargs.items():
            if key in defaults:
                self._data[key] = util.convert_like(value, defaults[key])
        # if configured as immutable, apply setting afer instantiation values
        # are set
        if 'mutable' in kwargs:
            self._mutable = bool(kwargs['mutable'])

        self.__doc__ = (('Store the attributes of a {prim_name} object.\n\n' +
                         'When these values are changed, the mesh geometry will \n' +
                         'automatically be updated to reflect the new values.\n\n' +
                         'Available properties and their default values are:\n {prim_defaults}\n\n' +
                         'Example\n---------------\n' +
                         'p = trimesh.primitives.{prim_name}()\n' +
                         'p.primitive.radius = 10\n\n').format(prim_name=parent.__class__.__name__,
                                                               prim_defaults=pprint.pformat(defaults, width=-1)[1:-1]))

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
                self._data[key] = util.convert_like(
                    value, self._defaults[key])
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
        '''
        Create a Cylinder Primitive, a subclass of Trimesh.

        Arguments
        ----------
        radius: float, radius of cylinder
        height: float, height of cylinder
        transform: (4,4) float, transformation matrix
        sections: int, number of facets in circle
        '''
        super(Cylinder, self).__init__(*args, **kwargs)

        defaults = {'height': 1.0,
                    'radius': 1.0,
                    'transform': np.eye(4),
                    'sections': 32}
        self.primitive = _PrimitiveAttributes(
            self._data, defaults, self, kwargs)

    def _create_mesh(self):
        log.info('Creating cylinder mesh with r=%f, h=%f and %d sections',
                 self.primitive.radius,
                 self.primitive.height,
                 self.primitive.sections)

        mesh = creation.cylinder(radius=self.primitive.radius,
                                 height=self.primitive.height,
                                 sections=self.primitive.sections)
        mesh.apply_transform(self.primitive.transform)

        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals


class Sphere(_Primitive):

    def __init__(self, *args, **kwargs):
        '''
        Create a Sphere Primitive, a subclass of Trimesh.

        Arguments
        ----------
        radius: float, radius of sphere
        center: (3,) float, center of sphere
        subdivisions: int, number of subdivisions for icosphere. Default is 3
        '''

        super(Sphere, self).__init__(*args, **kwargs)

        defaults = {'radius': 1.0,
                    'center': np.zeros(3, dtype=np.float64),
                    'subdivisions': 3}

        self.primitive = _PrimitiveAttributes(
            self._data, defaults, self, kwargs)

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

    @util.cache_decorator
    def volume(self):
        '''
        Volume of the current sphere Primitive.

        Returns
        --------
        volume: float, volume of the sphere Primitive
        '''

        volume = (4.0 * np.pi * (self.primitive.radius ** 3)) / 3.0
        return volume

    def _create_mesh(self):
        unit = creation.icosphere(subdivisions=self.primitive.subdivisions)
        unit.vertices *= self.primitive.radius
        unit.vertices += self.primitive.center

        self._cache['vertices'] = unit.vertices
        self._cache['faces'] = unit.faces
        self._cache['face_normals'] = unit.face_normals


class Box(_Primitive):

    def __init__(self, *args, **kwargs):
        '''
        Create a Box Primitive, a subclass of Trimesh

        Arguments
        ----------
        extents:   (3,)  float, size of box
        transform: (4,4) float, transformation matrix for box center
        '''
        super(Box, self).__init__(*args, **kwargs)

        defaults = {'transform': np.eye(4),
                    'extents': np.ones(3)}
        self.primitive = _PrimitiveAttributes(self._data, 
                                              defaults, 
                                              self, kwargs)

    def sample_volume(self, count):
        '''
        Return samples from inside the volume of the box.

        Arguments
        -------------
        count: int, number of samples to return

        Returns
        ----------
        samples: (count,3) float, points inside the volume
        '''
        samples = sample.volume_rectangular(extents=self.primitive.extents,
                                            count=count,
                                            transform=self.primitive.transform)
        return samples

    @property
    def is_oriented(self):
        '''
        Returns whether or not the current box is rotated at all.
        '''
        if util.is_shape(self.primitive.transform, (4, 4)):
            return not np.allclose(self.primitive.transform[
                                   0:3, 0:3], np.eye(3))
        else:
            return False

    @util.cache_decorator
    def volume(self):
        '''
        Volume of the box Primitive.

        Returns
        --------
        volume: float, volume of box
        '''
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
        '''
        Create an Extrusion Primitive, a subclass of Trimesh

        Arguments
        ----------
        polygon:   shapely.geometry.Polygon, polygon to extrude
        transform: (4,4) float, transform to apply after extrusion
        height:    float, height to extrude polygon by
        '''
        super(Extrusion, self).__init__(*args, **kwargs)

        defaults = {'polygon': None,
                    'transform': np.eye(4),
                    'height': 1.0}

        self.primitive = _PrimitiveAttributes(
            self._data, defaults, self, kwargs)

    @property
    def extrude_direction(self):
        '''
        Based on the extrudes transform, what is the vector along
        which the polygon will be extruded

        Returns
        ---------
        direction: (3,) float vector. If self.primitive.transform is an
                   identity matrix this will be [0.0, 0.0, 1.0]
        '''
        direction = np.dot(self.primitive.transform[:3, :3],
                           [0.0, 0.0, 1.0])
        return direction

    def slide(self, distance):
        '''
        Alter the transform of the current extrusion to slide it along its
        extrude_direction vector

        Arguments
        -----------
        distance: float, distance along self.extrude_direction to move
        '''
        distance = float(distance)
        translation = np.eye(4)
        translation[2, 3] = distance
        new_transform = np.dot(self.primitive.transform.copy(),
                               translation.copy())
        self.primitive.transform = new_transform

    def _create_mesh(self):
        log.debug('Creating mesh for extrude Primitive')
        mesh = creation.extrude_polygon(self.primitive.polygon[0],
                                        self.primitive.height)
        mesh.apply_transform(self.primitive.transform)
        self._cache['vertices'] = mesh.vertices
        self._cache['faces'] = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
