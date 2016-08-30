import numpy as np
import pprint

from . import util
from . import points
from . import creation

from .base      import Trimesh
from .constants import log
from .triangles import windings_aligned
   
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
        if util.is_shape(stored, (-1,3)):
            return stored
        self._create_mesh()
        #self._validate_face_normals()
        return self._cache['faces']

    @faces.setter
    def faces(self, values):
        log.warning('Primitive faces are immutable! Not setting!')

    @property
    def vertices(self):
        stored = self._cache['vertices']
        if util.is_shape(stored, (-1,3)):
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
        if util.is_shape(stored, (-1,3)):
            return stored
        self._create_mesh()
        return self._cache['face_normals']
    
    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            log.warning('Primitive face normals are immutable! Not setting!')

    def to_mesh(self):
        '''
        Return a copy of the Primitive object as a trimesh.
        '''
        result = Trimesh(vertices = self.vertices.copy(),
                         faces    = self.faces.copy(),
                         face_normals = self.face_normals.copy())
        return result
            
    def _create_mesh(self):
        raise ValueError('Primitive doesn\'t define mesh creation!')

        
class _PrimitiveAttributes(object):
    def __init__(self, data, defaults, parent, kwargs):
        self._data     = data
        self._defaults = defaults
        self._data.update(defaults)
        
        for key, value in kwargs.items():
            if key in defaults:
                self._data[key] = util.convert_like(value, defaults[key])
    
        self.__doc__ = (('Store the attributes of a {prim} object.\n\n' +
                        'When these values are changed, the mesh geometry will \n' +
                        'automatically be updated to reflect the new values.\n\n' +
                        'Available properties and their default values are:\n {def_s}\n\n' +
                        'Example\n---------------\n' +
                        'p = trimesh.primitives.{prim}()\n' +
                        'p.primitive.radius = 10\n\n').format(prim=parent.__class__.__name__,
                                                              def_s = pprint.pformat(defaults, width = -1)[1:-1]))

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
            self._data[key] = util.convert_like(value,  self._defaults[key])
        else:
            keys = list(self._defaults.keys())
            raise ValueError('Only default attributes {} can be set!'.format(keys))
            
    def __dir__(self):
        result = sorted(dir(type(self)) + 
                        self._defaults.keys())
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
        
        defaults = {'height'    : 1.0,
                    'radius'    : 1.0,
                    'transform' : np.eye(4),
                    'sections'  : 32}
        self.primitive = _PrimitiveAttributes(self._data, defaults, self, kwargs)

    def _create_mesh(self):
        log.info('Creating cylinder mesh with r=%f, h=%f and %d sections',
                 self.primitive.radius,
                 self.primitive.height,
                 self.primitive.sections)
                 
        mesh = creation.cylinder(radius   = self.primitive.radius,
                                 height   = self.primitive.height,
                                 sections = self.primitive.sections)
        mesh.apply_transform(self.primitive.transform)
        
        self._cache['vertices']     = mesh.vertices
        self._cache['faces']        = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
        
        
class Sphere(_Primitive):
    def __init__(self, *args, **kwargs):
        '''
        Create a Sphere _Primitive, which is a subclass of Trimesh.

        Arguments
        ----------
        radius: float, radius of sphere
        center: (3,) float, center of sphere
        subdivisions: int, number of subdivisions for icosphere. Default is 3
        '''

        super(Sphere, self).__init__(*args, **kwargs)
        
        defaults = {'radius'        : 1.0,
                    'center'        : np.zeros(3, dtype=np.float64),
                    'subdivisions'  : 3}
                    
        self.primitive = _PrimitiveAttributes(self._data, defaults, self, kwargs)

    @util.cache_decorator
    def volume(self):
        '''
        Volume of the current sphere _Primitive.

        Returns
        --------
        volume: float, volume of the sphere _Primitive
        '''
        
        volume = (4.0*np.pi * (self.primitive.radius** 3)) / 3.0
        return volume
    
    def _create_mesh(self):
        unit = creation.icosphere(subdivisions=self.primitive.subdivisions)
        unit.vertices *= self.primitive.radius
        unit.vertices += self.primitive.center
        
        self._cache['vertices']     = unit.vertices
        self._cache['faces']        = unit.faces
        self._cache['face_normals'] = unit.face_normals

class Box(_Primitive):    
    def __init__(self, *args, **kwargs):
        '''
        Create a Box _Primitive, which is a subclass of Trimesh

        Arguments
        ----------
        box_extents:   (3,) float, size of box
        box_transform: (4,4) float, transformation matrix for box
        box_center:    (3,) float, convience function which updates box_transform
                       with a translation- only matrix
        '''
        super(Box, self).__init__(*args, **kwargs)
     
        defaults = {'transform' : np.eye(4),
                    'extents'   : np.ones(3)}
        self.primitive = _PrimitiveAttributes(self._data, defaults, self, kwargs)

    @property
    def is_oriented(self):
        if util.is_shape(self.box_transform, (4,4)):
            return not np.allclose(self.box_transform[0:3,0:3], np.eye(3))
        else:
            return False

    @util.cache_decorator
    def volume(self):
        '''
        Volume of the box _Primitive.

        Returns
        --------
        volume: float, volume of box
        '''
        volume = float(np.product(self.primitive.extents))
        return volume
        
    def _create_mesh(self):
        log.debug('Creating mesh for box _Primitive')
        box = creation.box(extents   = self.primitive.extents,
                           transform = self.primitive.transform)
                              
        self._cache['vertices']     = box.vertices
        self._cache['faces']        = box.faces
        self._cache['face_normals'] = box.face_normals

class Extrusion(_Primitive):
    def __init__(self, *args, **kwargs):
        '''
        Create an Extrusion _Primitive, which subclasses Trimesh

        Arguments
        ----------
        extrude_polygon:   shapely.geometry.Polygon, polygon to extrude
        extrude_transform: (4,4) float, transform to apply after extrusion
        extrude_height:    float, height to extrude polygon by
        '''
        super(Extrusion, self).__init__(*args, **kwargs)

        defaults = {'polygon'   : None,
                    'transform' : np.eye(4),
                    'height'    : 1.0}
                    
        self.primitive = _PrimitiveAttributes(self._data, defaults, self, kwargs)

    @property
    def extrude_direction(self):
        direction = np.dot(self.extrude_transform[:3,:3], 
                           [0.0,0.0,1.0])
        return direction
    
    def slide(self, distance):
        distance = float(distance)
        translation = np.eye(4)
        translation[2,3] = distance
        new_transform = np.dot(self.extrude_transform.copy(),
                               translation.copy())
        self.extrude_transform = new_transform

    def _create_mesh(self):
        log.debug('Creating mesh for extrude _Primitive')
        mesh = creation.extrude_polygon(self.primitive.polygon[0],
                                        self.primitive.height)
        mesh.apply_transform(self.primitive.transform)
        self._cache['vertices']     = mesh.vertices
        self._cache['faces']        = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
