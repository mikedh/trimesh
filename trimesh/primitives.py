import numpy as np

from . import util
from . import points
from . import creation

from .base      import Trimesh
from .constants import log
from .triangles import windings_aligned
   
class Primitive(Trimesh):
    '''
    Geometric primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    '''
    def __init__(self, *args, **kwargs):
        super(Primitive, self).__init__(*args, **kwargs)
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

    def _create_mesh(self):
        raise ValueError('Primitive doesn\'t define mesh creation!')

class Sphere(Primitive):
    def __init__(self, *args, **kwargs):
        super(Sphere, self).__init__(*args, **kwargs)
        if 'sphere_radius' in kwargs:
            self.sphere_radius = kwargs['sphere_radius']
        if 'sphere_center' in kwargs:
            self.sphere_center = kwargs['sphere_center']
        if 'subdivisions' in kwargs:
            self._data['subdivisions'] = int(kwargs['subdivisions'])
        else:
            self._data['subdivisions'] = 3
        self._unit_sphere = creation.icosphere(subdivisions=self._data['subdivisions'])

    @property
    def sphere_center(self):
        stored = self._data['center']
        if stored is None:
            return np.zeros(3)
        return stored

    @sphere_center.setter
    def sphere_center(self, values):
        self._data['center'] = np.asanyarray(values, dtype=np.float64)

    @property
    def sphere_radius(self):
        stored = self._data['radius']
        if stored is None:
            return 1.0
        return stored

    @sphere_radius.setter
    def sphere_radius(self, value):
        self._data['radius'] = float(value)

    def _create_mesh(self):
        ico = self._unit_sphere
        self._cache['vertices'] = ((ico.vertices * self.sphere_radius) + 
                                   self.sphere_center)
        self._cache['faces'] = ico.faces
        self._cache['face_normals'] = ico.face_normals

class Box(Primitive):    
    def __init__(self, *args, **kwargs):
        super(Box, self).__init__(*args, **kwargs)
        if 'box_extents' in kwargs:
            self.box_extents = kwargs['box_extents']
        if 'box_transform' in kwargs:
            self.box_transform = kwargs['box_transform']
        if 'box_center' in kwargs:
            self.box_center = kwargs['box_center']
        self._unit_box = creation.box()

    @property
    def box_center(self):
        return self.box_transform[0:3,3]

    @box_center.setter
    def box_center(self, values):
        transform = self.box_transform
        transform[0:3,3] = values
        self._data['box_transform'] = transform

    @property
    def box_extents(self):
        stored = self._data['box_extents']
        if util.is_shape(stored, (3,)):
            return stored
        return np.ones(3)

    @box_extents.setter
    def box_extents(self, values):
        self._data['box_extents'] = np.asanyarray(values, dtype=np.float64)

    @property
    def box_transform(self):
        stored = self._data['box_transform']
        if util.is_shape(stored, (4,4)):
            return stored
        return np.eye(4)

    @box_transform.setter
    def box_transform(self, matrix):
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4,4):
            raise ValueError('Matrix must be (4,4)!')
        self._data['box_transform'] = matrix

    @property
    def is_oriented(self):
        if util.is_shape(self.box_transform, (4,4)):
            return not np.allclose(self.box_transform[0:3,0:3], np.eye(3))
        else:
            return False

    def _create_mesh(self):
        log.debug('Creating mesh for box primitive')
        box = self._unit_box
        vertices, faces, normals = box.vertices, box.faces, box.face_normals
        vertices = points.transform_points(vertices * self.box_extents, 
                                           self.box_transform)
        normals = np.dot(self.box_transform[0:3,0:3], 
                         normals.T).T
        aligned = windings_aligned(vertices[faces[:1]], normals[:1])[0]
        if not aligned:
            faces = np.fliplr(faces)        
        # for a primitive the vertices and faces are derived from other information
        # so it goes in the cache, instead of the datastore
        self._cache['vertices'] = vertices
        self._cache['faces']    = faces
        self._cache['face_normals'] = normals

class Extrusion(Primitive):
    def __init__(self, *args, **kwargs):
        super(Extrusion, self).__init__(*args, **kwargs)

        if 'extrude_polygon' in kwargs:
            self.extrude_polygon   = kwargs['extrude_polygon']
        if 'extrude_transform' in kwargs:
            self.extrude_transform = kwargs['extrude_transform']
        if 'extrude_height' in kwargs:
            self.extrude_height    = kwargs['extrude_height']

    @property
    def extrude_transform(self):
        stored = self._data['extrude_transform']
        if np.shape(stored) == (4,4):
            return stored
        return np.eye(4)

    @extrude_transform.setter
    def extrude_transform(self, matrix):
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4,4):
            raise ValueError('Matrix must be (4,4)!')
        self._data['extrude_transform'] = matrix

    @property
    def extrude_height(self):
        stored = self._data['extrude_height']
        if stored is None: 
            raise ValueError('extrude height not specified!')
        return stored[0]

    @extrude_height.setter
    def extrude_height(self, value):
        height = float(value)
        self._data['extrude_height'] = height

    @property
    def extrude_polygon(self):
        stored = self._data['extrude_polygon']
        if stored is None: 
            raise ValueError('extrude polygon not specified!')
        return stored[0]

    @extrude_polygon.setter
    def extrude_polygon(self, value):
        polygon = creation.validate_polygon(value)
        self._data['extrude_polygon'] = polygon

    def _create_mesh(self):
        log.debug('Creating mesh for extrude primitive')
        mesh = creation.extrude_polygon(self.extrude_polygon,
                                        self.extrude_height)
        mesh.transform(self.extrude_transform)
        self._cache['vertices']     = mesh.vertices
        self._cache['faces']        = mesh.faces
        self._cache['face_normals'] = mesh.face_normals
