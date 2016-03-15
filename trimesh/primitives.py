import numpy as np

from . import util
from . import points

from .base      import Trimesh
from .constants import log

class Primitive(Trimesh):
    '''
    Geometric primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    '''
    def __init__(self, *args, **kwargs):
        super(Primitive, self).__init__(*args, **kwargs)
        self._data.clear()

    @property
    def faces(self):
        stored = self._cache['faces']
        if util.is_shape(stored, (-1,3)):
            return stored
        self._create_mesh()
        self._validate_face_normals()
        return self._cache['faces']

    @faces.setter
    def faces(self, values):
        log.warning('Primitive faces are immutable! Not setting!')

    def _validate_face_normals(self, *args, **kwargs):
        stored = self._cache['faces']
        if not util.is_shape(stored, (-1,3)):
            self._create_mesh()
        super(Primitive, self)._validate_face_normals(self._cache['faces'])

    @property
    def vertices(self):
        stored = self._cache['vertices']
        if util.is_shape(stored, (-1,3)):
            return stored

        self._create_mesh()
        return self._cache['vertices']

    @vertices.setter
    def vertices(self, values):
        log.warning('Primitive vertices are immutable! Not setting!')
    
    def _create_mesh(self):
        raise ValueError('Primitive doesn\'t define mesh creation!')

class Box(Primitive):    
    def __init__(self, *args, **kwargs):
        if 'box_corners' in kwargs:
            self.box_corners = kwargs['box_corners']
        if 'box_transform' in kwargs:
            self.box_transform = kwargs['box_transform']
        super(Box, self).__init__(*args, **kwargs)
    
    @property
    def box_corners(self):
        return self._data['box_corners']

    @box_corners.setter
    def box_corners(self, values):
        self._data['box_corners'] = np.asanyarray(values, dtype=np.float64)

    @property
    def box_transform(self):
        return self._data['box_transform']

    @box_transform.setter
    def box_transform(self, matrix):
        self._data['box_transform'] = np.asanyarray(matrix, dtype=np.float64)

    @property
    def is_oriented(self):
        if util.is_shape(self.box_transform, (4,4)):
            return not np.allclose(self.box_transform, np.eye(4))
        else:
            return False

    def _create_mesh(self):
        log.debug('Creating mesh for box primitive')

        vertices = [0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1]        
        vertices = np.array(vertices, dtype=np.float64).reshape((-1,3))
        vertices *= np.abs(np.diff(self.box_corners, axis=0)).reshape(3)
        vertices += self.box_corners[0]

        faces = [1,3,0,4,1,0,0,3,2,2,4,0,1,7,3,5,1,4,
                 5,7,1,3,7,2,6,4,2,2,7,6,6,5,4,7,5,6] 
        faces = np.array(faces, dtype=np.int64).reshape((-1,3))

        if self.box_transform is not None:
            vertices = points.transform_points(vertices, self.box_transforms)

        # for a primitive the vertices and faces are derived from other information
        # so it goes in the cache, instead of the datastore
        self._cache['vertices'] = vertices
        self._cache['faces'] = faces
