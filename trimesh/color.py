import numpy as np
from colorsys import hsv_to_rgb
from .constants import TOL_ZERO

COLORS = {'red'    : [194,59,34],
          'purple' : [150,111,214],
          'blue'   : [119,158,203],
          'brown'  : [160,85,45]}
COLOR_DTYPE = np.uint8
DEFAULT_COLOR = COLORS['purple']

def random_color(dtype=COLOR_DTYPE):
    '''
    Return a random RGB color using datatype specified.
    '''
    golden_ratio_conjugate = 0.618033988749895
    h     = np.mod(np.random.random() + golden_ratio_conjugate, 1)
    color = np.array(hsv_to_rgb(h, 0.5, 0.95))
    if np.dtype(dtype).kind in 'iu':
        max_value = (2**(np.dtype(dtype).itemsize * 8)) - 1
        color    *= max_value
    color     = color.astype(dtype)
    return color

def face_to_vertex_color(mesh, face_colors, dtype=COLOR_DTYPE):
    '''
    Convert a set of face colors into a set of vertex colors.
    '''
    vertex_colors = np.zeros((len(mesh.vertices), 3,3))
    vertex_colors[[mesh.faces[:,0],0]] = face_colors
    vertex_colors[[mesh.faces[:,1],1]] = face_colors
    vertex_colors[[mesh.faces[:,2],2]] = face_colors    
    populated = np.all(vertex_colors > TOL_ZERO, axis=2).sum(axis=1)
    vertex_colors = np.sum(vertex_colors, axis=1) / populated.reshape((-1,1))
    return vertex_colors.astype(dtype)

class VisualAttributes(object):
    '''
    Hold the visual attributes (usually colors) for a mesh. 
    '''
    def __init__(self, mesh):
        self.mesh = mesh

        self._vertex_colors = None
        self._face_colors   = None

    @property
    def face_colors(self):
        if self._face_colors is None:
            face_colors = np.tile(DEFAULT_COLOR,
                                  (len(self.mesh.faces), 1))
            return face_colors.astype(COLOR_DTYPE)
        return self._face_colors

    @face_colors.setter
    def face_colors(self, values):
        if np.shape(values) == np.shape(self.mesh.faces):
            self._face_colors = values
        else: 
            raise ValueError('Face colors are the wrong shape!')

    @property
    def vertex_colors(self):
        if self._vertex_colors is None:
            vertex_colors = face_to_vertex_color(self.mesh, 
                                                 self.face_colors)
            return vertex_colors
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, values):
        if np.shape(values) == np.shape(self.mesh.vertices):
            self._vertex_colors = values
        else: 
            raise ValueError('Vertex colors are the wrong shape!')

    def update_faces(self, mask):
        if not (self._face_colors is None):
            self._face_colors = self._face_colors[mask]

    def update_vertices(self, mask):
        if not (self._vertex_colors is None):
            self._vertex_colors = self._vertex_colors[mask]
