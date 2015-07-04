import numpy as np
from colorsys import hsv_to_rgb
from .util import unitize
from .constants import TOL_ZERO
COLORS = {'red'    : [194,59,34],
          'purple' : [150,111,214],
          'blue'   : [119,158,203],
          'brown'  : [160,85,45]}

COLOR_DTYPE = np.uint8
DEFAULT_COLOR = COLORS['purple']



def random_color(dtype=np.uint8):
    golden_ratio_conjugate = 0.618033988749895
    h     = np.mod(np.random.random() + golden_ratio_conjugate, 1)
    color = np.array(hsv_to_rgb(h, 0.5, 0.95))
    if np.dtype(dtype).kind in 'iu':
        max_value = (2**(np.dtype(dtype).itemsize * 8)) - 1
        color    *= max_value
    color     = color.astype(dtype)
    return color

class VisualAttributes(object):
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
        
    @property
    def vertex_colors(self):
        if self._vertex_colors is None:
            vertex_colors = face_to_vertex_color(self.mesh, 
                                                 self.face_colors)
            return vertex_colors
        return self._face_colors

    def update_faces(self, mask):
        pass

    def update_vertices(self, mask):
        pass

    @property
    def vertex_colors_ok(self):
        return np.shape(self.vertex_colors) == self.mesh.vertices.shape

    @property
    def face_colors_ok(self):
        return np.shape(self.face_colors) == self.mesh.faces.shape

    def verify_colors(self):
        '''
        If face colors are not defined, define them. 
        '''
        self.verify_face_colors()
        self.verify_vertex_colors()

    def verify_face_colors(self):
        '''
        If face colors are not defined, define them. 
        '''
        if not self.face_colors_ok:
            self.set_face_colors()

    def verify_vertex_colors(self):
        '''
        Populate self.vertex_colors
        If self.face_colors are defined, we use those values to generate
        vertex colors. If not, we just set them to the DEFAULT_COLOR
        '''
        if self.vertex_colors_ok:
            return
        elif self.face_colors_ok:
            # case where face_colors is populated, but vertex_colors isn't
            # we then generate vertex colors from the face colors
            vertex_colors = np.zeros((len(self.mesh.vertices), 3,3))
            vertex_colors[[self.mesh.faces[:,0],0]] = self.face_colors
            vertex_colors[[self.mesh.faces[:,1],1]] = self.face_colors
            vertex_colors[[self.mesh.faces[:,2],2]] = self.face_colors
            vertex_colors  = geometry.unitize(np.mean(vertex_colors, axis=1))
            vertex_colors *= (255.0 / np.max(vertex_colors, axis=1).reshape((-1,1)))
            self.vertex_colors = vertex_colors.astype(int)
            log.debug('Setting vertex colors from face colors')
        else:
            self.vertex_colors = np.tile(color.DEFAULT_COLOR, (len(self.mesh.vertices), 1))
            log.debug('Vertex colors set to default')

    def set_face_colors(self, face_color=None):
        '''
        Apply face colors. If none are defined, set to default color
        '''
        if face_color is None: 
            face_color = color.DEFAULT_COLOR
        self.face_colors = np.tile(face_color, (len(self.mesh.faces), 1))
        log.debug('Set face colors to %s', str(face_color))

def face_to_vertex_color(mesh, face_colors):
    vertex_colors = np.zeros((len(mesh.vertices), 3,3))
    vertex_colors[[mesh.faces[:,0],0]] = face_colors
    vertex_colors[[mesh.faces[:,1],1]] = face_colors
    vertex_colors[[mesh.faces[:,2],2]] = face_colors    
    populated = np.all(vertex_colors > TOL_ZERO, axis=2).sum(axis=1)
    vertex_colors = np.sum(vertex_colors, axis=1) / populated.reshape((-1,1))
    return vertex_colors.astype(face_colors.dtype)
