import numpy as np
from colorsys import hsv_to_rgb
from collections import deque

from .constants import log

COLORS = {'red'    : [205,59,34],
          'purple' : [150,111,214],
          'blue'   : [119,158,203],
          'brown'  : [160,85,45]}
COLOR_DTYPE = np.uint8
DEFAULT_COLOR = COLORS['purple']

class VisualAttributes(object):
    '''
    Hold the visual attributes (usually colors) for a mesh. 
    '''
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh

        self._set = {'face'   : False,
                     'vertex' : False}

        colors =  _kwargs_to_color(mesh, **kwargs)
        self.vertex_colors, self.face_colors = colors

    @property
    def defined(self):
        defined = self._face_ok and self._set['face']
        defined = defined or (self._vertex_ok and self._set['vertex'])
        return defined

    @property
    def _face_ok(self):
        ok = self._face_colors is not None
        ok = ok and len(self._face_colors.shape) == 2
        ok = ok and len(self._face_colors) == len(self.mesh.faces)
        return ok

    @property
    def _vertex_ok(self):
        ok = self._vertex_colors is not None
        ok = ok and len(self._vertex_colors.shape) == 2
        ok = ok and (len(self._vertex_colors) == len(self.mesh.vertices))
        return ok

    @property
    def face_colors(self):
        if not self._face_ok:
            log.warn('Faces being set to default color')
            self._face_colors = np.tile(DEFAULT_COLOR,
                                        (len(self.mesh.faces), 1))
            self._set['face'] = False
        return self._face_colors

    @face_colors.setter
    def face_colors(self, values):
        if np.shape(values) == (3,):
            # case where a single RGB color has been passed to the setter
            # here we apply this color to all faces 
            self.face_colors = np.tile(values, 
                                       (len(self.mesh.faces), 1))
        else:
            self._face_colors = np.array(values)
        self._set['face'] = self._face_ok

    @property
    def vertex_colors(self):
        if not self._vertex_ok:
            log.warn('Vertex colors being generated.')
            self._vertex_colors = face_to_vertex_color(self.mesh, 
                                                       self.face_colors)
            self._set['vertex'] = False
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, values):
        self._vertex_colors = np.array(values)
        self._set['vertex'] = self._vertex_ok

    def update_faces(self, mask):
        try:
            self._face_colors = self._face_colors[mask]
        except:
            pass

    def update_vertices(self, mask):
        try:
            self.vertex_colors = self._vertex_colors[mask]
        except: 
            pass

def _kwargs_to_color(mesh, **kwargs):
    '''
    Given a set of keyword arguments, see if any reference color
    in their name, and match the dimensions of the mesh.
    '''

    def pick_option(vf):
        if any(i is None for i in vf):
            return vf
        result = [None, None]
        signal = [i.ptp(axis=0).sum() for i in vf]
        signal_max = np.argmax(signal)
        result[signal_max] = vf[signal_max]
        return result
    def pick_color(sequence):
        if len(sequence) == 0:
            return None
        elif len(sequence) == 1:
            return sequence[0]
        else:
            signal = [i.ptp(axis=0).sum() for i in sequence]
            signal_max = np.argmax(signal)
            return sequence[signal_max]
        
    vertex = deque()
    face   = deque()
    
    for key in kwargs.keys():
        if not ('color' in key): 
            continue
        value = np.asanyarray(kwargs[key])
        if len(value) == len(mesh.vertices):
            vertex.append(value)
        elif len(value) == len(mesh.faces):
            face.append(value)
    return pick_option([pick_color(i) for i in [vertex, face]])

def random_color(dtype=COLOR_DTYPE):
    '''
    Return a random RGB color using datatype specified.
    '''
    hue  = np.random.random() + .61803
    hue %= 1.0
    color = np.array(hsv_to_rgb(hue, .99, .99))
    if np.dtype(dtype).kind in 'iu':
        max_value = (2**(np.dtype(dtype).itemsize * 8)) - 1
        color    *= max_value
    color     = color.astype(dtype)
    return color

def face_to_vertex_color(mesh, face_colors, dtype=COLOR_DTYPE):
    '''
    Convert a set of face colors into a set of vertex colors.
    '''
    
    color_dim = np.shape(face_colors)[1]

    vertex_colors = np.zeros((len(mesh.vertices),3,color_dim))
    population    = np.zeros((len(mesh.vertices),3), dtype=np.bool)

    vertex_colors[[mesh.faces[:,0],0]] = face_colors
    vertex_colors[[mesh.faces[:,1],1]] = face_colors
    vertex_colors[[mesh.faces[:,2],2]] = face_colors

    population[[mesh.faces[:,0], 0]] = True
    population[[mesh.faces[:,1], 1]] = True
    population[[mesh.faces[:,2], 2]] = True

    # clip the population sum to 1, to avoid a division error in edge cases
    populated     = np.clip(population.sum(axis=1), 1, 3)
    vertex_colors = vertex_colors.sum(axis=1) / populated.reshape((-1,1))
    return vertex_colors.astype(dtype)
