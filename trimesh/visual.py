import numpy as np
from colorsys import hsv_to_rgb
from collections import deque

from .util      import is_sequence, is_shape, tracked_array, Cache
from .constants import log

COLORS = {'red'    : [205,59,34],
          'purple' : [150,111,214],
          'blue'   : [119,158,203],
          'brown'  : [160,85,45]}
COLOR_DTYPE = np.dtype(np.uint8)
DEFAULT_COLOR = COLORS['purple']

class VisualAttributes(object):
    '''
    Hold the visual attributes (usually colors) for a mesh. 
    '''
    def __init__(self, mesh=None, **kwargs):
        self.mesh = mesh
        self._set = {'face'   : False,
                     'vertex' : False}
                     
        colors = _kwargs_to_color(mesh, **kwargs)
        self.vertex_colors, self.face_colors = colors

        # cache computed values which are cleared when
        # self.md5() changes, forcing a recompute
        self._cache = Cache(id_function = self.md5)    

    @property
    def defined(self):
        defined = np.any(self._set.values()) and self.mesh is not None
        return defined

    @property
    def transparency(self):
        cached = self._cache.get('transparency')
        if cached is not None: 
            return cached
        transparency = False
        color_max = (2**(COLOR_DTYPE.itemsize*8)) - 1
        if self._set['face']:
            transparency = (self._face_colors.ndim == 2 and
                            self._face_colors.shape[1] == 4 and
                            np.any(self._face_colors[:,3] < color_max))
        elif self._set['vertex']:
            transparency = (self._vertex_colors.ndim == 2 and
                            self._vertex_colors.shape[1] == 4 and
                            np.any(self._vertex_colors[:,3] < color_max))

        return self._cache.set(key   = 'transparency',
                               value = bool(transparency))

    def md5(self):
        md5 = ''
        if hasattr(self.mesh, 'md5'):
            md5 += self.mesh.md5()            
        if self._set['face']:
            md5 += self._face_colors.md5()
        if self._set['vertex']:
            md5 += self._vertex_colors.md5()
        return md5

    @property
    def face_colors(self):
        if not (is_sequence(self._face_colors) and
                len(self._face_colors) == len(self.mesh.faces)):
            self._face_colors = tracked_array(np.tile(DEFAULT_COLOR,
                                                      (len(self.mesh.faces), 1)))
            log.debug('Setting default colors for faces')
        return self._face_colors
        
    @face_colors.setter
    def face_colors(self, values):
        values = np.asanyarray(values)
        if values.shape in ((3,), (4,)):
            # case where a single RGB/RGBa color has been passed to the setter
            # we apply this color to all faces 
            values = np.tile(values, (len(self.mesh.faces), 1))

        self._face_colors = tracked_array(values)
        # this will only consider colors defined if they were passed as a sequence
        # is_sequence gracefully handles both None and np.array(None)
        self._set['face'] = is_sequence(values)

    @property
    def vertex_colors(self):
        if not (is_sequence(self._vertex_colors) and
                len(self._vertex_colors) == len(self.mesh.vertices)):
            log.debug('Vertex colors being generated from face colors')
            self._vertex_colors = face_to_vertex_color(self.mesh, 
                                                       self.face_colors)
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, values):
        self._vertex_colors = tracked_array(values)
        self._set['vertex'] = is_sequence(values)

    def update_faces(self, mask):
        if not self._set['face']: 
            return
        try: self._face_colors = self._face_colors[mask]
        except: log.warning('Face colors not updated', exc_info=True) 
            
    def update_vertices(self, mask):
        if not self._set['vertex']:
            return
        try:    
            self.vertex_colors = self._vertex_colors[mask]
        except: 
            log.debug('Vertex colors not updated', exc_info=True)

    def subsets(self, faces_sequence):
        result = deque()
        for f in faces_sequence:
            result.append(VisualAttributes())
            if self._set['face']:
                result[-1].face_colors = self.face_colors[np.array(list(f))]
        return np.array(result)

    def union(self, others):
        return visuals_union(np.append(self, others))

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

    if mesh is None:
        result = [None, None]
        if 'face_colors' in kwargs:
            result[1] = np.asanyarray(kwargs['face_colors'])
        if 'vertex_colors' in kwargs:
            result[0] = np.asanyarray(kwargs['vertex_colors'])
        return result
        
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

def visuals_union(visuals, *args):
    visuals = np.append(visuals, args)
    color = {'face_colors'   : None,
             'vertex_colors' : None}
    if all(is_shape(i._face_colors, (-1,(3,4))) for i in visuals):
        color['face_colors'] = np.vstack([_rgba(i._face_colors) for i in visuals])
    if all(is_shape(i._vertex_colors, (-1,(3,4))) for i in visuals):
        color['vertex_colors'] = np.vstack([i._vertex_colors for i in visuals])
    return VisualAttributes(**color)

def _rgba(colors, dtype=COLOR_DTYPE):
    colors = np.asanyarray(colors)
    if is_shape(colors, (-1,3)):
        opaque = (2**(np.dtype(dtype).itemsize * 8)) - 1
        colors = np.column_stack((colors,
                                  opaque * np.ones(len(colors))))
    return colors.astype(dtype)

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
