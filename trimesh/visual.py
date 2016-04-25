import numpy as np
from colorsys import hsv_to_rgb
from collections import deque

from .util      import is_sequence, is_shape, tracked_array, Cache, DataStore
from .constants import log

COLORS = {'red'    : [205,59,34,255],
          'purple' : [150,111,214,255],
          'blue'   : [119,158,203,255],
          'brown'  : [160,85,45,255]}
COLOR_DTYPE = np.dtype(np.uint8)
DEFAULT_COLOR = np.array(COLORS['purple'], dtype=COLOR_DTYPE)

class VisualAttributes(object):
    '''
    Hold the visual attributes (usually colors) for a mesh. 

    This is a bit of a dumpster fire and probably needs a re-write
    '''
    def __init__(self, mesh=None, dtype=None, **kwargs):
        self.mesh = mesh

        self._validate = True
        self._data = DataStore()
        self._cache = Cache(id_function = self._data.md5)

        if dtype is None: 
            dtype = COLOR_DTYPE
        self.dtype = dtype

        colors = _kwargs_to_color(mesh, **kwargs)
        self.vertex_colors, self.face_colors = colors

    def choose(self):
        '''
        If both face and vertex colors are defined, choose one of them.
        '''
        if all(self._set.values()):
            sig_face   = self._data['face_colors'].ptp(axis=0).sum()
            sig_vertex = self._data['vertex_colors'].ptp(axis=0).sum()
            if sig_face > sig_vertex:
                self.vertex_colors = None
            else:
                self.face_colors = None
        
    @property
    def _set(self):
        result = {'face'   : is_shape(self._data['face_colors'], (-1, (3,4))),
                  'vertex' : is_shape(self._data['vertex_colors'], (-1,(3,4)))}
        return result

    @property
    def defined(self):
        defined = np.any(self._set.values()) 
        defined = defined and self.mesh is not None
        return defined

    @property
    def transparency(self):
        '''
        Returns
        ------------
        transparency: bool, does the visual attributes contain any transparency
        '''
        cached = self._cache.get('transparency')
        if cached is not None: 
            return cached
        transparency = False
        color_max = (2**(COLOR_DTYPE.itemsize*8)) - 1
        if self._set['face']:
            transparency = (is_shape(self._data['face_colors'], (-1,4)) and 
                            np.any(self._data['face_colors'][:,3] < color_max))
        elif self._set['vertex']:
            transparency = (is_shape(self._data['vertex_colors'], (-1,4)) and
                            np.any(self._vertex_colors[:,3] < color_max))

        return self._cache.set(key   = 'transparency',
                               value = bool(transparency))

    def md5(self):
        return self._data.md5()

    @property
    def face_colors(self):
        def ok(blob):
            return is_shape(blob, (len(self.mesh.faces), (3,4)))

        stored = self._data['face_colors']
        cached = self._cache['face_colors']
        if ok(stored):
            return stored
        elif ok(cached):
            return cached
        
        log.debug('Returning default colors for faces.')
        self._cache['face_colors'] = np.tile(DEFAULT_COLOR, 
                                             (len(self.mesh.faces), 1))
        return self._cache['face_colors']

    @face_colors.setter
    def face_colors(self, values):
        values = np.asanyarray(values)
        if values.shape in ((3,), (4,)):
            # case where a single RGB/RGBa color has been passed to the setter
            # we apply this color to all faces 
            values = np.tile(values, (len(self.mesh.faces), 1))
        self._data['face_colors'] = rgba(values, dtype=self.dtype)

    @property
    def vertex_colors(self):
        def ok(blob):
            return is_shape(blob, (len(self.mesh.vertices), (3,4)))

        stored = self._data['vertex_colors']
        cached = self._cache['vertex_colors']
        if ok(stored):
            return stored
        elif ok(cached):
            return cached

        log.debug('Vertex colors being generated from face colors')
        colors = face_to_vertex_color(self.mesh, self.face_colors)
        self._cache['vertex_colors'] = colors
        return colors
   
    @vertex_colors.setter
    def vertex_colors(self, values):
        self._data['vertex_colors'] = rgba(values, dtype=self.dtype)

    def update_faces(self, mask):
        stored = self._data['face_colors']
        if not is_shape(stored, (-1,(3,4))):
            return 
        try: 
            self._data['face_colors'] = stored[mask]
        except: 
            log.warning('Face colors not updated', exc_info=True) 
            
    def update_vertices(self, mask):
        stored = self._data['vertex_colors']
        if not is_shape(stored, (-1, (3,4))):
            return
        try:    
            self._data['vertex_colors'] = stored[mask]
        except: 
            log.debug('Vertex colors not updated', exc_info=True)

    def subsets(self, faces_sequence):
        result = [VisualAttributes() for i in range(len(faces_sequence))]
        if self._set['face']:
            face = self._data['face_colors']
            for i, f in enumerate(faces_sequence):
                result[i].face_colors = face[list(f)]
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

    vertex_ok = True
    vertex = [None] * len(visuals)

    face_ok = True
    face = [None] * len(visuals)

    for i, v in enumerate(visuals):
        face_ok   = face_ok and v._set['face']
        vertex_ok = vertex_ok and v._set['vertex']

        if face_ok:
            if v.mesh is None:
                # if the mesh is None, don't force a 
                # dimension check for the colors
                face[i] = rgba(v._data['face_colors'])
            else: 
                face[i] = rgba(v.face_colors)
        if vertex_ok:
            if v.mesh is None:
                vertex[i] = rgba(v._data['vertex_colors'])
            else:
                vertex[i] = rgba(v.vertex_colors)
            
    if face_ok:
        color['face_colors'] = np.vstack(face)
    if vertex_ok:
        color['vertex_colors'] = np.vstack(vertex)

    return VisualAttributes(**color)

def color_to_float(color, dtype=None):
    color = np.asanyarray(color)
    if dtype is None:
        dtype = color.dtype
    else:
        color = color.astype(dtype)
    if dtype.kind in 'ui':
        signed = int(dtype.kind == 'i')
        color_max = float((2**((dtype.itemsize*8) - signed)) - 1)
        color = color.astype(np.float) / color_max
    return color

def rgba(colors, dtype=None):
    '''
    Convert an RGB color to an RGBA color.

    Arguments
    ----------
    colors: (n,[3|4]) set of RGB or RGBA colors
    
    Returns
    ----------
    colors: (n,4) set of RGBA colors
    '''
    if not is_sequence(colors): 
        return
    if dtype is None:
        dtype = COLOR_DTYPE
    colors = np.asanyarray(colors, dtype=dtype)
    if is_shape(colors, (-1,3)):
        opaque = (2**(np.dtype(dtype).itemsize * 8)) - 1
        colors = np.column_stack((colors,
                                  opaque * np.ones(len(colors)))).astype(dtype)
    return colors

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
    color = color.astype(dtype)
    return color

def vertex_to_face_colors(vertex_colors, faces):
    face_colors = vertex_colors[faces].mean(axis=2).astype(vertex_colors.dtype)
    return face_colors

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
