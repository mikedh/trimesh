import pyglet
import numpy as np

from copy      import deepcopy
from collections import deque
from threading import Thread
from pyglet.gl import *

from ..constants import log
from ..transformations import Arcball

#smooth only when fewer faces than this
_SMOOTH_MAX_FACES = 100000

class SceneViewer(pyglet.window.Window):
    def __init__(self, 
                 scene, 
                 smooth = None,
                 save_image = None,
                 flags = None,
                 resolution = (640,480)):

        self.scene = scene
        self.scene._redraw = self._redraw
        
        self.reset_view(flags=flags)

        visible = save_image is None
        width, height = resolution

        try:
            conf = Config(sample_buffers = 1,
                          samples        = 4,
                          depth_size     = 16,
                          double_buffer  = True)
            super(SceneViewer, self).__init__(config=conf, 
                                              visible=visible, 
                                              resizable=True,
                                              width=width,
                                              height=height)
        except pyglet.window.NoSuchConfigException:
            conf = Config(double_buffer=True)
            super(SceneViewer, self).__init__(config = conf,
                                              resizable=True,
                                              visible = visible,
                                              width=width,
                                              height=height)

        self.batch = pyglet.graphics.Batch()
        self._img  = save_image

        self.vertex_list     = {}
        self.vertex_list_md5 = {}

        for name, mesh in scene.meshes.items():
            self._add_mesh(name, mesh, smooth)
            
        self.init_gl()
        self.set_size(*resolution)
        self.update_flags()
        pyglet.app.run()

    def _redraw(self):
        self.on_draw()

    def _update_meshes(self):
        for name, mesh in self.scene.meshes.items():
            md5 = mesh.md5() + mesh.visual.md5()
            if self.vertex_list_md5[name] != md5:
                self._add_mesh(name, mesh)

    def _add_mesh(self, name_mesh, mesh, smooth=None):    
        if smooth is None:
            smooth = len(mesh.faces) < _SMOOTH_MAX_FACES
        if smooth:
            display = mesh.smoothed()
        else:
            display = mesh.copy()
            display.unmerge_vertices()
        self.vertex_list[name_mesh] = self.batch.add_indexed(*mesh_to_vertex_list(display))
        self.vertex_list_md5[name_mesh] = mesh.md5() + mesh.visual.md5()

    def reset_view(self, flags=None):
        '''
        Set view to base.
        '''
        self.view = {'wireframe'   : False,
                     'cull'        : True,
                     'translation' : np.zeros(3),
                     'center'      : self.scene.centroid,
                     'scale'       : self.scene.scale,
                     'ball'        : Arcball()}
        if isinstance(flags, dict):
            for k,v in flags.items():
                if k in self.view:
                    self.view[k] = v
        self.update_flags()

    def init_gl(self):
        glClearColor(.93, .93, 1, 1)
        #glColor3f(1, 0, 0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)

        glLightfv(GL_LIGHT0, GL_POSITION, _gl_vector(.5, .5, 1, 0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, _gl_vector(.5, .5, 1, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
        glLightfv(GL_LIGHT1, GL_POSITION, _gl_vector(1, 0, .5, 0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
        glLightfv(GL_LIGHT1, GL_SPECULAR, _gl_vector(1, 1, 1, 1))

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)

        glMaterialfv(GL_FRONT, GL_AMBIENT, _gl_vector(0.192250, 0.192250, 0.192250))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, _gl_vector(0.507540, 0.507540, 0.507540))
        glMaterialfv(GL_FRONT, GL_SPECULAR, _gl_vector(.5082730,.5082730,.5082730))

        glMaterialf(GL_FRONT, GL_SHININESS, .4 * 128.0);

        glEnable(GL_BLEND) 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 


    def toggle_culling(self):
        self.view['cull'] = not self.view['cull']
        self.update_flags()

    def toggle_wireframe(self):
        self.view['wireframe'] = not self.view['wireframe']
        self.update_flags()

    def update_flags(self):
        if self.view['wireframe']: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.view['cull']:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., width / float(height), .01, 1000.)
        glMatrixMode(GL_MODELVIEW)
        self.view['ball'].place([width/2, height/2], (width+height)/2)
        
    def on_mouse_press(self, x, y, buttons, modifiers):
        self.view['ball'].down([x,-y])
        
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        delta = np.array([dx, dy], dtype=np.float) / [self.width, self.height]

        #left mouse button, with control key down (pan)
        if ((buttons == pyglet.window.mouse.LEFT) and 
            (modifiers & pyglet.window.key.MOD_CTRL)):
            self.view['translation'][0:2] += delta

        #left mouse button, no modifier keys pressed (rotate)
        elif (buttons == pyglet.window.mouse.LEFT):
            self.view['ball'].drag([x,-y])

    def on_mouse_scroll(self, x, y, dx, dy):
        self.view['translation'][2] += ((float(dy) / self.height) * 
                                        self.view['scale'] * 5)
        
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()

    def on_draw(self):
        self._update_meshes()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # pull the new camera transform from the scene
        transform_camera = self.scene.transforms['camera']
        # apply the camera transform to the matrix stack
        glMultMatrixf(_gl_matrix(transform_camera))
        
        # dragging the mouse moves the view transform (but doesn't alter the scene)
        transform_view = _view_transform(self.view)
        glMultMatrixf(_gl_matrix(transform_view))

        # we want to render fully opaque objects first,
        # followed by objects which have transparency
        items = deque(self.scene.nodes.items())
        count_original = len(items)
        count = -1
        
        while len(items) > 0:
            count += 1
            item = items.popleft()
            name_node, name_mesh = item

            # if the flag isn't defined, this will be None
            # by checking False explicitly, it makes the default
            # behaviour to render meshes with no flag defined. 
            if self.node_flag(name_node, 'visible') == False:
                continue
                
            if self.scene.meshes[name_mesh].visual.transparency:
                # put the current item onto the back of the queue
                if count < count_original:
                    items.append(item)
                    continue

            transform = self.scene.transforms[name_node]
            # add a new matrix to the model stack
            glPushMatrix()
            # transform by the nodes transform
            glMultMatrixf(_gl_matrix(transform))
            # draw the mesh with its transform applied
            self.vertex_list[name_mesh].draw(mode=GL_TRIANGLES)
            # pop the matrix stack as we drew what we needed to draw
            glPopMatrix()

    def node_flag(self, node, flag):
        if flag in self.scene.flags[node]:
            return self.scene.flags[node][flag]
        return None
        
    def save_image(self, filename):
        colorbuffer = pyglet.image.get_buffer_manager().get_color_buffer()
        colorbuffer.save(filename)

    def flip(self):
        '''
        This function is the last thing executed in the event loop,
        so if we want to close the window (self) here is the place to do it.
        '''
        super(self.__class__, self).flip()

        if self._img is not None:
            self.save_image(self._img)
            self.close()

def _view_transform(view):
    '''
    Given a dictionary containing view parameters,
    calculate a transformation matrix. 
    '''
    transform         = view['ball'].matrix()
    transform[0:3,3]  = view['center']
    transform[0:3,3] -= np.dot(transform[0:3,0:3], view['center'])
    transform[0:3,3] += view['translation'] * view['scale']
    return transform

def mesh_to_vertex_list(mesh, group=None):
    '''
    Convert a Trimesh object to arguments for an 
    indexed vertex list constructor. 
    '''
    mesh.visual.choose()
    
    normals  = mesh.vertex_normals.reshape(-1).tolist()
    colors   = mesh.visual.vertex_colors.reshape(-1).tolist()
    faces    = mesh.faces.reshape(-1).tolist()
    vertices = mesh.vertices.reshape(-1).tolist()
    
    color_dimension = mesh.visual.vertex_colors.shape[1]
    color_type = 'c' + str(color_dimension) + 'B/static'

    args = (len(mesh.vertices), # number of vertices
            GL_TRIANGLES,       # mode 
            group,              # group
            faces,              # indices 
            ('v3f/static', vertices),
            ('n3f/static', normals),
            (color_type,   colors))
    return args
    
def _gl_matrix(array):
    '''
    Convert a sane numpy transformation matrix (row major, (4,4))
    to an stupid GLfloat transformation matrix (column major, (16,))
    '''
    a = np.array(array).T.reshape(-1)
    return (GLfloat * len(a))(*a)

def _gl_vector(array, *args):
    '''
    Convert an array and an optional set of args into a flat vector of GLfloat
    '''
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (GLfloat * len(array))(*array)
    return vector
