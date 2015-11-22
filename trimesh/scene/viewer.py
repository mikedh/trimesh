import pyglet
import numpy as np

from copy      import deepcopy
from threading import Thread
from pyglet.gl import *

from ..transformations import euler_matrix

#smooth only when fewer faces than this
_SMOOTH_MAX_FACES = 20000

class SceneViewer(pyglet.window.Window):
    def __init__(self, 
                 scene, 
                 smooth = None,
                 save_image = None,
                 resolution = (640,480)):

        self.events = []
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
        self.scene = scene
        self._img  = save_image
        self._vertex_list = {}
        self.reset_view()
        
        for name, mesh in scene.meshes.items():
            self._add_mesh(name, mesh, smooth)
            
        self.init_gl()
        self.set_size(*resolution)
        pyglet.app.run()

    def _add_mesh(self, node_name, mesh, smooth=None):
        if smooth is None:
            smooth = len(mesh.faces) < _SMOOTH_MAX_FACES

        # we don't want the render object to mess with the original mesh
        mesh_render = mesh.copy()
        
        if smooth:
            # merge vertices close in angle (uses a KDtree), can be slow on large meshes
            mesh_render.smooth()
        else:
            mesh_render.unmerge_vertices()

        self._vertex_list[node_name] = self.batch.add_indexed(*_mesh_to_vertex_list(mesh_render))

    def reset_view(self):
        '''
        Set view to base.
        '''
        self.view = {'wireframe'   : False,
                     'cull'        : True,
                     'rotation'    : np.zeros(3),
                     'translation' : np.zeros(3),
                     'center'      : self.scene.centroid,
                     'scale'       : self.scene.scale}

        
    def init_gl(self):
        glClearColor(1, 1, 1, 1)
        glColor3f(1, 0, 0)
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

    def toggle_culling(self):
        self.view['cull'] = not self.view['cull']
        if self.view['cull']:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)
        
    def toggle_wireframe(self):
        self.view['wireframe'] = not self.view['wireframe']
        if self.view['wireframe']: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., width / float(height), .01, 1000.)
        glMatrixMode(GL_MODELVIEW)
        self.events.append('resize')
        
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        delta = np.array([dx, dy], dtype=np.float) / [self.width, self.height]

        #left mouse button, with control key down (pan)
        if ((buttons == pyglet.window.mouse.LEFT) and 
            (modifiers & pyglet.window.key.MOD_CTRL)):
            self.view['translation'][0:2] += delta

        #left mouse button, no modifier keys pressed (rotate)
        elif (buttons == pyglet.window.mouse.LEFT):
            self.view['rotation'][0:2]+= delta[::-1]*[-1,1]

    def on_mouse_scroll(self, x, y, dx, dy):
        self.view['translation'][2] += (float(dy) / self.height) * 25

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()

    def on_draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # pull the new camera transform from the scene
        transform_camera = self.scene.transforms.get('camera')
        # apply the camera transform to the matrix stack
        glMultMatrixf(_gl_matrix(transform_camera))

        # dragging the mouse moves the view transform (but doesn't alter the scene)
        transform_view = _view_transform(self.view)
        glMultMatrixf(_gl_matrix(transform_view))

        for name_node, name_mesh in self.scene.instances.items():
            if ('visible' in self.scene.flags[name_node] and 
                not self.scene.flags[name_node]['visible']):
                continue
            transform = self.scene.transforms.get(name_node)
            # add a new matrix to the model stack
            glPushMatrix()
            # transform by the nodes transform
            glMultMatrixf(_gl_matrix(transform))
            # draw the mesh with its transform applied
            self._vertex_list[name_mesh].draw(mode=GL_TRIANGLES)
            # pop the matrix stack as we drew what we needed to draw
            glPopMatrix()
        
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
    transform = euler_matrix(*(view['rotation']*np.pi))
    transform[0:3,3]  = view['center']
    transform[0:3,3] -= np.dot(transform[0:3,0:3], view['center'])
    transform[0:3,3] += view['translation']*view['scale']
    return transform

def _mesh_to_vertex_list(mesh, group=None):
    '''
    Convert a Trimesh object to arguments for an 
    indexed vertex list constructor. 
    '''
    normals  = mesh.vertex_normals.reshape(-1).tolist()
    colors   = mesh.visual.vertex_colors.reshape(-1).tolist()
    indices  = mesh.faces.reshape(-1).tolist()
    vertices = mesh.vertices.reshape(-1).tolist()
    
    args = (len(vertices) // 3, # count
            GL_TRIANGLES,       # mode 
            group,              # group
            indices,            # indices 
            ('v3f/static', vertices),
            ('n3f/static', normals),
            ('c3B/static', colors))
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
    Convert an array and a set of args into a flat vector of GLfloat
    '''
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (GLfloat * len(array))(*array)
    return vector
