import pyglet
import numpy as np

from copy      import deepcopy
from threading import Thread
from pyglet.gl import *

#smooth only when fewer faces than this
_SMOOTH_MAX_FACES = 20000

class SceneViewer(pyglet.window.Window):
    def __init__(self, 
                 scene, 
                 smooth = None,
                 save_image = None,
                 resolution = (640,480)):

        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)

        self._img = save_image
        visible = save_image is None
        width, height = resolution

        try: 
            super(SceneViewer, self).__init__(config=conf, 
                                              visible=visible, 
                                              resizable=True,
                                              width=width,
                                              height=height)
        except pyglet.window.NoSuchConfigException:
            super(SceneViewer, self).__init__(resizable=True,
                                              visible = visible,
                                              width=width,
                                              height=height)
            
        self.batch        = pyglet.graphics.Batch()
        self._vertex_list = {}
        self.scene        = scene
        
        for name, mesh in scene.meshes.items():
            self._add_mesh(name, mesh, smooth)
            
        self.reset_view()
        self.init_gl()
        self.set_size(*resolution)
        self.run()

    def _add_mesh(self, node_name, mesh, smooth=None):
        if smooth is None:
            smooth = len(mesh.faces) < _SMOOTH_MAX_FACES

        # we don't want the render object to mess with the original mesh
        mesh = deepcopy(mesh)
        if smooth:
            # will merge vertices close in angle
            mesh.smooth()
        else:
            # will show faceted surfaces instead of super wrong blending
            mesh.unmerge_vertices()

        self._vertex_list[node_name] = self.batch.add_indexed(*_mesh_to_vla(mesh))

    def reset_view(self):
        self.view = {'wireframe'   : False,
                     'cull'        : True,
                     'rotation'    : np.zeros(3),
                     'translation' : np.zeros(3),
                     'center'      : self.scene.centroid()}
        
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

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        delta = np.array([dy, dx], dtype=np.float) / [self.height, -self.width]

        #left mouse button, with control key down (pan)
        if ((buttons == pyglet.window.mouse.LEFT) and 
            (modifiers & pyglet.window.key.MOD_CTRL)):
            self.view['translation'][0:2] += delta

        #left mouse button, no modifier keys pressed (rotate)
        elif (buttons == pyglet.window.mouse.LEFT):
            self.view['rotation'][0:2]+= delta

    def on_mouse_scroll(self, x, y, dx, dy):
        self.view['translation'][2] += float(dy) / self.height

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

        # update the scene with the altered view from the interface
        if (self.view['rotation'] > .01).any():
            self.scene.set_camera(ypr=self.view['rotation']*np.pi)

        # pull the new camera transform from the scene
        transform_camera = self.scene.transforms.get('camera')
        # apply the camera transform to the matrix stack
        glMultMatrixf(_gl_matrix(transform_camera))
        
        for name_node, name_mesh in self.scene.instances.items():
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
        pyglet.image.get_buffer_manager().get_color_buffer().save(filename)

    def flip(self):
        '''
        This function is the last thing executed in the event loop.
        '''
        super(self.__class__, self).flip()

        if self._img is not None:
            self.save_image(self._img)
            self.close()

    def run(self):
        pyglet.app.run()

def _mesh_to_vla(mesh):
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
            None,               # group
            indices,            # indices 
            ('v3f/static', vertices),
            ('n3f/static', normals),
            ('c3B/static', colors))
    return args
    
def _gl_matrix(array):
    # turn a numpy transformation matrix (row major, 4x4)
    # to an GLfloat transformation matrix (column major, 16)
    a = np.array(array).T.reshape(-1)
    return (GLfloat * len(a))(*a)

def _gl_vector(*args):
    return (GLfloat * len(args))(*args)
