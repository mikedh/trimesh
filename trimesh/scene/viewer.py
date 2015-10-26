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
                 block  = True):

        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        try: 
            super(SceneViewer, self).__init__(config=conf, resizable=True)
        except pyglet.window.NoSuchConfigException:
            super(SceneViewer, self).__init__(resizable=True)
            
        self.batch        = pyglet.graphics.Batch()        
        self.rotation     = np.zeros(3)
        self.translation  = np.zeros(3)
        self.wireframe    = False
        self.cull         = True
        self.init_gl()

        self._vertex_list = {}
        self._scale  = 1.0
        self.scene = scene
        
        for name, mesh in scene.meshes.items():
            self._add_mesh(name, mesh, smooth)

        if block: 
            self.run()
        else:
            self._thread = Thread(target=self.run)
            self._thread.start()

    def _add_mesh(self, node_name, mesh, smooth=False):                
        if smooth is None:
            smooth = len(mesh.faces) < _SMOOTH_MAX_FACES

        smooth = False
        # we don't want the render object to mess with the original mesh
        mesh = deepcopy(mesh)
        if smooth:
            # will merge vertices close in angle
            mesh.smooth()
        else:
            # will show faceted surfaces instead of super wrong blending
            mesh.unmerge_vertices()

        self.set_base_view(mesh)    
        self._scale = mesh.scale
        self._vertex_list[node_name] = self.batch.add_indexed(*_mesh_to_vla(mesh))
        
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

        #glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, _gl_vector(1,1,1,1))
        #glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, _gl_vector((1, 1, 1, 1))
        #glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        
        #glShadeModel(GL_FLAT)


    def set_base_view(self, mesh):
        self.rotation    = np.zeros(3)
        self.translation = mesh.centroid + [0,0,-np.max(mesh.box_size)]
        
    def toggle_culling(self):
        self.cull = not self.cull
        if self.cull: glEnable(GL_CULL_FACE)
        else:         glDisable(GL_CULL_FACE)
        
    def toggle_wireframe(self):
        self.wireframe = not self.wireframe
        if self.wireframe: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:              glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., width / float(height), .01, 1000.)
        glMatrixMode(GL_MODELVIEW)
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        #left mouse button, with control key down (pan)
        if ((buttons == pyglet.window.mouse.LEFT) and 
            (modifiers & pyglet.window.key.MOD_CTRL)):
            scale = self._scale / 100.0
            self.translation[0:2] += np.array([dx, dy]) * scale
        #left mouse button, no modifier keys pressed (rotate)
        elif (buttons == pyglet.window.mouse.LEFT):
            scale = 1.0
            self.rotation[0:2] += np.array([-1*dy, dx]) * scale
            self.rotation = np.mod(self.rotation, 720)

    def on_mouse_scroll(self, x, y, dx, dy):
        scale = self._scale / 10.0
        self.translation[2] += dy * scale

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.set_base_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()

    def on_draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # move the 'view' before drawing anything
        glTranslatef(*self.translation)
        for i in range(2):
            glRotatef(self.rotation[i], *np.roll([1,0,0], i))

        for name_node, name_mesh in self.scene.nodes.items():
            transform = self.scene.transforms.get(name_node)
            glPushMatrix()
            glMultMatrixf(_gl_matrix(transform))
            self._vertex_list[name_mesh].draw(mode=GL_TRIANGLES)
            glPopMatrix()

    def run(self):
        pyglet.app.run()

def _mesh_to_vla(mesh):
    '''
    Convert a Trimesh object to args for an 
    indexed vertex list constructor. 
    '''
    vertices = mesh.vertices.reshape(-1).tolist()
    normals  = mesh.vertex_normals.reshape(-1).tolist()
    colors   = mesh.visual.vertex_colors.reshape(-1).tolist()
    indices  = mesh.faces.reshape(-1).tolist()

    args = (len(vertices) // 3, # count
            GL_TRIANGLES,       # mode 
            None,               # group
            indices,            # indices 
            ('v3f/static', vertices),
            ('n3f/static', normals),
            ('c3B/static', colors))
    return args
    
def _gl_matrix(array):
    a = np.array(array).T.reshape(-1)
    return (GLfloat * len(a))(*a)

def _gl_vector(*args):
    return (GLfloat * len(args))(*args)
