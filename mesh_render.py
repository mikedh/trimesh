import pyglet
from pyglet.gl import *
import numpy as np
from copy import deepcopy

class MeshRender(pyglet.window.Window):
    def __init__(self, 
                 mesh         = None, 
                 smooth       = False,
                 smooth_angle = np.radians(20)):
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        try: 
            super(MeshRender, self).__init__(config=conf, resizable=True)
        except pyglet.window.NoSuchConfigException:
            super(MeshRender, self).__init__(resizable=True)
            
        self.smooth       = smooth
        self.smooth_angle = smooth_angle
        self.batch        = pyglet.graphics.Batch()        
        self.rotation     = np.zeros(3)
        self.translation  = np.zeros(3)
        self.wireframe    = False
        self.cull         = False
        self.init_gl()
        
        if mesh <> None: 
            self.add_mesh(mesh)
            self.run()

    def add_mesh(self, mesh):
        self.translation = np.array([0,0,-np.max(mesh.box_size)])
        
        if self.smooth:
            self.mesh = deepcopy(mesh)
            self.mesh.unmerge_vertices()
            self.mesh.merge_vertices_kdtree(angle_max=self.smooth_angle)
        else: self.mesh = mesh
            
        self.mesh.verify_normals()
        vertices = (self.mesh.vertices-self.mesh.centroid).reshape(-1).tolist()
        normals  = self.mesh.vertex_normals.reshape(-1).tolist()
        indices  = self.mesh.faces.reshape(-1).tolist()
        self.vertex_list = self.batch.add_indexed(len(vertices)//3, 
                                                  GL_TRIANGLES,
                                                  None,
                                                  indices,
                                                  ('v3f/static', vertices),
                                                  ('n3f/static', normals))

    def init_gl(self):
        # One-time GL setup
        glClearColor(1, 1, 1, 1)
        glColor3f(1, 0, 0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
        # but this is not the case on Linux or Mac, so remember to always 
        # include it.
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)

        # Define a simple function to create ctypes arrays of floats:
        def vec(*args):
            return (GLfloat * len(args))(*args)

        glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.5, .5, 1, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))
        glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .5, .5, 1))
        glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 1, 1, 1))

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.6, .2, 0.5, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
        glDisable(GL_CULL_FACE)
        
    def set_base_view(self):
        self.rotation    = np.zeros(3)
        self.translation = np.array([0,0,-np.max(self.mesh.box_size)])
        
    def toggle_culling(self):
        self.cull = not self.cull
        if self.cull: glEnable(GL_CULL_FACE)
        else:         glDisable(GL_CULL_FACE)
        
    def toggle_wireframe(self):
        self.wireframe = not self.wireframe
        if self.wireframe: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:              glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
    def on_resize(self, width, height):
        # Override the default on_resize handler to create a 3D projection
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., width / float(height), .01, 1000.)
        glMatrixMode(GL_MODELVIEW)
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        #left mouse button, with control key down (pan)
        if ((buttons == pyglet.window.mouse.LEFT) and 
            (modifiers & pyglet.window.key.MOD_CTRL)):
            scale = 1.0 / 100.0
            self.translation[0:2] += np.array([dx, dy]) * scale
        #left mouse button, no modifier keys pressed (rotate)
        elif (buttons == pyglet.window.mouse.LEFT):
            scale = 1.0
            self.rotation[0:2] += np.array([-1*dy, dx]) * scale
            self.rotation = np.mod(self.rotation, 720)

    def on_mouse_scroll(self, x, y, dx, dy):
        scale = 1.0 / 10.0
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
        glTranslatef(*self.translation)
        for i in xrange(2):
            glRotatef(self.rotation[i], *np.roll([1,0,0], i))
        self.batch.draw() 
        
    def run(self):
        pyglet.app.run()
