import pyglet
from pyglet.gl import *
import numpy as np
from copy import deepcopy

from string import Template


# when smoothing isn't specified, only smooth if there
# are fewer faces than this, to prevent lockups in normal use
FACE_COUNT_SMOOTH = 20000

class MeshRender(pyglet.window.Window):
    def __init__(self, 
                 mesh         = None, 
                 smooth       = None,
                 smooth_angle = np.radians(20),
                 base_trans   = None,
                 base_rot     = None,
                 center       = True):
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
        self.cull         = True
        self.init_gl()
        
        if mesh != None: 
            self.add_mesh(mesh)
            self.run()

    def add_mesh(self, mesh):
        self.translation = np.array([0,0,-np.max(mesh.box_size)])
        
        if self.smooth == None: smooth = len(mesh.faces) < FACE_COUNT_SMOOTH
        else:                   smooth = self.smooth
        
        if smooth:
            self.mesh = deepcopy(mesh)
            self.mesh.unmerge_vertices()
            #self.mesh.merge_vertices(angle_max=self.smooth_angle)
        else: 
            self.mesh = mesh
     
        self.mesh.generate_vertex_colors()
        self.mesh.verify_normals()
        
        vertices = (self.mesh.vertices-self.mesh.centroid).reshape(-1).tolist()
        normals  = self.mesh.vertex_normals.reshape(-1).tolist()
        colors   = self.mesh.vertex_colors.reshape(-1).tolist()
        indices  = self.mesh.faces.reshape(-1).tolist()
        
        self.vertex_list = self.batch.add_indexed(len(vertices)//3, # count
                                                  GL_TRIANGLES,     # mode 
                                                  None,             # group
                                                  indices,          # indices 
                                                  ('v3f/static', vertices),
                                                  ('n3f/static', normals),
                                                  ('c3B/static', colors))

    def init_gl(self):
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

        #glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(1,1,1,1))
        #glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
        #glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        
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
        for i in range(2):
            glRotatef(self.rotation[i], *np.roll([1,0,0], i))
        self.batch.draw() 
        
    def run(self):
        pyglet.app.run()

def to_yafaray_xml(mesh, 
                   material_xml='<set_material sval=\"defaultMat\"/>'):

    '''
    Return the yafaray xml string for the mesh specified. 

    http://www.yafaray.org/development/documentation/XMLspecs#Mesh
    http://www.wings3d.com/forum/showthread.php?tid=146
    '''
    from string import Template
    from StringIO import StringIO

    format_dict = {'vertex_count' : len(mesh.vertices),
                   'face_count'   : len(mesh.faces)}

    s = StringIO()
    np.savetxt(s, mesh.vertices, '<p x=\"%.6f\" y=\"%.6f\" z=\"%.6f\"/>'); s.seek(0)
    format_dict['vertex_xml'] = s.read()

    s = StringIO()
    np.savetxt(s, mesh.faces, '<f a=\"%d\" b=\"%d\" c=\"%d\"/>'); s.seek(0)
    format_dict['face_xml'] = s.read()

    mesh_template  = '<mesh vertices=\"$vertex_count\" '
    mesh_template += 'faces=\"$face_count\" '
    mesh_template += 'has_orco=\"false\" has_uv=\"false\" type=\"0\">\n'
    mesh_template += '$vertex_xml\n'
    mesh_template += material_xml
    mesh_template += '$face_xml\n'
    mesh_template += '</mesh>\n'
    mesh_template += '<smooth ID=\"0\" angle="25.000\"/>'

    result = Template(mesh_template).substitute(format_dict)
    return result

def yafa_scene(mesh):
    import os, inspect

    MODULE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    template    = Template(open(os.path.join(MODULE_PATH,
                                             'templates',
                                             'yafaray_template.xml'), 'rb').read())

    mesh_xml = to_yafaray_xml(mesh)

    camera_origin = tuple(mesh.centroid + [0,0,mesh.box_size.max()])
    camera_dest   = tuple(mesh.centroid)
    camera_up     = (0.,1.,0.)
    camera_res    = (640, 480)

    camera_xml  = '<camera name=\"cam\">\n'
    camera_xml += '<from x=\"%.6f\" y=\"%.6f\" z=\"%.6f\"/>\n' % camera_origin
    camera_xml += '<resx ival=\"%d\"/>\n<resy ival=\"%d\"/>\n' % camera_res
    camera_xml += '<to x=\"%.6f\" y=\"%.6f\" z=\"%.6f\"/>\n'   % camera_dest
    camera_xml += '<up x=\"%.6f\" y=\"%.6f\" z=\"%.6f\"/>\n'   % camera_up
    camera_xml += '<type sval=\"perspective\"/>\n</camera>'

    res_xml     = '<width ival=\"%d\">\n<height ival=\"%d\">\n' % camera_res

    result = template.substitute(MESH_XML   = mesh_xml, 
                                 CAMERA_XML = camera_xml,
                                 RES_XML    = res_xml)
    return result
    
