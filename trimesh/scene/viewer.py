import pyglet
import pyglet.gl as gl
import numpy as np

import platform

from collections import deque

from .. import util
from ..transformations import Arcball

# smooth only when fewer faces than this
_SMOOTH_MAX_FACES = 100000


class SceneViewer(pyglet.window.Window):

    def __init__(self,
                 scene,
                 smooth=True,
                 save_image=None,
                 flags=None,
                 resolution=(640, 480)):

        self.scene = scene
        self.scene._redraw = self._redraw

        if not ('camera' in scene.graph.nodes):
            scene.set_camera()

        self.reset_view(flags=flags)

        visible = (save_image is None) or (platform.system() != 'Linux')
        width, height = resolution

        try:
            conf = gl.Config(sample_buffers=1,
                             samples=4,
                             depth_size=16,
                             double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              visible=visible,
                                              resizable=True,
                                              width=width,
                                              height=height)
        except pyglet.window.NoSuchConfigException:
            conf = gl.Config(double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              resizable=True,
                                              visible=visible,
                                              width=width,
                                              height=height)

        self.batch = pyglet.graphics.Batch()
        self._img = save_image
        self._smooth = smooth

        self.vertex_list = {}
        self.vertex_list_md5 = {}
        self.vertex_list_mode = {}

        for name, mesh in scene.geometry.items():
            self.add_geometry(name=name,
                              geometry=mesh)
        self.init_gl()
        self.set_size(*resolution)
        self.update_flags()
        pyglet.app.run()

    def _redraw(self):
        self.on_draw()

    def _update_meshes(self):
        for name, mesh in self.scene.geometry.items():
            if self.vertex_list_md5[name] != geometry_md5(mesh):
                self.add_geometry(name, mesh)

    def _add_mesh(self, name, mesh):
        if self._smooth and len(mesh.faces) < _SMOOTH_MAX_FACES:
            display = mesh.smoothed()
        else:
            display = mesh.copy()
            display.unmerge_vertices()

        self.vertex_list[name] = self.batch.add_indexed(
            *mesh_to_vertex_list(display))
        self.vertex_list_md5[name] = geometry_md5(mesh)
        self.vertex_list_mode[name] = gl.GL_TRIANGLES

    def _add_path(self, name, path):
        self.vertex_list[name] = self.batch.add_indexed(
            *path_to_vertex_list(path))
        self.vertex_list_md5[name] = geometry_md5(path)
        self.vertex_list_mode[name] = gl.GL_LINES

    def _add_points(self, name, pointcloud):
        self.vertex_list[name] = self.batch.add_indexed(*points_to_vertex_list(pointcloud.vertices,
                                                                               pointcloud.vertices_color))
        self.vertex_list_md5[name] = geometry_md5(pointcloud)
        self.vertex_list_mode[name] = gl.GL_POINTS

    def add_geometry(self, name, geometry):
        if util.is_instance_named(geometry, 'Trimesh'):
            return self._add_mesh(name, geometry)
        elif util.is_instance_named(geometry, 'Path3D'):
            return self._add_path(name, geometry)
        elif util.is_instance_named(geometry, 'Path2D'):
            return self._add_path(name, geometry.to_3D())
        elif util.is_instance_named(geometry, 'PointCloud'):
            return self._add_points(name, geometry)
        else:
            raise ValueError('Geometry passed is not a viewable type!')

    def reset_view(self, flags=None):
        '''
        Set view to base.
        '''
        self.view = {'wireframe': False,
                     'cull': True,
                     'translation': np.zeros(3),
                     'center': self.scene.centroid,
                     'scale': self.scene.scale,
                     'ball': Arcball()}

        if (self.width is not None and
                self.height is not None):
            self.view['ball'].place([self.width / 2.0,
                                     self.height / 2.0],
                                    (self.width + self.height) / 2.0)

        if isinstance(flags, dict):
            for k, v in flags.items():
                if k in self.view:
                    self.view[k] = v
        self.update_flags()

    def init_gl(self):
        gl.glClearColor(.93, .93, 1, 1)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(.5, .5, 1, 0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, _gl_vector(.5, .5, 1, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, _gl_vector(1, 0, .5, 0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))

        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT,
                        _gl_vector(0.192250, 0.192250, 0.192250))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE,
                        _gl_vector(0.507540, 0.507540, 0.507540))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR,
                        _gl_vector(.5082730, .5082730, .5082730))

        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, .4 * 128.0)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        gl.glLineWidth(1.5)
        gl.glPointSize(4)

    def toggle_culling(self):
        self.view['cull'] = not self.view['cull']
        self.update_flags()

    def toggle_wireframe(self):
        self.view['wireframe'] = not self.view['wireframe']
        self.update_flags()

    def update_flags(self):
        if self.view['wireframe']:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if self.view['cull']:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

    def on_resize(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(60., width / float(height), .01,
                          self.scene.scale * 5.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        self.view['ball'].place([width / 2, height / 2], (width + height) / 2)

    def on_mouse_press(self, x, y, buttons, modifiers):
        self.view['ball'].down([x, -y])

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        delta = np.array([dx, dy], dtype=np.float) / [self.width, self.height]

        # left mouse button, with control key down (pan)
        if ((buttons == pyglet.window.mouse.LEFT) and
                (modifiers & pyglet.window.key.MOD_CTRL)):
            self.view['translation'][0:2] += delta

        # left mouse button, no modifier keys pressed (rotate)
        elif (buttons == pyglet.window.mouse.LEFT):
            self.view['ball'].drag([x, -y])

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
        self._update_meshes()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # pull the new camera transform from the scene
        transform_camera, _junk = self.scene.graph['camera']
        # apply the camera transform to the matrix stack
        gl.glMultMatrixf(_gl_matrix(transform_camera))

        # dragging the mouse moves the view transform (but doesn't alter the
        # scene)
        transform_view = _view_transform(self.view)
        gl.glMultMatrixf(_gl_matrix(transform_view))

        # we want to render fully opaque objects first,
        # followed by objects which have transparency
        node_names = deque(self.scene.graph.nodes_geometry)
        count_original = len(node_names)
        count = -1

        while len(node_names) > 0:
            count += 1
            current_node = node_names.popleft()

            # if the flag isn't defined, this will be None
            # by checking False explicitly, it makes the default
            # behaviour to render meshes with no flag defined.
            # if self.node_flag(name_node, 'visible') is False:
            #    continue

            transform, geometry_name = self.scene.graph[current_node]

            if geometry_name is None:
                continue

            mesh = self.scene.geometry[geometry_name]

            if (hasattr(mesh, 'visual') and
                    mesh.visual.transparency):
                # put the current item onto the back of the queue
                if count < count_original:
                    node_names.append(current_node)
                    continue

            # add a new matrix to the model stack
            gl.glPushMatrix()
            # transform by the nodes transform
            gl.glMultMatrixf(_gl_matrix(transform))
            # get the mode of the current geometry
            mode = self.vertex_list_mode[geometry_name]
            # draw the mesh with its transform applied
            self.vertex_list[geometry_name].draw(mode=mode)
            # pop the matrix stack as we drew what we needed to draw
            gl.glPopMatrix()

    def node_flag(self, node, flag):
        if (hasattr(self.scene, 'flags') and
            node in self.scene.flags and
                flag in self.scene.flags[node]):
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
        super(SceneViewer, self).flip()

        if self._img is not None:
            self.save_image(self._img)
            self.close()


def _view_transform(view):
    '''
    Given a dictionary containing view parameters,
    calculate a transformation matrix.
    '''
    transform = view['ball'].matrix()
    transform[0:3, 3] = view['center']
    transform[0:3, 3] -= np.dot(transform[0:3, 0:3], view['center'])
    transform[0:3, 3] += view['translation'] * view['scale'] * 5.0
    return transform


def geometry_md5(geometry):
    md5 = geometry.md5()
    if hasattr(geometry, 'visual'):
        md5 += str(geometry.visual.crc())
    return md5


def mesh_to_vertex_list(mesh, group=None):
    '''
    Convert a Trimesh object to arguments for an
    indexed vertex list constructor.
    '''
    normals = mesh.vertex_normals.reshape(-1).tolist()
    faces = mesh.faces.reshape(-1).tolist()
    vertices = mesh.vertices.reshape(-1).tolist()
    color_gl = _validate_colors(mesh.visual.vertex_colors, len(mesh.vertices))

    args = (len(mesh.vertices),  # number of vertices
            gl.GL_TRIANGLES,    # mode
            group,              # group
            faces,              # indices
            ('v3f/static', vertices),
            ('n3f/static', normals),
            color_gl)
    return args


def path_to_vertex_list(path, group=None):
    vertices = path.vertices
    lines = np.vstack([util.stack_lines(e.discrete(path.vertices))
                       for e in path.entities])
    index = np.arange(len(lines))

    args = (len(lines),         # number of vertices
            gl.GL_LINES,        # mode
            group,              # group
            index.reshape(-1).tolist(),  # indices
            ('v3f/static', lines.reshape(-1)),
            ('c3f/static', np.array([.5, .10, .20] * len(lines))))
    return args


def points_to_vertex_list(points, colors, group=None):
    points = np.asanyarray(points)

    if not util.is_shape(points, (-1, 3)):
        raise ValueError('Pointcloud must be (n,3)!')

    color_gl = _validate_colors(colors, len(points))

    index = np.arange(len(points))

    args = (len(points),         # number of vertices
            gl.GL_POINTS,        # mode
            group,              # group
            index.reshape(-1),  # indices
            ('v3f/static', points.reshape(-1)),
            color_gl)
    return args


def _validate_colors(colors, count):
    '''
    Given a list of colors (or None) return a GL- acceptable list of colors

    Parameters
    ------------
    colors: (count, (3 or 4)) colors

    Returns
    ---------
    colors_type: str, color type
    colors_gl:   list, count length
    '''

    colors = np.asanyarray(colors)
    count = int(count)
    if util.is_shape(colors, (count, (3, 4))):
        # convert the numpy dtype code to an opengl one
        colors_dtype = {'f': 'f',
                        'i': 'B',
                        'u': 'B'}[colors.dtype.kind]
        # create the data type description string pyglet expects
        colors_type = 'c' + str(colors.shape[1]) + colors_dtype + '/static'
        # reshape the 2D array into a 1D one and then convert to a python list
        colors = colors.reshape(-1).tolist()
    else:
        # case where colors are wrong shape, use a default color
        colors = np.tile([.5, .10, .20], (count, 1)).reshape(-1).tolist()
        colors_type = 'c3f/static'

    return colors_type, colors


def _gl_matrix(array):
    '''
    Convert a sane numpy transformation matrix (row major, (4,4))
    to an stupid GLfloat transformation matrix (column major, (16,))
    '''
    a = np.array(array).T.reshape(-1)
    return (gl.GLfloat * len(a))(*a)


def _gl_vector(array, *args):
    '''
    Convert an array and an optional set of args into a flat vector of GLfloat
    '''
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (gl.GLfloat * len(array))(*array)
    return vector
