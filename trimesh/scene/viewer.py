import pyglet
import pyglet.gl as gl

import numpy as np

import tempfile
import platform
import collections

from .. import rendering
from ..transformations import Arcball

# smooth only when fewer faces than this
_SMOOTH_MAX_FACES = 100000


class SceneViewer(pyglet.window.Window):

    def __init__(self,
                 scene,
                 smooth=True,
                 flags=None,
                 visible=True,
                 resolution=(640, 480),
                 start_loop=True,
                 **kwargs):

        self.scene = scene
        self.scene._redraw = self._redraw
        self.reset_view(flags=flags)
        self.batch = pyglet.graphics.Batch()
        self._smooth = smooth

        self.vertex_list = {}
        self.vertex_list_hash = {}
        self.vertex_list_mode = {}

        try:
            # try enabling antialiasing
            # if you have a graphics card this will probably work
            conf = gl.Config(sample_buffers=1,
                             samples=4,
                             depth_size=24,
                             double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              visible=visible,
                                              resizable=True,
                                              width=resolution[0],
                                              height=resolution[1])
        except pyglet.window.NoSuchConfigException:
            conf = gl.Config(double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              resizable=True,
                                              visible=visible,
                                              width=resolution[0],
                                              height=resolution[1])

        if 'camera' not in scene.graph:
            # if the camera hasn't been set, set it now
            scene.set_camera()

        for name, mesh in scene.geometry.items():
            self.add_geometry(name=name,
                              geometry=mesh)
        self.init_gl()
        self.set_size(*resolution)
        self.update_flags()

        if start_loop:
            pyglet.app.run()

    def _redraw(self):
        self.on_draw()

    def _update_meshes(self):
        for name, mesh in self.scene.geometry.items():
            if self.vertex_list_hash[name] != geometry_hash(mesh):
                self.add_geometry(name, mesh)

    def add_geometry(self, name, geometry, **kwargs):
        # convert geometry to constructor args
        args = rendering.convert_to_vertexlist(geometry, **kwargs)
        # create the indexed vertex list
        self.vertex_list[name] = self.batch.add_indexed(*args)
        # save the MD5 of the geometry
        self.vertex_list_hash[name] = geometry_hash(geometry)
        # save the rendering mode from the constructor args
        self.vertex_list_mode[name] = args[1]

    def reset_view(self, flags=None):
        """
        Set view to base view.
        """
        self.view = {'wireframe': False,
                     'cull': True,
                     'translation': np.zeros(3),
                     'center': self.scene.centroid,
                     'scale': self.scene.scale,
                     'ball': Arcball()}

        try:
            self.view['ball'].place([self.width / 2.0,
                                     self.height / 2.0],
                                    (self.width + self.height) / 2.0)
            if isinstance(flags, dict):
                for k, v in flags.items():
                    if k in self.view:
                        self.view[k] = v
                self.update_flags()
        except BaseException:
            pass

    def init_gl(self):
        gl.glClearColor(.97, .97, .97, 1.0)
        max_depth = (np.abs(self.scene.bounds).max(axis=1) ** 2).sum() ** .5
        max_depth = np.clip(max_depth, 500.00, np.inf)
        gl.glDepthRange(0.0, max_depth)

        gl.glClearDepth(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)

        # put the light at one corner of the scenes AABB
        gl.glLightfv(gl.GL_LIGHT0,
                     gl.GL_POSITION,
                     rendering.vector_to_gl(np.append(self.scene.bounds[1], 0)))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR,
                     rendering.vector_to_gl(.5, .5, 1, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                     rendering.vector_to_gl(1, 1, 1, .75))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                     rendering.vector_to_gl(.1, .1, .1, .2))

        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMaterialfv(gl.GL_FRONT,
                        gl.GL_AMBIENT,
                        rendering.vector_to_gl(0.192250, 0.192250, 0.192250))
        gl.glMaterialfv(gl.GL_FRONT,
                        gl.GL_DIFFUSE,
                        rendering.vector_to_gl(0.507540, 0.507540, 0.507540))
        gl.glMaterialfv(gl.GL_FRONT,
                        gl.GL_SPECULAR,
                        rendering.vector_to_gl(.5082730, .5082730, .5082730))

        gl.glMaterialf(gl.GL_FRONT,
                       gl.GL_SHININESS,
                       .4 * 128.0)

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
        try:
            # for high DPI screens viewport size
            # will be different then the passed size
            width, height = self.get_viewport_size()
        except BaseException:
            # older versions of pyglet may not have this
            pass
        # set the new viewport size
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(60.,
                          width / float(height),
                          .01,
                          self.scene.scale * 5.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        self.view['ball'].place([width / 2, height / 2],
                                (width + height) / 2)

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
        magnitude = 10
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()
        elif symbol == pyglet.window.key.LEFT:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([-magnitude, 0])
        elif symbol == pyglet.window.key.RIGHT:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([magnitude, 0])
        elif symbol == pyglet.window.key.DOWN:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([0, -magnitude])
        elif symbol == pyglet.window.key.UP:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([0, magnitude])

    def on_draw(self):
        self._update_meshes()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # pull the new camera transform from the scene
        transform_camera, _junk = self.scene.graph['camera']

        # apply the camera transform to the matrix stack
        gl.glMultMatrixf(rendering.matrix_to_gl(transform_camera))

        # dragging the mouse moves the view transform (but doesn't alter the
        # scene)
        transform_view = view_to_transform(self.view)
        gl.glMultMatrixf(rendering.matrix_to_gl(transform_view))

        # we want to render fully opaque objects first,
        # followed by objects which have transparency
        node_names = collections.deque(self.scene.graph.nodes_geometry)
        count_original = len(node_names)
        count = -1

        while len(node_names) > 0:
            count += 1
            current_node = node_names.popleft()

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
            gl.glMultMatrixf(rendering.matrix_to_gl(transform))
            # get the mode of the current geometry
            mode = self.vertex_list_mode[geometry_name]
            # draw the mesh with its transform applied
            self.vertex_list[geometry_name].draw(mode=mode)
            # pop the matrix stack as we drew what we needed to draw
            gl.glPopMatrix()

    def save_image(self, file_obj):
        """
        Save the current color buffer to a file object, in PNG format.

        Parameters
        -------------
        file_obj: file name, or file- like object
        """
        colorbuffer = pyglet.image.get_buffer_manager().get_color_buffer()
        if hasattr(file_obj, 'write'):
            colorbuffer.save(file=file_obj)
        else:
            colorbuffer.save(filename=file_obj)


def view_to_transform(view):
    """
    Given a dictionary containing view parameters,
    calculate a transformation matrix.
    """
    transform = view['ball'].matrix()
    transform[0:3, 3] = view['center']
    transform[0:3, 3] -= np.dot(transform[0:3, 0:3], view['center'])
    transform[0:3, 3] += view['translation'] * view['scale'] * 5.0
    return transform


def geometry_hash(geometry):
    """
    Get an MD5 for a geometry object

    Parameters
    ------------
    geometry : object

    Returns
    ------------
    MD5 : str
    """
    if hasattr(geometry, 'md5'):
        # for most of our trimesh objects
        md5 = geometry.md5()
    elif hasattr(geometry, 'tostring'):
        # for unwrapped ndarray objects
        md5 = str(hash(geometry.tostring()))

    if hasattr(geometry, 'visual'):
        # if visual properties are defined
        md5 += str(geometry.visual.crc())
    return md5


def render_scene(scene, resolution=(1080, 1080), visible=True, **kwargs):
    """
    Render a preview of a scene to a PNG.

    Parameters
    ------------
    scene:      trimesh.Scene object
    resolution: (2,) int, resolution in pixels
    kwargs:     passed to SceneViewer

    Returns
    ---------
    render: bytes, image in PNG format
    """
    window = SceneViewer(scene,
                         start_loop=False,
                         visible=visible,
                         resolution=resolution,
                         **kwargs)

    if visible is None:
        visible = platform.system() != 'Linux'

    # need to run loop twice to display anything
    for i in range(2):
        pyglet.clock.tick()
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()

    with tempfile.TemporaryFile() as file_obj:
        window.save_image(file_obj)
        file_obj.seek(0)
        render = file_obj.read()
    window.close()

    return render
