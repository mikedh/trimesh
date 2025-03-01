from dataclasses import dataclass

import numpy as np
import pyglet
from pyglet.gl import GL_CULL_FACE, GL_DEPTH_TEST, glEnable
from pyglet.graphics.shader import ShaderProgram
from pyglet.math import Mat4

from ...scene import Scene
from ..trackball import Trackball


def render_scene(*args, **kwargs):
    raise NotImplementedError()


class SceneViewer(pyglet.window.Window):
    """
    A 3D trackball viewer to debug `trimesh.Scene` objects.
    """

    def __init__(
        self,
        scene: Scene,
    ) -> None:
        """Initialize the camera."""

        super().__init__(resizable=True)

        # create the batch all geometry will be added to
        self._batch = pyglet.graphics.Batch()

        # assign the scene to this object
        self.set_scene(scene)

        # hold current viewer position and settings
        self._pose = View(
            trackball=Trackball(
                pose=self._initial_camera_transform,
                size=self.scene.camera.resolution,
                scale=self.scene.scale,
                target=self.scene.centroid,
            ),
        )

        self._exclusive_mouse = False

        # default GL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # run the initial frame render
        self.on_refresh(0.0)

        pyglet.app.run()

    def set_scene(self, scene: Scene):
        if not isinstance(scene, Scene):
            scene = Scene(scene)

        models = {}
        for name, geometry in scene.geometry.items():
            models[name] = mesh_to_pyglet(geometry, batch=self._batch)

        self.scene = scene
        self._initial_camera_transform = scene.camera_transform.copy()

    def on_refresh(self, delta_time: float) -> None:
        """
        Called before the window content is drawn.

        Runs every frame applying the camera movement.
        """

        # self.position += (translation + up * self._elevation) * walk_speed

        # Look forward from the new position
        # self._pose = Mat4.look_at(self.position, self.position + forward, self.UP)

        # todo : is there a better way of altering this view in-place?

    def on_draw(self):
        self.view = Mat4(*self._pose.trackball.pose.ravel())
        self.clear()
        self._batch.draw()

        print("projection\n", np.array(self.projection).reshape((4, 4)))
        print("view\n", np.array(self.view).reshape((4, 4)))

    def on_resize(self, width: int, height: int) -> bool:
        """Update the viewport and projection matrix on window resize."""

        # `width` and `height` are the new dimensions of the window
        # where `actual` is the actual size of the framebuffer in pixels
        actual = self.get_framebuffer_size()
        self.viewport = (0, 0, *actual)

        actual = np.array(actual, dtype=np.float64)
        actual *= 2.0 / actual.max()
        self.scene.camera.resolution = actual
        self._pose.trackball.resize(actual)

        self.projection = Mat4(*self.scene.camera.K.ravel())

        return pyglet.event.EVENT_HANDLED

    def on_mouse_press(self, x, y, buttons, modifiers):
        """
        Set the start point of the drag.
        """
        self._pose.trackball.set_state(Trackball.STATE_ROTATE)
        if buttons == pyglet.window.mouse.LEFT:
            ctrl = modifiers & pyglet.window.key.MOD_CTRL
            shift = modifiers & pyglet.window.key.MOD_SHIFT
            if ctrl and shift:
                self._pose.trackball.set_state(Trackball.STATE_ZOOM)
            elif shift:
                self._pose.trackball.set_state(Trackball.STATE_ROLL)
            elif ctrl:
                self._pose.trackball.set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.MIDDLE:
            self._pose.trackball.set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.RIGHT:
            self._pose.trackball.set_state(Trackball.STATE_ZOOM)

        self._pose.trackball.down(np.array([x, y]))
        self.scene.camera_transform = self._pose.trackball.pose

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Pan or rotate the view.
        """
        self._pose.trackball.drag(np.array([x, y]))
        # self.scene.camera_transform = self._pose.trackball.pose

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Zoom the view.
        """
        self._pose.trackball.scroll(dy)
        # self.scene.camera_transform = self._pose.trackball.pose

    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()
        elif symbol == pyglet.window.key.A:
            self.toggle_axis()
        elif symbol == pyglet.window.key.G:
            self.toggle_grid()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.M:
            self.maximize()
        elif symbol == pyglet.window.key.F:
            self.toggle_fullscreen()

        if symbol in [
            pyglet.window.key.LEFT,
            pyglet.window.key.RIGHT,
            pyglet.window.key.DOWN,
            pyglet.window.key.UP,
        ]:
            magnitude = 10
            self._pose.trackball.down([0, 0])
            if symbol == pyglet.window.key.LEFT:
                self._pose.trackball.drag([-magnitude, 0])
            elif symbol == pyglet.window.key.RIGHT:
                self._pose.trackball.drag([magnitude, 0])
            elif symbol == pyglet.window.key.DOWN:
                self._pose.trackball.drag([0, -magnitude])
            elif symbol == pyglet.window.key.UP:
                self._pose.trackball.drag([0, magnitude])
            self.scene.camera_transform = self._pose.trackball.pose


def get_default_shader() -> ShaderProgram:
    return pyglet.gl.current_context.create_program(
        (pyglet.model.MaterialGroup.default_vert_src, "vertex"),
        (pyglet.model.MaterialGroup.default_frag_src, "fragment"),
    )


@dataclass
class View:
    # keep the pose of a trackball
    trackball: Trackball

    # enable backface culling
    cull: bool = True

    # display a grid
    grid: bool = False

    # enable fullscreen mode
    fullscreen: bool = False

    # display meshes as a wireframe
    wireframe: bool = False

    def __hash__(self) -> int:
        return hash(
            (
                self.cull,
                self.grid,
                self.fullscreen,
                self.wireframe,
                self.trackball.pose.tobytes(),
            )
        )


def mesh_to_pyglet(
    mesh, batch: pyglet.graphics.Batch | None = None
) -> pyglet.model.Model:
    """
    Convert a Trimesh object into a Pyglet model.

    Parameters
    ------------
    mesh
      The Trimesh object to convert.
    batch
      The Pyglet batch to add the model to.

    Returns
    ------------
    model
      The Pyglet model reference.
    """
    if batch is None:
        batch = pyglet.graphics.Batch()

    # todo : probably should be vendored in the future
    program = get_default_shader()

    idx = program.vertex_list_indexed(
        len(mesh.vertices),
        pyglet.gl.GL_TRIANGLES,
        mesh.faces.ravel(),
        batch=batch,
        group=None,
        POSITION=("f", mesh.vertices.ravel()),
        NORMAL=("f", mesh.vertex_normals.ravel()),
        COLOR_0=("f", np.ones(len(mesh.vertices) * 4)),
    )

    return pyglet.model.Model([idx], [], batch)
