"""
A basic first-person camera example.

This is ideal for inspecting 3D models/scenes and can be adapted
to your needs for first person games.

* Supports mouse and keyboard input with WASD + QE for up/down.
* Supports controller input for movement and rotation including left and right triggers
  to move up and down,
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees

import numpy as np
import pyglet
from pyglet.gl import GL_CULL_FACE, GL_DEPTH_TEST, glEnable
from pyglet.graphics.shader import ShaderProgram
from pyglet.math import Vec2, Vec3
from pyglet.window import key as _key

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
            models[name] = to_pyglet(geometry, batch=self._batch)

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

    def on_draw(self):
        self.clear()
        self._batch.draw()

    def on_resize(self, width: int, height: int) -> bool:
        """Update the viewport and projection matrix on window resize."""
        self._poseport = (0, 0, *self.get_framebuffer_size())

        self._pose.trackball.resize((width, height))

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
        self.scene.camera_transform = self._pose.trackball.pose

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Zoom the view.
        """
        self._pose.trackball.scroll(dy)
        self.scene.camera_transform = self._pose.trackball.pose

    def on_deactivate(self) -> None:
        """Reset the movement states when the window loses focus."""
        self.controller_look = Vec2()
        self.controller_move = Vec2()

    def teleport(self, position: Vec3, target: Vec3 | None = None) -> None:
        """Teleport the camera to a new position.

        An optional new view target can be provided. If no target is
        provided, the camera will look in the same direction as before.
        """
        if target is not None:
            direction = (target - self.position).normalize()
            pitch, yaw = direction.get_pitch_yaw()
            self.yaw = degrees(yaw)
            self.pitch = degrees(pitch)

        self.position = position

    # --- Mouse input ---

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        """Read the mouse input and update the camera's yaw and pitch."""
        if not self._exclusive_mouse:
            return

        self.mouse_look = Vec2(dx, dy)

    def on_mouse_press(self, x: int, y: int, button, modifiers) -> None:
        """Capture the mouse input when the window is clicked."""
        if not self._exclusive_mouse:
            self._exclusive_mouse = True
            self.set_exclusive_mouse(True)

    # --- Keyboard input ---

    def on_key_press(self, symbol: int, mod: int) -> bool:
        """Handle keyboard input."""

        if symbol in (_key.Q, pyglet.window.key.ESCAPE):
            self._exclusive_mouse = False

            self.set_exclusive_mouse(False)

            self.close()
            pyglet.app.exit()

            return pyglet.event.EVENT_HANDLED

        return False


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


def to_pyglet(
    mesh: Trimesh, batch: pyglet.graphics.Batch | None = None
) -> pyglet.model.Model:
    """
    Convert a Trimesh object into a Pyglet model.


    """
    if batch is None:
        batch = pyglet.graphics.Batch()

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


if __name__ == "__main__":
    m = trimesh.load("models/rabbit.obj")
    m.visual = m.visual.to_color()

    # w = Pyglet2Viewer(geometry=m, position=Vec3(0.0, 0.0, 5.0))
