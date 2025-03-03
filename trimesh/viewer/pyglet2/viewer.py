from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyglet
from pyglet.gl import GL_CULL_FACE, GL_DEPTH_TEST, glEnable
from pyglet.graphics.shader import ShaderProgram
from pyglet.math import Mat4
from pyglet.model import MaterialGroup, SimpleMaterial

from ...scene import Scene
from ..trackball import Trackball

# the axis marker can be in several states
_AXIS_STATES = [None, "world", "all", "without_world"]
_AXIS_TYPE = Literal[None, "world", "all", "without_world"]


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
        """
        Set the current viewer to a trimesh Scene.
        """
        if not isinstance(scene, Scene):
            scene = Scene(scene)

        models = {}
        for name, geometry in scene.geometry.items():
            models[name] = mesh_to_pyglet(geometry, batch=self._batch)

        self.scene = scene
        self._scale = scene.scale
        self._models = models
        self._initial_camera_transform = scene.camera_transform.copy()

    def on_refresh(self, delta_time: float) -> None:
        """
        Called before the window content is drawn.

        Runs every frame applying the camera movement.
        """
        # todo : is there a better way of altering this view in-place?
        self.view = Mat4(*np.linalg.inv(self._pose.trackball.pose).T.ravel())

    def on_draw(self):
        self.clear()
        self._batch.draw()

    def on_resize(self, width: int, height: int) -> bool:
        """
        Update the viewport and projection matrix on window resize.
        """

        # `width` and `height` are the new dimensions of the window
        # where `actual` is the actual size of the framebuffer in pixels
        actual = self.get_framebuffer_size()
        self.viewport = (0, 0, *actual)

        self.scene.camera.resolution = actual
        self._pose.trackball.resize(actual)

        self.projection = Mat4(*self.scene.camera.projection.T.ravel())

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

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Zoom the view.
        """
        self._pose.trackball.scroll(dy)

    def toggle_wireframe(self):
        """
        Toggle the wireframe mode.
        """
        self._pose.wireframe = not self._pose.wireframe
        self._update_flags()

    def reset_view(self):
        """
        Reset the view to the initial camera transform.
        """
        raise NotImplementedError()

    def toggle_culling(self):
        """
        Toggle backface culling.
        """
        self._pose.cull = not self._pose.cull
        self._update_flags()

    def toggle_axis(self):
        """
        Toggle a rendered XYZ/RGB axis marker:
        off, world frame, every frame
        """
        # the state after toggling
        index = (_AXIS_STATES.index(self._pose["axis"]) + 1) % len(_AXIS_STATES)
        # update state to next index
        self._pose.axis = _AXIS_STATES[index]
        # perform gl actions
        self._update_flags()

    def toggle_grid(self):
        """
        Toggle a rendered grid.
        """
        # update state to next index
        self._pose.grid = not self._pose.grid
        # perform gl actions
        self._update_flags()

    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """

        actions = {
            pyglet.window.key.W: self.toggle_wireframe,
            pyglet.window.key.Z: self.reset_view,
            pyglet.window.key.C: self.toggle_culling,
            pyglet.window.key.A: self.toggle_axis,
            pyglet.window.key.G: self.toggle_grid,
            pyglet.window.key.Q: self.on_close,
            pyglet.window.key.M: self.maximize,
            pyglet.window.key.F: self.toggle_fullscreen,
        }

        if symbol in actions:
            actions[symbol]()

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

    def toggle_fullscreen(self):
        """
        Toggle the window between fullscreen and windowed mode.
        """
        self._pose.fullscreen = not self._pose.fullscreen
        self._update_flags()

    def _update_flags(self):
        """
        Check the view flags, and call required GL functions.
        """
        # view mode, filled vs wirefrom
        # if self._pose.wireframe:
        #    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        # else:
        #    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # set fullscreen or windowed
        self.set_fullscreen(fullscreen=self._pose.fullscreen)

        """
        # backface culling on or off
        if self.view.cull:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        # case where we WANT an axis and NO vertexlist
        # is stored internally
        if self.view["axis"] and self._axis is None:
            from ... import creation

            # create an axis marker sized relative to the scene
            axis = creation.axis(origin_size=self.scene.scale / 100)
            # create ordered args for a vertex list
            args = conversion.mesh_to_vertexlist(axis)
            # store the axis as a reference
            self._axis = self.batch.add_indexed(*args)
        # case where we DON'T want an axis but a vertexlist
        # IS stored internally
        elif not self.view["axis"] and self._axis is not None:
            # remove the axis from the rendering batch
            self._axis.delete()
            # set the reference to None
            self._axis = None

        if self.view["grid"] and self._grid is None:
            try:
                # create a grid marker
                from ...path.creation import grid

                bounds = self.scene.bounds
                center = bounds.mean(axis=0)
                # set the grid to the lowest Z position
                # also offset by the scale to avoid interference
                center[2] = bounds[0][2] - (np.ptp(bounds[:, 2]) / 100)
                # choose the side length by maximum XY length
                side = np.ptp(bounds, axis=0)[:2].max()
                # create an axis marker sized relative to the scene
                grid_mesh = grid(side=side, count=4, transform=translation_matrix(center))
                # convert the path to vertexlist args
                args = conversion.convert_to_vertexlist(grid_mesh)
                # create ordered args for a vertex list
                self._grid = self.batch.add_indexed(*args)
            except BaseException:
                util.log.warning("failed to create grid!", exc_info=True)
        elif not self.view["grid"] and self._grid is not None:
            self._grid.delete()
            self._grid = None

        """


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

    axis: _AXIS_TYPE = None

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
    mesh,
    batch: pyglet.graphics.Batch | None = None,
    group=None,
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

    diffuse = [1.0, 1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 1.0, 1.0]
    specular = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 100.0

    default_material = SimpleMaterial(
        "Default", diffuse, ambient, specular, emission, shininess
    )

    program: ShaderProgram = pyglet.gl.current_context.create_program(
        (pyglet.model.MaterialGroup.default_vert_src, "vertex"),
        (pyglet.model.MaterialGroup.default_frag_src, "fragment"),
    )

    matgroup = MaterialGroup(default_material, program, order=0, parent=group)

    idx = program.vertex_list_indexed(
        len(mesh.vertices),
        pyglet.gl.GL_TRIANGLES,
        mesh.faces.ravel(),
        batch=batch,
        group=matgroup,
        POSITION=("f", mesh.vertices.ravel()),
        NORMAL=("f", mesh.vertex_normals.ravel()),
        COLOR_0=("f", np.ones(len(mesh.vertices) * 4)),
    )

    return pyglet.model.Model([idx], [matgroup], batch)
