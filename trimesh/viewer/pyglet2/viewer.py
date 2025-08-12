from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyglet
from pyglet.gl import GL_CULL_FACE, GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glClearColor, glEnable, glBlendFunc, glLineWidth
from pyglet.graphics.shader import ShaderProgram
from pyglet.math import Mat4
from pyglet.model import MaterialGroup, SimpleMaterial

from ...scene import Scene
from ..trackball import Trackball
from .shaders import get_shader_manager
from .conversion import convert_to_pyglet2

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
        background=None,
        point_size=4.0,
        line_width=2.0,
    ) -> None:
        """Initialize the camera."""

        super().__init__(resizable=True)

        # store rendering settings
        self.point_size = point_size
        self.line_width = line_width

        # create the batch all geometry will be added to
        self._batch = pyglet.graphics.Batch()

        # get shader manager
        self._shader_manager = get_shader_manager()

        # assign the scene to this object
        self.set_scene(scene)

        # hold current viewer position and settings
        self._pose = View(
            trackball=Trackball(
                pose=self._initial_camera_transform,
                size=self.scene.camera.resolution,
                scale=self.scene.scale * 3.0,  # Scale up for better interaction
                target=self.scene.centroid,
            ),
        )

        self._exclusive_mouse = False

        # default GL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if background is None:
            background = np.ones(4)
        glClearColor(*background)

        # set a window caption
        self.set_caption("trimesh viewer")

        # initialize axis and grid as None
        self._axis = None
        self._grid = None

        # set point and line width
        glLineWidth(self.line_width)

        # run the initial frame render
        self.on_refresh(0.0)

        pyglet.app.run()

        # run the initial frame render
        self.on_refresh(0.0)

        pyglet.app.run()

    def set_scene(self, scene: Scene):
        """
        Set the current viewer to a trimesh Scene.
        """
        if not isinstance(scene, Scene):
            scene = Scene(scene)

        # Improve camera clipping planes for better depth precision
        scene_scale = scene.scale

        scene.camera.z_near = 0.00001 #-scene_scale * 100
        scene.camera.z_far = scene_scale * 100.0  # Far plane based on scene size

        models = {}
        for name, geometry in scene.geometry.items():
            if geometry.is_empty:
                continue
            models[name] = convert_to_pyglet2(geometry, batch=self._batch)

        self.scene = scene
        self._scale = scene.scale
        self._models = models
        self._initial_camera_transform = scene.camera_transform.copy()

    def on_refresh(self, delta_time: float) -> None:
        """
        Called before the window content is drawn.

        Runs every frame applying the camera movement.
        """
        # Update camera matrices
        view_matrix = np.linalg.inv(self._pose.trackball.pose).astype(np.float32)


        # save the first and last view matrices
        if getattr(self, "_view_initial", None) is None:
            self._view_initial = view_matrix.copy()
        self._view_last = view_matrix.copy()

        if getattr(self, "_projection_initial", None) is None:
            self._projection_initial = self.scene.camera.projection.copy()
        self._projection_last = self.scene.camera.projection.copy()

        # Setup shader uniforms for PBR
        pbr_shader = self._shader_manager.get_shader('pbr')
        pbr_shader.use()
        pbr_shader['view'] = view_matrix.ravel()
        pbr_shader['projection'] = self.scene.camera.projection.astype(np.float32).ravel()
        
        # Set view position for lighting calculations
        inv_view = np.linalg.inv(view_matrix)
        view_pos = inv_view[:3, 3]
        pbr_shader['viewPos'] = view_pos
        
        # Setup lighting from scene lights with proper positions
        light_nodes = [node for node in self.scene.graph.nodes if 'light' in node.lower()]
        
        for i, light in enumerate(self.scene.lights):
            if i >= 4:  # Max 4 lights
                break
            
            # Get the corresponding light node transform
            if i < len(light_nodes):
                light_transform, _ = self.scene.graph.get(light_nodes[i])
                light_pos = light_transform[:3, 3]
            else:
                # Fallback position if no transform found
                light_pos = [2.0, 2.0, 2.0]
            
            # Normalize color from uint8 to float
            color = light.color[:3]  # Take RGB, ignore alpha
            if color.dtype == np.uint8:
                color_normalized = color.astype(np.float32) / float(np.iinfo(color.dtype).max)
            else:
                color_normalized = color.astype(np.float32)
            
            pbr_shader[f'lightPosition{i}'] = light_pos
            pbr_shader[f'lightColor{i}'] = color_normalized
        
        pbr_shader['numLights'] = min(len(self.scene.lights), 4)
        pbr_shader.stop()

    def on_draw(self):
        self.clear()
        self.on_refresh(0.0)  # Update matrices and shaders
        
        # Update model matrices for all geometry
        self._update_model_matrices()
        
        # Draw the batch
        self._batch.draw()
        
        # Draw axis if enabled
        if self._axis is not None and self._pose.axis:
            self._axis.batch.draw()
            
        # Draw grid if enabled
        if self._grid is not None and self._pose.grid:
            self._grid.batch.draw()

    def _update_model_matrices(self):
        """Update model matrices for all rendered objects."""
        identity = np.eye(4, dtype=np.float32)
        
        # Update model matrices for scene geometry
        for name, model in self._models.items():
            # Get transform from scene graph
            node_name = None
            for node in self.scene.graph.nodes_geometry:
                transform, geometry_name = self.scene.graph.get(node)
                if geometry_name == name:
                    node_name = node
                    break
                    
            if node_name is not None:
                transform, _ = self.scene.graph.get(node_name)
                model_matrix = transform.astype(np.float32)
            else:
                model_matrix = identity
                
            # Update shader uniforms for this model
            if hasattr(model, 'groups'):
                for group in model.groups:
                    if hasattr(group, 'shader'):
                        group.shader.use()
                        group.shader['model'] = model_matrix.ravel()
                        group.shader.stop()

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

    def on_close(self):
        """
        Handle the window close event.
        """
        print('Window closed')
        print("View initial", self._view_initial)
        print("View last", self._view_last)

        print("Projection initial",    self._projection_initial)    
        print("Projection last", self._projection_last)

        super().on_close()

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

    # display an axis marker
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



