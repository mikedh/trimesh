"""
trimesh.viewer.pyglet2.window
-----------------------------

Pyglet 2.x debug viewer for `trimesh.Scene`.
"""

from __future__ import annotations

import numpy as np
import pyglet
from pyglet.gl import (
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClearColor,
    glDisable,
    glEnable,
    glPolygonMode,
    glViewport,
)
from pyglet.model import MaterialGroup, Model, TexturedMaterialGroup

from ... import util
from ...scene import Scene
from ...visual import to_rgba
from ..trackball import Trackball
from .conversion import Programs, matrix_to_pyglet, to_model
from .helper import (
    FRAG_SRC,
    HIDDEN_MATRIX,
    KEYS,
    TEXTURED_FRAG_SRC,
    ViewerState,
    build_help,
    drop_model,
    ensure_decoration,
)


class SceneViewer:
    """
    A 3D trackball viewer for `trimesh.Scene` objects.

    Holds the underlying `pyglet.window.Window` on `self.window`. Reach for
    pyglet-specific things there (e.g. `viewer.window.set_visible(False)`);
    everything trimesh-shaped lives on `self`.
    """

    def __init__(
        self,
        scene,
        background=None,
        resolution=None,
        fullscreen: bool = False,
        resizable: bool = True,
        visible: bool = True,
        start_loop: bool = True,
        callback=None,
        callback_period=None,
        caption: str | None = None,
        flags: dict | None = None,
        fixed=None,
        config=None,
        window_conf=None,
    ):
        if not isinstance(scene, Scene):
            scene = Scene(scene)
        if resolution is None:
            resolution = scene.camera.resolution
        else:
            scene.camera.resolution = resolution
        self.scene = scene
        self.callback = callback

        # `window_conf` is the legacy pyglet1 name; accept both.
        self.window = pyglet.window.Window(
            width=int(resolution[0]),
            height=int(resolution[1]),
            caption=caption or "trimesh SceneViewer (`h` for help)",
            resizable=resizable,
            fullscreen=fullscreen,
            visible=visible,
            config=config or window_conf,
        )
        # wire pyglet events to our public `on_*` methods. method names
        # must match pyglet's event names.
        self.window.push_handlers(
            on_draw=self.on_draw,
            on_resize=self.on_resize,
            on_mouse_press=self.on_mouse_press,
            on_mouse_drag=self.on_mouse_drag,
            on_mouse_scroll=self.on_mouse_scroll,
            on_key_press=self.on_key_press,
        )

        self._initial_camera = scene.camera_transform.copy()

        # one untextured + one textured shader program, shared across every
        # Model. Vertex shaders are pyglet's stock; fragment shaders are
        # local copies that pass the input alpha through unchanged so blend
        # doesn't make opaque meshes see-through.
        ctx = pyglet.gl.current_context
        self._programs = Programs(
            plain=ctx.create_program(
                (MaterialGroup.default_vert_src, "vertex"),
                (FRAG_SRC, "fragment"),
            ),
            textured=ctx.create_program(
                (TexturedMaterialGroup.default_vert_src, "vertex"),
                (TEXTURED_FRAG_SRC, "fragment"),
            ),
        )
        self._batch = pyglet.graphics.Batch()
        # one Model per scene-graph node, keyed by node name.
        self._models: dict[str, Model] = {}
        self._geometry_id: dict[str, tuple] = {}
        self._axis_model: Model | None = None
        self._grid_model: Model | None = None

        self.state = ViewerState(
            trackball=Trackball(
                pose=self._initial_camera,
                # mouse events are in window (logical) pixels.
                size=self.window.get_size(),
                scale=scene.scale,
                target=scene.centroid,
            ),
            fixed=set(fixed) if fixed else set(),
        )

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        bg = (
            np.ones(4, dtype=np.float32)
            if background is None
            else np.asarray(to_rgba(background), dtype=np.float32) / 255.0
        )
        glClearColor(*bg)

        self.rebuild_models()
        if flags:
            for k, v in flags.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)
        self.apply_flags()

        if callback is not None:

            def _on_tick(dt):
                callback(self.scene)
                self.rebuild_models()

            pyglet.clock.schedule_interval(
                _on_tick,
                1.0 / 30.0 if callback_period is None else callback_period,
            )

        if start_loop:
            pyglet.app.run()

    # --- public API ---------------------------------------------------

    def close(self) -> None:
        self.window.close()

    def maximize(self) -> None:
        self.window.maximize()

    def save_image(self, file_obj):
        """Save the current color buffer as a PNG to a path or file-like."""
        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        if hasattr(file_obj, "write"):
            buf.save(file=file_obj)
        else:
            buf.save(filename=file_obj)
        return file_obj

    def hide_geometry(self, name: str) -> None:
        self.state.hidden.add(name)

    def unhide_geometry(self, name: str) -> None:
        self.state.hidden.discard(name)

    def reset_view(self) -> None:
        self.state.trackball = Trackball(
            pose=self._initial_camera,
            size=self.window.get_size(),
            scale=self.scene.scale,
            target=self.scene.centroid,
        )
        self.scene.camera_transform = self._initial_camera.copy()

    def toggle_wireframe(self) -> None:
        self.state.wireframe = not self.state.wireframe
        self.apply_flags()

    def toggle_culling(self) -> None:
        self.state.cull = not self.state.cull
        self.apply_flags()

    def toggle_fullscreen(self) -> None:
        self.state.fullscreen = not self.state.fullscreen
        self.apply_flags()

    def toggle_axis(self) -> None:
        self.state.axis = "world" if self.state.axis is None else None
        self.apply_flags()

    def toggle_grid(self) -> None:
        self.state.grid = not self.state.grid
        self.apply_flags()

    def print_help(self) -> None:
        print(build_help())  # noqa: T201

    # --- pyglet event handlers ---------------------------------------

    def on_draw(self) -> None:
        # The scene is the source of truth: input updates
        # `scene.camera_transform`, and we re-read it here so any external
        # call to `Scene.camera_project` agrees with the rendered frame.
        # `view`/`projection` setters are safe to use: they just write
        # into the `WindowBlock` UBO that pyglet's stock 3D shaders read.
        # Unlike `viewport`, which secretly multiplies by `dpi/96`.
        cam_inv = np.linalg.inv(self.scene.camera_transform)
        self.window.view = matrix_to_pyglet(cam_inv)
        self.window.projection = matrix_to_pyglet(self.scene.camera.projection)
        self.apply_transforms(cam_inv)
        self.window.clear()
        self._batch.draw()

    def on_resize(self, width: int, height: int) -> None:
        # We bypass `window.viewport = ...` because its setter multiplies
        # values by `dpi/96`, which is correct only in
        # `dpi_scaling="stretch"` mode. In default mode the framebuffer is
        # unscaled, so the multiplied glViewport overshoots and the scene
        # falls into the top-right corner.
        fb = self.window.get_framebuffer_size()
        glViewport(0, 0, *fb)
        self.scene.camera.resolution = fb
        # mouse events are in window (logical) pixels.
        self.state.trackball.resize((width, height))

    def on_mouse_press(self, x, y, buttons, modifiers) -> None:
        ball = self.state.trackball
        ctrl = modifiers & pyglet.window.key.MOD_CTRL
        shift = modifiers & pyglet.window.key.MOD_SHIFT
        if buttons == pyglet.window.mouse.LEFT:
            if ctrl and shift:
                ball.set_state(Trackball.STATE_ZOOM)
            elif shift:
                ball.set_state(Trackball.STATE_ROLL)
            elif ctrl:
                ball.set_state(Trackball.STATE_PAN)
            else:
                ball.set_state(Trackball.STATE_ROTATE)
        elif buttons == pyglet.window.mouse.MIDDLE:
            ball.set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.RIGHT:
            ball.set_state(Trackball.STATE_ZOOM)
        ball.down(np.array([x, y]))
        self.scene.camera_transform = ball.pose

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        self.state.trackball.drag(np.array([x, y]))
        self.scene.camera_transform = self.state.trackball.pose

    def on_mouse_scroll(self, x, y, dx, dy) -> None:
        self.state.trackball.scroll(dy)
        self.scene.camera_transform = self.state.trackball.pose

    def on_key_press(self, symbol, modifiers) -> None:
        key = pyglet.window.key
        for letter, _, method in KEYS:
            if symbol == getattr(key, letter.upper()):
                getattr(self, method)()
                return
        if symbol in (key.LEFT, key.RIGHT, key.DOWN, key.UP):
            ball = self.state.trackball
            arrows = {
                key.LEFT: [-10, 0],
                key.RIGHT: [10, 0],
                key.DOWN: [0, -10],
                key.UP: [0, 10],
            }
            ball.down([0, 0])
            ball.drag(arrows[symbol])
            self.scene.camera_transform = ball.pose

    # --- model bookkeeping -------------------------------------------

    def rebuild_models(self) -> None:
        """Sync `self._models` with the current scene graph: build new
        nodes, drop nodes that disappeared, replace nodes whose geometry
        changed."""
        graph = self.scene.graph
        live = set(graph.nodes_geometry)
        for node in live:
            _, geom_name = graph.get(node)
            geometry = self.scene.geometry.get(geom_name)
            if geometry is None:
                continue
            current_id = (
                geom_name,
                hash(geometry),
                hash(getattr(geometry, "visual", None)),
            )
            if self._geometry_id.get(node) == current_id:
                continue
            drop_model(self._models.pop(node, None))
            self._geometry_id.pop(node, None)
            try:
                model = to_model(geometry, self._programs, self._batch)
            except BaseException:
                util.log.warning(f"failed to add node `{node}`", exc_info=True)
                continue
            if model is None:
                continue
            self._models[node] = model
            self._geometry_id[node] = current_id
        for node in list(self._models):
            if node not in live:
                drop_model(self._models.pop(node, None))
                self._geometry_id.pop(node, None)

    def apply_transforms(self, cam_inv=None) -> None:
        """Push every Model's transform from the scene graph to the GPU.
        Called every frame from `on_draw`. `cam_inv` is the inverse of the
        current camera transform; passing it from `on_draw` avoids recomputing
        it. `fixed_anchor` is the per-frame constant that anchors `fixed`
        geometry to the initial camera; computed lazily."""
        graph = self.scene.graph
        state = self.state
        fixed_anchor = None
        for node in graph.nodes_geometry:
            model = self._models.get(node)
            if model is None:
                continue
            transform, geom_name = graph.get(node)
            if geom_name in state.hidden:
                model.matrix = HIDDEN_MATRIX
                continue
            if geom_name in state.fixed:
                if fixed_anchor is None:
                    if cam_inv is None:
                        cam_inv = np.linalg.inv(self.scene.camera_transform)
                    fixed_anchor = np.linalg.inv(self._initial_camera @ cam_inv)
                transform = transform @ fixed_anchor
            model.matrix = matrix_to_pyglet(transform)

    def apply_flags(self) -> None:
        """Apply UI state to GL + the window + decoration models."""
        state = self.state
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if state.wireframe else GL_FILL)
        (glEnable if state.cull else glDisable)(GL_CULL_FACE)
        self.window.set_fullscreen(fullscreen=state.fullscreen)

        def _make_axis():
            from ... import creation

            return to_model(
                creation.axis(origin_size=self.scene.scale / 100),
                self._programs,
                self._batch,
            )

        def _make_grid():
            from ...path.creation import grid
            from ...transformations import translation_matrix

            bounds = self.scene.bounds
            center = bounds.mean(axis=0)
            center[2] = bounds[0][2] - (np.ptp(bounds[:, 2]) / 100)
            side = np.ptp(bounds, axis=0)[:2].max()
            return to_model(
                grid(side=side, count=4, transform=translation_matrix(center)),
                self._programs,
                self._batch,
            )

        self._axis_model = ensure_decoration(self._axis_model, state.axis, _make_axis)
        self._grid_model = ensure_decoration(self._grid_model, state.grid, _make_grid)


def render_scene(
    scene,
    resolution=None,
    visible=True,
    fullscreen=False,
    resizable=True,
    **kwargs,
) -> bytes:
    """
    Render a preview of a scene to PNG bytes.

    Whether this works at all is platform and driver dependent. Many
    desktop environments refuse to render hidden windows.
    """
    viewer = SceneViewer(
        scene,
        start_loop=False,
        visible=visible,
        resolution=resolution,
        fullscreen=fullscreen,
        resizable=resizable,
        **kwargs,
    )
    # Pump three frames before sampling: pyglet defers work to the first
    # draw (shader compile, batch upload), and double-buffering means the
    # first flip puts our content on the back buffer, so we need at least
    # two flips before `get_color_buffer` will see anything.
    for _ in range(3):
        pyglet.clock.tick()
        viewer.window.switch_to()
        viewer.window.dispatch_events()
        viewer.on_draw()
        viewer.window.flip()
    buf = util.BytesIO()
    viewer.save_image(buf)
    viewer.close()
    buf.seek(0)
    return buf.read()
