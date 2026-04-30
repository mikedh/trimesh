"""
trimesh.viewer.pyglet2
----------------------

Pyglet 2.x debug viewer for `trimesh.Scene`.

Composition over inheritance: `SceneViewer` *holds* a
`pyglet.window.Window` (`self.window`) rather than subclassing it.
pyglet 2 reserves a long list of attributes on its window class
(`_view`, `_viewport`, `view`, `projection`, `dpi`, `scale`, ...) and
silently breaks if a subclass shadows them. That's how this viewer ended
up rendering into the corner of the framebuffer before composition. The
only external requirement is "a pyglet window with a GL 3.3+ context";
if pyglet 3 reshuffles its internals, only this module's construction
site needs to follow.

The default 3D shaders ship with `pyglet.model.MaterialGroup`: they
provide a `WindowBlock { projection, view }` UBO, a per-object
`uniform mat4 model`, and Lambert lighting. Each scene-graph node gets
its own `pyglet.model.Model`; for instanced geometry that duplicates
vertex data per node, which a debug viewer can afford. The alternative
is hundreds of lines of custom shaders + raw GL because pyglet 2.1's
`vertex_list_instanced[_indexed]` silently mis-routes per-instance
attribute data into the per-vertex region. Revisit when that's fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyglet
from pyglet.gl import (
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINE,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TRIANGLES,
    glActiveTexture,
    glBindTexture,
    glBlendFunc,
    glClearColor,
    glDisable,
    glEnable,
    glPolygonMode,
    glViewport,
)
from pyglet.math import Mat4
from pyglet.model import BaseMaterialGroup, MaterialGroup, Model, TexturedMaterialGroup

from ... import util
from ...scene import Scene
from ...typed import NDArray, float64
from ...visual import to_rgba
from ..trackball import Trackball

# (input, action). Mouse / drag bindings handled directly by `on_mouse_*`.
_MOUSE: tuple[tuple[str, str], ...] = (
    ("click + drag", "rotate"),
    ("ctrl + drag", "pan"),
    ("shift + drag", "roll"),
    ("wheel", "zoom"),
    ("arrows", "nudge"),
)

# (key, action, SceneViewer method). Single source of truth for both
# `on_key_press` dispatch and the help message.
_KEYS: tuple[tuple[str, str, str], ...] = (
    ("w", "toggle wireframe", "toggle_wireframe"),
    ("z", "reset view", "reset_view"),
    ("c", "toggle backface culling", "toggle_culling"),
    ("a", "toggle axis", "toggle_axis"),
    ("g", "toggle grid", "toggle_grid"),
    ("f", "toggle fullscreen", "toggle_fullscreen"),
    ("m", "maximize", "maximize"),
    ("q", "quit", "close"),
    ("h", "help", "print_help"),
)


def _build_help() -> str:
    """Render the help text. Column width is derived from the data so
    adding a binding never breaks alignment."""
    rows = (*_MOUSE, *((k, action) for k, action, _ in _KEYS))
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        (
            "trimesh SceneViewer",
            "",
            *(f"  {label:<{width}}  {action}" for label, action in rows),
        )
    )


# --- texture helpers --------------------------------------------------


def to_texture(material, upsize: bool = True):
    """
    Convert a `trimesh.visual.material.Material` to a `pyglet.image.Texture`,
    or `None` if the material has no usable image. The caller must hold an
    active GL context.
    """
    image = getattr(material, "image", None)
    if image is None:
        image = getattr(material, "baseColorTexture", None)
    if image is None:
        return None
    if upsize:
        try:
            from ...visual.texture import power_resize

            image = power_resize(image)
        except BaseException:
            util.log.warning("texture power_resize failed", exc_info=True)
    with util.BytesIO() as buffer:
        image.save(buffer, format="png")
        buffer.seek(0)
        return pyglet.image.load(filename=".png", file=buffer).get_texture()


def _texture_for(visual):
    """
    `(pyglet.image.Texture, uvs)` for a `TextureVisuals`, or `(None, None)`.

    trimesh stores UVs in OpenGL convention (origin bottom-left); the glTF
    loader flips V on import. Don't flip again here.
    """
    if (
        visual is None
        or getattr(visual, "uv", None) is None
        or not hasattr(visual, "material")
    ):
        return None, None
    try:
        texture = to_texture(visual.material)
    except BaseException:
        util.log.warning("failed to convert texture", exc_info=True)
        return None, None
    if texture is None:
        return None, None
    return texture, np.asarray(visual.uv, dtype=np.float32)


# --- pyglet draw group -------------------------------------------------


class NodeGroup(BaseMaterialGroup):
    """
    Per-node draw group with identity equality and optional texture binding.

    Pyglet's default group equality is `(class, order, parent)`, which would
    consolidate every order-0 group into one draw call: a single `model`
    matrix would win and only one node would render. Identity equality keeps
    each node's draw distinct. `set_state` binds the texture only if there
    is one, so this single class covers both textured and untextured cases.
    """

    def __init__(self, program, texture=None, order: int = 0) -> None:
        super().__init__(material=None, program=program, order=order)
        self.texture = texture

    def set_state(self) -> None:
        if self.texture is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(self.texture.target, self.texture.id)
        self.program.use()
        self.program["model"] = self.matrix

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# A literal zero matrix produces NaN clip coords on Mesa-EGL and drops the
# whole draw, so collapse to a vanishingly small scale instead.
_TINY = 1e-30
_HIDDEN_MATRIX = Mat4(_TINY, 0, 0, 0, 0, _TINY, 0, 0, 0, 0, _TINY, 0, 0, 0, 0, 1)


def _drop_model(model) -> None:
    """Best-effort delete of every `VertexList` owned by a `pyglet.model.Model`."""
    if model is None:
        return
    for vertex_list in getattr(model, "vertex_lists", ()):
        try:
            vertex_list.delete()
        except BaseException:
            pass


def matrix_to_pyglet(matrix: NDArray[float64]) -> Mat4:
    """
    Row-major numpy 4x4 to column-major `pyglet.math.Mat4`. Skipping this
    silently transposes the upload: wrong picture, no error.
    """
    return Mat4(*np.asarray(matrix, dtype=np.float32).flatten("F"))


# --- UI state ---------------------------------------------------------


@dataclass
class _State:
    """All viewer UI state, in one place. None of these fields are read by
    pyglet, so adding a new toggle is just adding a field here + a method."""

    trackball: Trackball
    wireframe: bool = False
    cull: bool = True
    fullscreen: bool = False
    axis: str | None = None  # None | "world"
    grid: bool = False
    hidden: set = field(default_factory=set)
    fixed: set = field(default_factory=set)


# --- viewer -----------------------------------------------------------


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
        **kwargs,
    ):
        if not isinstance(scene, Scene):
            scene = Scene(scene)
        if resolution is None:
            resolution = scene.camera.resolution
        else:
            scene.camera.resolution = resolution
        self.scene = scene
        # let scene mutators trigger a redraw
        self.scene._redraw = self.on_draw
        self.callback = callback

        # `window_conf` is the legacy pyglet1 name; accept both.
        config = config or kwargs.pop("window_conf", None)
        self.window = pyglet.window.Window(
            width=int(resolution[0]),
            height=int(resolution[1]),
            caption=caption or "trimesh SceneViewer (`h` for help)",
            resizable=resizable,
            fullscreen=fullscreen,
            visible=visible,
            config=config,
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

        # one untextured + one textured shader program, shared across every Model.
        ctx = pyglet.gl.current_context
        self._program = ctx.create_program(
            (MaterialGroup.default_vert_src, "vertex"),
            (MaterialGroup.default_frag_src, "fragment"),
        )
        self._textured_program = ctx.create_program(
            (TexturedMaterialGroup.default_vert_src, "vertex"),
            (TexturedMaterialGroup.default_frag_src, "fragment"),
        )
        self._batch = pyglet.graphics.Batch()
        # one Model per scene-graph node, keyed by node name.
        self._models: dict[str, Model] = {}
        self._geometry_id: dict[str, tuple] = {}
        self._axis_model: Model | None = None
        self._grid_model: Model | None = None

        self._state = _State(
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

        self._update_models()
        if flags:
            for k, v in flags.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)
        self._update_flags()

        if callback is not None:
            pyglet.clock.schedule_interval(
                self._dispatch_callback,
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
        self._state.hidden.add(name)

    def unhide_geometry(self, name: str) -> None:
        self._state.hidden.discard(name)

    def reset_view(self) -> None:
        self._state.trackball = Trackball(
            pose=self._initial_camera,
            size=self.window.get_size(),
            scale=self.scene.scale,
            target=self.scene.centroid,
        )
        self.scene.camera_transform = self._initial_camera.copy()

    def toggle_wireframe(self) -> None:
        self._state.wireframe = not self._state.wireframe
        self._update_flags()

    def toggle_culling(self) -> None:
        self._state.cull = not self._state.cull
        self._update_flags()

    def toggle_fullscreen(self) -> None:
        self._state.fullscreen = not self._state.fullscreen
        self._update_flags()

    def toggle_axis(self) -> None:
        self._state.axis = "world" if self._state.axis is None else None
        self._update_flags()

    def toggle_grid(self) -> None:
        self._state.grid = not self._state.grid
        self._update_flags()

    def print_help(self) -> None:
        print(_build_help())  # noqa: T201

    # --- pyglet event handlers ---------------------------------------

    def on_draw(self) -> None:
        # The scene is the source of truth: input updates
        # `scene.camera_transform`, and we re-read it here so any external
        # call to `Scene.camera_project` agrees with the rendered frame.
        # `view`/`projection` setters are safe to use: they just write
        # into the `WindowBlock` UBO that pyglet's stock 3D shaders read.
        # Unlike `viewport`, which secretly multiplies by `dpi/96`.
        self.window.view = matrix_to_pyglet(np.linalg.inv(self.scene.camera_transform))
        self.window.projection = matrix_to_pyglet(self.scene.camera.projection)
        self._update_model_matrices()
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
        self._state.trackball.resize((width, height))

    def on_mouse_press(self, x, y, buttons, modifiers) -> None:
        ball = self._state.trackball
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
        self._state.trackball.drag(np.array([x, y]))
        self.scene.camera_transform = self._state.trackball.pose

    def on_mouse_scroll(self, x, y, dx, dy) -> None:
        self._state.trackball.scroll(dy)
        self.scene.camera_transform = self._state.trackball.pose

    def on_key_press(self, symbol, modifiers) -> None:
        key = pyglet.window.key
        for letter, _, method in _KEYS:
            if symbol == getattr(key, letter.upper()):
                getattr(self, method)()
                return
        if symbol in (key.LEFT, key.RIGHT, key.DOWN, key.UP):
            ball = self._state.trackball
            arrows = {
                key.LEFT: [-10, 0],
                key.RIGHT: [10, 0],
                key.DOWN: [0, -10],
                key.UP: [0, 10],
            }
            ball.down([0, 0])
            ball.drag(arrows[symbol])
            self.scene.camera_transform = ball.pose

    # --- geometry plumbing -------------------------------------------

    def _build_model(
        self,
        count: int,
        mode: int,
        indices,
        attributes: dict,
        *,
        texture=None,
        order: int = 0,
    ) -> Model:
        """Create a single-vertex-list Model in our shared batch."""
        program = self._textured_program if texture is not None else self._program
        group = NodeGroup(program, texture=texture, order=order)
        if indices is None:
            vertex_list = program.vertex_list(
                count, mode, batch=self._batch, group=group, **attributes
            )
        else:
            vertex_list = program.vertex_list_indexed(
                count, mode, indices, batch=self._batch, group=group, **attributes
            )
        return Model([vertex_list], [group], self._batch)

    def _to_model(self, geometry) -> Model | None:
        """Build a `Model` for one piece of geometry, or `None`."""
        # voxels: bake to boxes and fall through to Trimesh
        if type(geometry).__name__ == "VoxelGrid":
            geometry = geometry.as_boxes()
        kind = type(geometry).__name__

        if kind == "Trimesh":
            if getattr(geometry, "is_empty", False):
                return None
            visual = geometry.visual
            texture, uvs = _texture_for(visual)
            count = len(geometry.vertices)
            colors = (
                np.ones(count * 4, dtype=np.float32)
                if texture is not None
                else (np.asarray(visual.vertex_colors, dtype=np.float32) / 255.0).ravel()
            )
            attrs = {
                "POSITION": (
                    "f",
                    np.asarray(geometry.vertices, dtype=np.float32).ravel(),
                ),
                "NORMAL": (
                    "f",
                    np.asarray(geometry.vertex_normals, dtype=np.float32).ravel(),
                ),
                "COLOR_0": ("f", colors),
            }
            if uvs is not None:
                attrs["TEXCOORD_0"] = ("f", uvs.ravel())
            return self._build_model(
                count,
                GL_TRIANGLES,
                np.asarray(geometry.faces, dtype=np.uint32).ravel(),
                attrs,
                texture=texture,
                order=1 if getattr(visual, "transparency", False) else 0,
            )

        if kind == "PointCloud" or isinstance(geometry, np.ndarray):
            raw = geometry.vertices if kind == "PointCloud" else geometry
            points = np.asarray(raw, dtype=np.float32)
            if len(points) == 0:
                return None
            count = len(points)
            raw_colors = geometry.colors if kind == "PointCloud" else None
            if raw_colors is None:
                colors = np.ones(count * 4, dtype=np.float32)
            else:
                rgba = np.asarray(to_rgba(raw_colors), dtype=np.float32) / 255.0
                # broadcast a single rgba tuple across all points if needed
                if rgba.size == 4:
                    rgba = np.broadcast_to(rgba, (count, 4))
                colors = rgba.ravel()
            return self._build_model(
                count,
                GL_POINTS,
                None,
                {
                    "POSITION": ("f", points.ravel()),
                    "NORMAL": ("f", np.tile([0.0, 0.0, 1.0], count).astype(np.float32)),
                    "COLOR_0": ("f", colors),
                },
            )

        if kind in ("Path2D", "Path3D"):
            polylines, indices, offset = [], [], 0
            for poly in geometry.discrete:
                poly = np.asarray(poly, dtype=np.float32)
                if len(poly) < 2:
                    continue
                if poly.shape[1] == 2:
                    poly = np.column_stack([poly, np.zeros(len(poly), dtype=np.float32)])
                polylines.append(poly)
                for i in range(len(poly) - 1):
                    indices.extend((offset + i, offset + i + 1))
                offset += len(poly)
            if not polylines:
                return None
            return self._build_model(
                offset,
                GL_LINES,
                np.asarray(indices, dtype=np.uint32),
                {
                    "POSITION": ("f", np.concatenate(polylines).ravel()),
                    "NORMAL": ("f", np.tile([0.0, 0.0, 1.0], offset).astype(np.float32)),
                    "COLOR_0": ("f", np.ones(offset * 4, dtype=np.float32)),
                },
            )

        util.log.warning(f"_to_model: unsupported geometry {kind!r}")
        return None

    def _delete_model(self, key) -> None:
        _drop_model(self._models.pop(key, None))
        self._geometry_id.pop(key, None)

    def _update_models(self) -> None:
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
            self._delete_model(node)
            try:
                model = self._to_model(geometry)
            except BaseException:
                util.log.warning(f"failed to add node `{node}`", exc_info=True)
                continue
            if model is None:
                continue
            self._models[node] = model
            self._geometry_id[node] = current_id
        for node in list(self._models):
            if node not in live:
                self._delete_model(node)

    def _update_model_matrices(self) -> None:
        graph = self.scene.graph
        cam_inv = None  # lazy: only needed when `fixed` is non-empty
        for node in graph.nodes_geometry:
            transform, geom_name = graph.get(node)
            model = self._models.get(node)
            if model is None:
                continue
            if geom_name in self._state.hidden:
                model.matrix = _HIDDEN_MATRIX
                continue
            if geom_name in self._state.fixed:
                if cam_inv is None:
                    cam_inv = np.linalg.inv(self.scene.camera_transform)
                # cancel out camera so this geometry stays put in view
                transform = transform @ np.linalg.inv(self._initial_camera @ cam_inv)
            model.matrix = matrix_to_pyglet(transform)

    # --- flags / decoration models -----------------------------------

    def _update_flags(self) -> None:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self._state.wireframe else GL_FILL)
        (glEnable if self._state.cull else glDisable)(GL_CULL_FACE)
        self.window.set_fullscreen(fullscreen=self._state.fullscreen)
        self._sync_axis_model()
        self._sync_grid_model()

    def _sync_axis_model(self) -> None:
        if self._state.axis and self._axis_model is None:
            from ... import creation

            try:
                self._axis_model = self._to_model(
                    creation.axis(origin_size=self.scene.scale / 100)
                )
            except BaseException:
                util.log.warning("failed to create axis", exc_info=True)
        elif not self._state.axis and self._axis_model is not None:
            _drop_model(self._axis_model)
            self._axis_model = None

    def _sync_grid_model(self) -> None:
        if self._state.grid and self._grid_model is None:
            try:
                from ...path.creation import grid
                from ...transformations import translation_matrix

                bounds = self.scene.bounds
                center = bounds.mean(axis=0)
                center[2] = bounds[0][2] - (np.ptp(bounds[:, 2]) / 100)
                side = np.ptp(bounds, axis=0)[:2].max()
                self._grid_model = self._to_model(
                    grid(side=side, count=4, transform=translation_matrix(center))
                )
            except BaseException:
                util.log.warning("failed to create grid", exc_info=True)
        elif not self._state.grid and self._grid_model is not None:
            _drop_model(self._grid_model)
            self._grid_model = None

    # --- internal ----------------------------------------------------

    def _dispatch_callback(self, dt) -> None:
        if self.callback is not None:
            self.callback(self.scene)
        self._update_models()


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
