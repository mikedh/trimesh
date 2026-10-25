"""
viewer.py
-----------

An OpenGL viewer for `trimesh.Scene` which also renders images
offscreen. Geometry is converted to numpy arrays in `conversion.py`,
uploaded and drawn with raw OpenGL in `opengl/`, and shaded by the
small GLSL programs in `trimesh/viewer/opengl/shaders`.
"""

import time
from dataclasses import dataclass, field, fields, replace
from typing import NamedTuple

import numpy as np
import pyglet

from ... import util
from ...bounds import corners
from ...scene import Scene
from ...scene.cameras import current_view
from ...scene.metadata import RenderSettings
from ...typed import (
    ArrayLike,
    Integer,
    Iterable,
    NDArray,
    NDArray2D,
    NDArray3D,
    Number,
    Sequence,
)
from ...visual import to_rgba
from .. import opengl
from ..conversion import (
    ALPHA_MODES,
    geometry_to_arrays,
    light_arrays,
    shadow_projections,
)
from ..trackball import Trackball
from . import gl, shaders

# (input, action) for the help message
MOUSE_BINDINGS = (
    ("drag", "rotate"),
    ("ctrl + drag", "pan"),
    ("shift + drag", "roll"),
    ("wheel", "zoom"),
    ("arrows", "nudge"),
)

# (key, action, `ViewerState` switch or `SceneViewer` method) used for
# both key dispatch and help
KEY_BINDINGS = (
    ("w", "toggle wireframe", "wireframe"),
    ("s", "toggle smooth shading", "smooth"),
    ("z", "reset view", "reset_view"),
    ("c", "toggle back face culling", "cull"),
    ("a", "toggle axis", "axis"),
    ("g", "toggle grid", "grid"),
    ("f", "toggle fullscreen", "fullscreen"),
    ("m", "maximize", "maximize"),
    ("q", "quit", "close"),
    ("p", "pause animation", "playing"),
    ("l", "toggle shadows", "shadows"),
    ("o", "toggle ambient occlusion", "ambient_occlusion"),
    ("v", "toggle fast drawing", "fast"),
    ("h", "print help", "print_help"),
)

IDENTITY = np.eye(4)

# a visible viewer drawing slower than this switches to fast drawing
SLOW_FPS = 15.0

# maps clip space of -1.0 to 1.0 to texture coordinates of 0.0 to 1.0
SHADOW_BIAS = np.array(
    [[0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5], [0, 0, 0, 1.0]]
)

# one hidden window shared by every offscreen render, so rendering
# thousands of images doesn't create thousands of OpenGL contexts
_HIDDEN_CONTEXT = {}


@dataclass
class ViewerState:
    """
    Everything the user can toggle, in one place.
    """

    # set once the window exists
    trackball: Trackball | None = None
    smooth: bool = True
    wireframe: bool = False
    cull: bool = True
    fullscreen: bool = False
    axis: bool = False
    grid: bool = False
    # in pixels, or None for the scene's `RenderSettings`
    line_width: Number | None = None
    point_size: Number | None = None
    # scene graph nodes which aren't drawn
    hidden: set = field(default_factory=set)
    # geometry names which don't move with the camera
    fixed: Iterable = field(default_factory=set)
    # play `scene.animations` if there are any
    playing: bool = True
    # directional and spot lights cast shadows
    shadows: bool = True
    # nearby surfaces hide the environment from creases and corners
    ambient_occlusion: bool = True
    # skip shadows, ambient occlusion and softboxes for slow machines,
    # which a visible viewer turns on by itself if drawing is too slow
    fast: bool = False
    # the scene needs to be drawn again
    dirty: bool = True
    # geometry or transforms may have changed since they were uploaded,
    # which also needs a draw
    scene_changed: bool = True


class RenderArrays(NamedTuple):
    """
    What `SceneViewer.to_array` read back, with the first row at the top.
    """

    # (height, width, 4) uint8 antialiased image
    color: NDArray3D[np.uint8]
    # (height, width) distance along the view axis under the center of
    # every pixel, zero for the background, or None if not asked for
    depth: NDArray2D[np.float32] | None
    # (height, width) index into `scene.graph.nodes_geometry` under the
    # center of every pixel, -1 for the background, or None if not asked for
    segmentation: NDArray2D[np.int64] | None


@dataclass
class DrawTable:
    """
    What to draw, which only changes with the scene, so drawing a frame
    only has to look up where each node is. One row per node with drawable
    geometry, followed by one row each for the axis and grid.
    """

    # scene graph node for each row before the axis and grid
    nodes: list
    # `opengl.GeometryBuffers` for each row
    buffers: list
    # rows which stay put relative to the camera
    fixed: np.ndarray
    # rows which are translucent meshes, drawn last from back to front
    blended: np.ndarray
    # rows which are meshes rather than lines or points
    mesh: np.ndarray
    # rows sharing an index draw the same buffers or material
    geometry_index: np.ndarray
    material_index: np.ndarray
    # (n, 3) center of each row's vertices for sorting translucency
    centroids: np.ndarray
    # (n,) index into `scene.graph.nodes_geometry` or -1 for decorations
    node_index: np.ndarray
    # (n,) rows which are the invisible floor which only shows shadows
    catcher: np.ndarray
    # (n,) row of each node's first joint in the joint matrices uploaded
    # every frame, one skin after another, or -1
    joint_offset: np.ndarray
    # every joint of every skin once, and each skin joint's index into them
    joint_nodes: list
    joint_index: np.ndarray
    # (b,) skin joint and row pairs whose corners bound skinned rows
    bound_joints: np.ndarray
    bound_rows: np.ndarray
    # scene graph nodes whose world transform is needed every frame:
    # `nodes`, then the scene's own lights if it has any, then `joint_nodes`
    world_nodes: list
    # every edge between the world and those nodes, parents before
    # children, with the row of each edge's parent edge or -1 for the
    # world, the rows of each depth of the tree, and the row of each of
    # `world_nodes` or -1 for one which is the world itself
    chain_edges: list
    chain_parent: np.ndarray
    chain_levels: list
    world_rows: np.ndarray
    # (j, 4, 4) inverse bind matrix of every skin joint in order
    inverse_binds: np.ndarray
    # (n, 8, 3) corners of the box around each row's vertices
    corners: np.ndarray
    # `conversion.LightArrays` from `Scene.lights`
    lights: object = None


class SceneViewer:
    """
    An interactive window for a `trimesh.Scene`.

    The OpenGL window is `viewer.window`, and the camera being drawn is
    `viewer.camera` at `viewer.camera_transform`: a copy of the scene's
    camera, or one framing the scene, so viewing never changes a scene.
    """

    def __init__(
        self,
        scene,
        visible: bool = True,
        resolution=None,
        start_loop: bool = True,
        callback=None,
        callback_period: float | None = None,
        caption: str | None = None,
        background=None,
        config=None,
        flags: dict | None = None,
        **settings,
    ):
        """
        Create a viewer for a scene.

        Parameters
        ------------
        scene : trimesh.Scene or geometry
          Geometry to view.
        visible : bool
          Show a window: if False the viewer only renders offscreen.
        resolution : (2,) int or None
          Window size, defaults to `scene.camera.resolution`.
        start_loop : bool
          Run the event loop until the window is closed.
        callback : function or None
          Called with the scene every `callback_period` seconds.
        callback_period : float or None
          Seconds between callbacks, defaults to 1/30.
        caption : str or None
          Window title.
        background : None or color
          Background color, defaults to white.
        config : pyglet config or None
          Override the OpenGL context configuration.
        flags : dict or None
          Starting values for `ViewerState`, the same as passing them
          as keywords.
        settings : dict
          Starting values for `ViewerState`, i.e. `smooth=False`,
          `line_width=4.0` or `fixed=["axis"]`: anything else raises.
        """
        if not isinstance(scene, Scene):
            scene = Scene(scene)
        self.scene = scene
        self.callback = callback
        self.closed = False
        self.frame_count = 0
        # how long drawing took, to switch to fast drawing on slow machines
        self.draw_seconds = []
        self.started = time.perf_counter()
        self.chose_fast = False
        # everything the user can toggle, where a misspelled name raises
        self.state = replace(ViewerState(), **{**(flags or {}), **settings})

        if background is None:
            self.background = np.ones(4)
        else:
            self.background = to_rgba(background) / 255.0

        # the viewer's own camera, starting where the scene's camera is
        self.camera, camera_transform = current_view(scene)
        self.place_camera(camera_transform)
        if resolution is None:
            resolution = self.camera.resolution

        if visible:
            self.context = opengl.create_context(
                width=resolution[0],
                height=resolution[1],
                visible=True,
                resizable=True,
                fullscreen=self.state.fullscreen,
                caption=caption or "trimesh SceneViewer (`h` for help)",
                config=config,
            )
            framebuffer_size = self.context.window.get_framebuffer_size()
        else:
            self.context = hidden_context()
            framebuffer_size = resolution
        self.window = self.context.window
        self.window.switch_to()

        # geometry name to `GeometryBuffers` and the hash it was built from
        self.geometry_buffers = {}
        self.geometry_keys = {}
        # the scene graph hash and what it drew where when last checked
        self.synced_graph = None
        self.graph_structure = None
        # the axis and grid, which are drawn with an identity transform
        self.decoration_buffers = {}
        # updated with the geometry so the depth range fits the scene
        self.scene_bounds = None
        # the scene graph hash the world transforms were placed for
        self.placed = None
        # each shadow casting light's projection fit around the scene and
        # its texel size, from `shadow_projections` when the maps are drawn
        self.shadow_fit = None

        self.initial_camera = self.camera_transform.copy()
        # fixed geometry is drawn as if the camera never moved
        self.initial_view = np.linalg.inv(self.initial_camera)
        # the default lights and environment are fixed in the world
        # around the camera the viewer starts at
        self.studio = RenderSettings.from_metadata(scene.metadata).resolve(
            scene, self.initial_camera
        )
        self.images = []
        self.allocate(framebuffer_size)
        # the softboxes as the frame stores them, in world space
        softboxes = self.studio.environment.softboxes[: shaders.MAX_SOFTBOXES]
        self.softboxes = np.zeros(len(softboxes), dtype=shaders.FRAME["softboxes"].base)
        rows = np.reshape(
            [[*box.direction, box.sharpness, *box.color] for box in softboxes], (-1, 7)
        )
        self.softboxes["direction"][:, :3] = util.unitize(rows[:, :3])
        self.softboxes["direction"][:, 3] = rows[:, 3]
        self.softboxes["color"][:, :3] = rows[:, 4:]
        # playing animations poses the scene's own graph, so what they
        # drive is put back when the viewer closes
        forest = scene.graph.transforms
        animated = dict.fromkeys(
            child for animation in scene.animations for _, child in animation.edges
        )
        # every node has one parent, which is the edge an animation drives
        self.animated_edges = [
            (forest.parents[child], child)
            for child in animated
            if child in forest.parents
        ]
        self.animated_matrices = np.reshape(
            [
                forest.edge_data[edge].get("matrix", IDENTITY)
                for edge in self.animated_edges
            ],
            (-1, 4, 4),
        )
        self.state.trackball = self.create_trackball()
        self.update_decorations()

        if visible:
            # pyglet calls these by name when the window gets events
            self.window.push_handlers(
                self.on_draw,
                self.on_resize,
                self.on_mouse_press,
                self.on_mouse_drag,
                self.on_mouse_scroll,
                self.on_key_press,
            )
        if callback is not None:
            pyglet.clock.schedule_interval(
                self.on_callback, callback_period or 1.0 / 30.0
            )
        # seconds into the animation, advanced by the clock while playing
        self.animation_time = 0.0
        if visible and len(scene.animations) > 0:
            pyglet.clock.schedule_interval(self.on_animate, 1.0 / 60.0)
        if start_loop and visible:
            self.run()

    def create_trackball(self) -> Trackball:
        """
        A trackball starting at the initial camera.

        Returns
        ------------
        trackball : Trackball
          Sized to the window and centered on the scene.
        """
        return Trackball(
            pose=self.initial_camera,
            # mouse events are in window pixels, not framebuffer pixels
            size=np.array(self.window.get_size()),
            scale=self.scene.scale,
            target=self.scene.centroid,
        )

    def run(self) -> None:
        """
        Run the event loop until the window is closed. This is a plain
        loop rather than `pyglet.app.run` so `show()` can be called
        any number of times and always returns.
        """
        if pyglet.options["headless"]:
            util.log.warning("no display available: use `scene.save_image` instead")
            self.close()
            return
        window = self.window
        try:
            while not (self.closed or window.has_exit):
                pyglet.clock.tick()
                window.switch_to()
                window.dispatch_events()
                if self.closed or window.has_exit:
                    break
                if not self.on_draw():
                    # nothing changed so don't spin the CPU
                    time.sleep(1.0 / 60.0)
                window.flip()
        finally:
            # including when drawing raised
            self.close()

    def close(self) -> None:
        """
        Close the window and free everything on the GPU.
        """
        if self.closed:
            return
        self.closed = True
        pyglet.clock.unschedule(self.on_callback)
        pyglet.clock.unschedule(self.on_animate)
        self.window.switch_to()
        for item in (
            *self.geometry_buffers.values(),
            *self.decoration_buffers.values(),
            *self.images,
        ):
            item.delete()
        self.geometry_buffers.clear()
        self.decoration_buffers.clear()
        if len(self.animated_edges) > 0:
            # put back the pose the scene had before playing its animations
            self.scene.graph.update_matrices(self.animated_edges, self.animated_matrices)
        # the hidden window is shared and stays open
        if self.context is not _HIDDEN_CONTEXT.get("context"):
            self.context.delete()
            self.window.close()

    def allocate(self, size: Sequence[Integer] | NDArray) -> None:
        """
        Allocate the framebuffers and images every frame is drawn into,
        freeing any from before, and fit the camera to their size.

        Parameters
        ------------
        size : (2,) int
          Width and height in pixels.
        """
        for image in self.images:
            image.delete()
        # pixels are square, keeping the camera's vertical field of view
        self.camera.fit_resolution(size)
        self.framebuffer = opengl.create_framebuffer(*size)
        self.resolve_framebuffer = opengl.create_framebuffer(*size, samples=0)
        self.occlusion = opengl.create_ambient_occlusion(*size)
        self.images = [self.framebuffer, self.resolve_framebuffer, self.occlusion]
        # images for bloom only if the scene has any
        self.bloom = None
        if self.studio.bloom > 0.0:
            self.bloom = opengl.create_bloom(*size)
            self.images.append(self.bloom)

    def place_camera(self, camera_transform: ArrayLike) -> None:
        """
        Move the viewer's camera to where a camera node is. glTF lets a
        camera node have a scale, which files from 3ds Max do and which
        only scales the camera's planes, so it is kept in `camera_scale`
        and `camera_transform`, the trackball and geometry fixed to the
        camera only ever see a rotation and a translation.

        Parameters
        ------------
        camera_transform : (4, 4) float
          Camera to world transform, which may have a scale.
        """
        pose = np.array(camera_transform, dtype=np.float64)
        scale = np.linalg.norm(pose[:3, :3], axis=0)
        pose[:3, :3] /= scale
        self.camera_transform = pose
        # one length for the planes, if the scale isn't the same on every axis
        self.camera_scale = scale.mean()
        self.state.dirty = True

    def maximize(self) -> None:
        """
        Maximize the window.
        """
        self.window.maximize()

    def print_help(self) -> None:
        """
        Print the mouse and key bindings.
        """
        rows = (*MOUSE_BINDINGS, *((key, action) for key, action, _ in KEY_BINDINGS))
        width = max(len(label) for label, _ in rows)
        lines = (f"  {label:<{width}}  {action}" for label, action in rows)
        print("\n".join(["trimesh SceneViewer", *lines]))  # noqa: T201

    def hide_geometry(self, node: str) -> None:
        """
        Stop drawing the geometry at a node.

        Parameters
        ------------
        node : str
          Scene graph node with geometry.
        """
        self.state.hidden.add(node)
        self.state.scene_changed = True

    def unhide_geometry(self, node: str) -> None:
        """
        Draw a node hidden with `hide_geometry` again.

        Parameters
        ------------
        node : str
          Scene graph node with geometry.
        """
        self.state.hidden.discard(node)
        self.state.scene_changed = True

    def reset_view(self, flags: dict | None = None) -> None:
        """
        Move the camera back to where it started.

        Parameters
        ------------
        flags : dict or None
          New values for `ViewerState`, i.e. `{"wireframe": True}`.
        """
        for name, value in (flags or {}).items():
            setattr(self.state, name, value)
        self.state.trackball = self.create_trackball()
        self.camera_transform = self.initial_camera.copy()
        self.state.dirty = True

    def toggle(self, name: str) -> None:
        """
        Flip one of the `ViewerState` switches, i.e. "wireframe".

        Parameters
        ------------
        name : str
          A boolean field of `ViewerState`.
        """
        state = self.state
        setattr(state, name, not getattr(state, name))
        state.dirty = True
        if name == "fullscreen":
            self.window.set_fullscreen(state.fullscreen)
        elif name in ("axis", "grid"):
            self.update_decorations()
        elif name == "fast":
            # the user decided so never switch on the user's behalf
            self.chose_fast = True

    def update_decorations(self) -> None:
        """
        Create or delete the axis, grid and floor to match the state.
        """
        from ...base import Trimesh
        from ...creation import axis
        from ...path.creation import grid
        from ...transformations import translation_matrix

        wanted = {
            "axis": bool(self.state.axis),
            "grid": bool(self.state.grid),
            "ground": self.studio.ground is not None,
        }
        for name, show in wanted.items():
            if show == (name in self.decoration_buffers):
                continue
            if not show:
                self.decoration_buffers.pop(name).delete()
                continue
            bounds = self.scene.bounds
            if name == "axis":
                geometry = axis(origin_size=self.scene.scale / 100.0)
            elif name == "ground":
                geometry = Trimesh(
                    vertices=self.studio.ground,
                    faces=[[0, 1, 2], [0, 2, 3]],
                    process=False,
                )
            else:
                # a grid just under the scene
                center = bounds.mean(axis=0)
                center[2] = bounds[0][2] - np.ptp(bounds[:, 2]) / 100.0
                geometry = grid(
                    side=np.ptp(bounds, axis=0)[:2].max(),
                    count=4,
                    transform=translation_matrix(center),
                )
            arrays = geometry_to_arrays(geometry)
            if name == "ground":
                # a floor which only shows shadows, blended over the background
                arrays.material.shadow_catcher = True
                arrays.material.alpha_mode = ALPHA_MODES["BLEND"]
            self.window.switch_to()
            self.decoration_buffers[name] = opengl.upload_geometry(arrays)
        self.state.scene_changed = True

    def sync_geometry(self) -> bool:
        """
        Upload any geometry that is new or changed since the last
        frame and delete any that was removed from the scene.

        Returns
        ------------
        changed : bool
          If any geometry changed, or which nodes of the scene graph
          draw it, so the draw table has to be built again.
        """
        # a key which changes whenever geometry would draw differently,
        # including colors which aren't part of every geometry's hash
        keys = {}
        for name, geometry in self.scene.geometry.items():
            colors = getattr(geometry, "colors", None)
            if colors is not None:
                colors = hash(np.asarray(colors).tobytes())
            keys[name] = (hash(geometry), hash(getattr(geometry, "visual", None)), colors)
        changed = keys != self.geometry_keys
        for name, key in keys.items():
            if self.geometry_keys.get(name) == key:
                continue
            previous = self.geometry_buffers.pop(name, None)
            if previous is not None:
                previous.delete()
            arrays = geometry_to_arrays(self.scene.geometry[name])
            if arrays is not None:
                self.geometry_buffers[name] = opengl.upload_geometry(arrays)
        for name in set(self.geometry_buffers).difference(keys):
            self.geometry_buffers.pop(name).delete()
        self.geometry_keys = keys
        # every matrix is read every frame, so only the parents and the
        # geometry of every node matter, which only change with the hash
        graph = self.scene.graph
        if hash(graph) != self.synced_graph:
            self.synced_graph = hash(graph)
            forest = graph.transforms
            structure = (
                dict(forest.parents),
                {node: data.get("geometry") for node, data in forest.node_data.items()},
            )
            changed = changed or structure != self.graph_structure
            self.graph_structure = structure
        return changed

    def draw_table(self) -> DrawTable:
        """
        Work out what to draw from the scene graph and uploaded geometry.

        Returns
        ------------
        table : DrawTable
          Everything about drawing which only changes with the scene.
        """
        graph = self.scene.graph
        state = self.state
        node_data = graph.transforms.node_data
        drawn = {node: node_data[node].get("geometry") for node in graph.nodes_geometry}
        nodes = [
            node
            for node, name in drawn.items()
            if name in self.geometry_buffers and node not in state.hidden
        ]
        names = [drawn[node] for node in nodes]
        buffers = [self.geometry_buffers[name] for name in names]
        buffers.extend(self.decoration_buffers.values())
        # numbered by first appearance, so the draw order doesn't depend
        # on where in memory the buffers happen to be
        first = {}
        geometry_index = np.array(
            [first.setdefault(id(item), len(first)) for item in buffers], dtype=np.int64
        )
        _, material_index = np.unique(
            [hash(item.material_key) for item in buffers], return_inverse=True
        )
        mesh = np.array([item.kind == "mesh" for item in buffers], dtype=bool)
        alpha = np.array([item.material.alpha_mode for item in buffers], dtype=np.int64)
        fixed = np.zeros(len(buffers), dtype=bool)
        fixed[: len(names)] = [name in state.fixed for name in names]
        position = {node: index for index, node in enumerate(graph.nodes_geometry)}
        node_index = np.full(len(buffers), -1.0)
        node_index[: len(nodes)] = [position[node] for node in nodes]

        # skinned meshes look up their joints from where their skin's
        # four rows per joint start, with a skin shared by nodes once
        skinned = [
            (row, self.scene.skins[node])
            for row, node in enumerate(nodes)
            if node in self.scene.skins and buffers[row].skinned
        ]
        skins = list({id(skin): skin for _, skin in skinned}.values())
        order = {id(skin): index for index, skin in enumerate(skins)}
        skin_index = np.full(len(buffers), -1)
        skin_index[[row for row, _ in skinned]] = [order[id(skin)] for _, skin in skinned]
        # where each skin's joints start in the matrices uploaded every frame
        joint_starts = np.cumsum([0] + [len(skin.joints) for skin in skins])
        joint_offset = np.where(skin_index >= 0, 4.0 * joint_starts[skin_index], -1.0)
        # a skinned row is anywhere one of its skin's joints could put it,
        # so its box is bounded by every corner under every one of them
        rows = np.flatnonzero(skin_index >= 0)
        bound_rows = np.repeat(rows, np.diff(joint_starts)[skin_index[rows]])
        bound_joints = np.concatenate(
            [np.zeros(0, dtype=np.int64)]
            + [np.arange(joint_starts[s], joint_starts[s + 1]) for s in skin_index[rows]]
        )
        # skins usually share a skeleton, so each joint is looked up once
        joint_nodes = list(
            dict.fromkeys(joint for skin in skins for joint in skin.joints)
        )
        lookup = {node: index for index, node in enumerate(joint_nodes)}
        lights = light_arrays(
            self.studio.lights, limit=shaders.MAX_LIGHTS, shadows=shaders.MAX_SHADOWS
        )
        # the scene's own lights move with the scene graph, and the
        # default lights stay where the viewer started
        scene_lights = []
        if self.scene.has_lights:
            scene_lights = lights.nodes
        # every node between the world and the ones whose transforms are needed
        world_nodes = [*nodes, *scene_lights, *joint_nodes]
        # and how many of them are above each, so parents come first
        parents = graph.transforms.parents
        depth = {}
        for node in world_nodes:
            walked = []
            while node in parents and node not in depth:
                walked.append(node)
                node = parents[node]
            above = depth.get(node, -1)
            depth.update(
                (item, above + count) for count, item in enumerate(walked[::-1], 1)
            )
        ordered = sorted(depth, key=depth.get)
        row = {node: index for index, node in enumerate(ordered)}
        chain_edges = [(parents[node], node) for node in ordered]
        chain_parent = np.array(
            [row.get(parents[node], -1) for node in ordered], dtype=np.int64
        )
        depth = np.array([depth[node] for node in ordered], dtype=np.int64)

        return DrawTable(
            nodes=nodes,
            buffers=buffers,
            fixed=fixed,
            blended=mesh & (alpha == ALPHA_MODES["BLEND"]),
            mesh=mesh,
            geometry_index=geometry_index,
            material_index=material_index,
            centroids=np.reshape([item.centroid for item in buffers], (-1, 3)),
            node_index=node_index,
            catcher=np.array(
                [item.material.shadow_catcher for item in buffers], dtype=bool
            ).reshape(-1),
            joint_offset=joint_offset,
            joint_nodes=joint_nodes,
            joint_index=np.array(
                [lookup[joint] for skin in skins for joint in skin.joints], dtype=np.int64
            ),
            inverse_binds=np.concatenate(
                [np.zeros((0, 4, 4))] + [skin.inverse_bind_matrices for skin in skins]
            ),
            bound_joints=bound_joints,
            bound_rows=bound_rows,
            world_nodes=world_nodes,
            chain_edges=chain_edges,
            chain_parent=chain_parent,
            chain_levels=[
                np.flatnonzero(depth == level)
                for level in range(depth.max(initial=-1) + 1)
            ],
            world_rows=np.array(
                [row.get(node, -1) for node in world_nodes], dtype=np.int64
            ),
            corners=np.reshape([item.corners for item in buffers], (-1, 8, 3)),
            lights=lights,
        )

    def projection(self, view: np.ndarray) -> np.ndarray:
        """
        The camera projection with its near and far planes moved in to
        just contain the scene, so depth precision is never wasted.
        This only changes depth: pixels land where the camera says, and
        nothing past the camera's own planes is drawn.

        Parameters
        ------------
        view : (4, 4) float
          World to camera transform.

        Returns
        ------------
        projection : (4, 4) float
          Camera to clip space.
        """
        # the scale of the camera's node scales its planes and how far
        # across an orthographic camera sees
        camera = self.camera.copy()
        near, far = camera.z_near * self.camera_scale, camera.z_far * self.camera_scale
        camera.z_near, camera.z_far = near, far
        if camera.magnification is not None:
            camera.magnification = camera.magnification * self.camera_scale
        bounds = self.scene_bounds
        if bounds is None:
            return camera.projection
        # the scene corners, the origin for the axis, and the floor
        points = np.vstack((corners(bounds), np.zeros(3)))
        if self.studio.ground is not None:
            points = np.vstack((points, self.studio.ground))
        depth = -(points @ view[:3, :3].T + view[:3, 3])[:, 2]
        # the same as `Scene.scale` without measuring the scene again
        margin = 0.05 * np.linalg.norm(np.ptp(bounds, axis=0))
        camera.z_far = min(depth.max() + margin, far)
        # a 24 bit depth buffer tells apart depths `far^2 / (near * 2^24)`
        # apart at the far plane, which this keeps under 1% of `far`
        camera.z_near = max(depth.min() - margin, near, camera.z_far * 1e-5)
        if camera.z_near >= camera.z_far:
            # the scene is entirely outside the camera's planes
            camera.z_near, camera.z_far = near, far
        return camera.projection

    def draw(self, target: opengl.Framebuffer | None = None) -> None:
        """
        Draw the scene into `self.framebuffer`.

        Parameters
        ------------
        target : None or opengl.Framebuffer
          Instead draw the view depth and node index of every pixel
          into this single-sample `GL_RG32F` framebuffer.
        """
        state = self.state
        if state.scene_changed:
            self.sync_geometry()
            self.table = self.draw_table()
            self.placed = None
            state.scene_changed = False
        framebuffer = self.framebuffer
        write = target is not None
        if write:
            framebuffer = target
        context = self.context
        table = self.table
        graph = self.scene.graph

        view = np.linalg.inv(self.camera_transform)
        # the only per-node work: where every node, light and joint is,
        # which only changes with the scene graph and not the camera
        if self.placed != hash(graph):
            self.placed = hash(graph)
            # every edge's matrix, then the world transform of every node
            # from its parent's, one depth of the tree at a time
            edge_data = graph.transforms.edge_data
            chain = np.reshape(
                [edge_data[edge].get("matrix", IDENTITY) for edge in table.chain_edges],
                (-1, 4, 4),
            )
            for rows in table.chain_levels[1:]:
                chain[rows] = np.matmul(chain[table.chain_parent[rows]], chain[rows])
            # with the identity for anything which is the world itself
            world = np.concatenate((chain, IDENTITY[None]))[table.world_rows]
            count = len(table.nodes)
            lit = len(table.world_nodes) - len(table.joint_nodes)
            # with an identity for the axis and grid
            transforms = np.tile(IDENTITY, (len(table.buffers), 1, 1))
            transforms[:count] = world[:count]
            lights = self.studio.transforms[: len(table.lights.nodes)]
            if lit > count:
                lights = world[count:lit]
            # `world(joint) @ inverse_bind` for every joint of every skin
            joints = np.matmul(world[lit:][table.joint_index], table.inverse_binds)
            # where the scene is from the box around every node's geometry,
            # and every skinned row's box under every joint of its skin, which
            # fits the depth range and shadows without touching vertices
            boxes = np.concatenate((transforms[:count], joints[table.bound_joints]))
            rows = np.concatenate((np.arange(count), table.bound_rows))
            points = (
                np.matmul(table.corners[rows], np.swapaxes(boxes[:, :3, :3], 1, 2))
                + boxes[:, None, :3, 3]
            ).reshape((-1, 3))
            self.scene_bounds = None
            if len(points) > 0:
                self.scene_bounds = np.array([points.min(axis=0), points.max(axis=0)])
            # a skinned mesh's joints put it in world space
            transforms[table.joint_offset >= 0] = IDENTITY
            self.world = (transforms, lights, joints.astype(np.float32))
        transforms, lights, joints = self.world
        # every viewer of the hidden window shares its joint matrices and
        # shadow maps, so they are filled again if anything else filled them
        moved = context.placement is not self.world
        if moved:
            opengl.upload_array(gl.GL_TEXTURE_BUFFER, context.joint_buffer, joints)
            # until the shadow maps are drawn for these joints too
            context.placement = None
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["joint_matrices"])
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, context.joint_texture)
        model_view = np.matmul(view, transforms)
        model_view[table.fixed] = np.matmul(self.initial_view, transforms[table.fixed])

        # everything every program reads once per frame
        frame = np.zeros(1, dtype=shaders.FRAME)
        frame["projection"] = self.projection(view)
        frame["inverse_projection"] = np.linalg.inv(frame["projection"][0])
        frame["occlusion_radius"][0, 0] = self.studio.occlusion_radius
        frame["exposure"][0, 0] = self.studio.exposure
        if self.bloom is not None:
            frame["bloom"][0, :3] = self.studio.bloom, self.bloom.spread, self.bloom.level
        frame["viewport_size"][:, :2] = (framebuffer.width, framebuffer.height)
        frame["write_depth_and_index"][:, 0] = write
        self.fill_lights(frame, view, lights)
        if state.shadows and not write and not state.fast:
            self.draw_shadows(frame, transforms, lights, self.camera_transform, moved)
            context.placement = self.world
        # a white image is no occlusion at all
        occlusion = context.white_texture
        if state.ambient_occlusion and not write and not state.fast:
            occlusion = self.draw_ambient_occlusion(frame, model_view)
        opengl.upload_array(gl.GL_UNIFORM_BUFFER, context.frame_buffer, frame)
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["shadow_maps"])
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, context.shadow_maps.texture)
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["ambient_occlusion"])
        gl.glBindTexture(gl.GL_TEXTURE_2D, occlusion)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer.framebuffer)
        gl.glViewport(0, 0, framebuffer.width, framebuffer.height)
        # depth writes have to be on for the clear to reset depth
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClearColor(*self.background)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self.studio.show_environment and not write:
            # the environment behind everything, from one triangle
            # covering the screen which writes no depth
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_BLEND)
            gl.glUseProgram(context.background_program.program)
            gl.glBindVertexArray(context.empty_vertex_array)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # equal depth passes so line segments sharing an endpoint join
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_BLEND)
        if write:
            # background is depth zero and node -1, which a clamped
            # `glClearColor` might not write, and nothing is blended
            gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, -1.0, 0.0, 0.0))
            gl.glDisable(gl.GL_BLEND)
        # every program writes color already multiplied by its alpha, so
        # a transparent surface can still add its reflections
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        # meshes are pushed back slightly so lines on their surface win
        gl.glPolygonOffset(1.0, 1.0)

        if len(table.buffers) > 0:
            blended, mesh = table.blended, table.mesh
            geometry_index = table.geometry_index
            # opaque meshes grouped by material and geometry so every node
            # sharing a mesh is one instanced draw call, then translucent
            # meshes one at a time from the farthest to the nearest
            view_z = model_view[:, 2]
            depth = (view_z[:, :3] * table.centroids).sum(axis=1) + view_z[:, 3]
            order = np.lexsort(
                (
                    np.where(blended, depth, geometry_index),
                    np.where(blended, 0, table.material_index),
                    blended,
                )
            )
            if write:
                # the axis and grid aren't part of the scene
                order = order[order < len(table.nodes)]
            # nodes scaled to nothing aren't drawn, which is how glTF hides
            # things and otherwise points and lines would leave a dot
            order = order[np.abs(model_view[order, :3, :3]).max(axis=(1, 2)) > 0.0]

            # normals transform by the inverse transpose, which up to a scale
            # the shader normalizes away is the cofactor matrix: the cross
            # products of the columns, which unlike an inverse exists even
            # for a node animated to zero scale. a mirror flips the sign
            linear = model_view[order, :3, :3]
            columns = linear.transpose((0, 2, 1))
            cofactor = np.cross(
                np.roll(columns, -1, axis=1), np.roll(columns, -2, axis=1)
            )
            cofactor *= np.sign(np.linalg.det(linear))[:, None, None]
            # the per-instance matrices in draw order, stored transposed
            # for OpenGL: the rows of `cofactor` already are its columns
            instances = np.zeros(len(order), dtype=shaders.INSTANCE)
            instances["model_view"] = model_view[order].transpose((0, 2, 1))
            instances["normal_matrix"] = cofactor
            instances["node_index"] = table.node_index[order]
            instances["joint_offset"] = table.joint_offset[order]
            # start a new draw call whenever the geometry changes and for
            # every translucent mesh and every line
            single = (blended | ~mesh)[order]
            starts = np.flatnonzero(
                (np.diff(geometry_index[order], prepend=-1) != 0) | single
            )
            ends = np.append(starts[1:], len(order))

            bound = None
            for start, end in zip(starts, ends):
                item = table.buffers[order[start]]
                if single[start] and mesh[order[start]] and not write:
                    # translucent meshes don't hide what is behind them
                    gl.glDepthMask(gl.GL_FALSE)
                if item is not bound:
                    program = self.bind_geometry(item, bound)
                    bound = item
                if item.kind == "mesh":
                    opengl.upload_array(
                        gl.GL_ARRAY_BUFFER, item.instance_buffer, instances[start:end]
                    )
                    gl.glDrawElementsInstanced(
                        gl.GL_TRIANGLES, item.count, gl.GL_UNSIGNED_INT, None, end - start
                    )
                else:
                    program.set("model_view", model_view[order[start]])
                    program.set("node_index", table.node_index[order[start]])
                    gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, item.count)

        gl.glBindVertexArray(context.empty_vertex_array)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDepthMask(gl.GL_TRUE)
        self.frame_count += 1
        state.dirty = False

    def fill_lights(self, frame: np.ndarray, view: np.ndarray, world: np.ndarray) -> None:
        """
        Put every light where it is this frame and the environment
        around the scene, in view space where the shader lights everything.

        Parameters
        ------------
        frame : (1,) shaders.FRAME
          The frame values to fill in.
        view : (4, 4) float
          World to camera transform.
        world : (n, 4, 4) float
          Where each light in `self.table.lights` is.
        """
        lights = self.table.lights
        count = len(lights.nodes)
        light_view = np.matmul(view, world)
        # a light shines down its -Z so its +Z points back towards it
        towards = util.unitize(light_view[:, :3, 2])
        rows = frame["lights"][0, :count]
        rows["position"][:, :3] = np.where(
            lights.directional[:, None], towards, light_view[:, :3, 3]
        )
        rows["position"][:, 3] = ~lights.directional
        rows["direction"][:, :3] = -towards
        rows["color"][:, :3] = lights.colors
        rows["falloff"] = lights.falloffs
        # nothing is shadowed until `draw_shadows` draws the shadows
        rows["shadow"][:, 0] = -1.0
        frame["light_count"][0, 0] = count

        environment = self.studio.environment
        frame["sky_color"][0, :3] = environment.sky
        frame["ground_color"][0, :3] = environment.ground
        frame["up_direction"][0, :3] = view[:3, :3] @ util.unitize(environment.up)
        count = len(self.softboxes)
        if self.state.fast:
            # just the gradient, which costs almost nothing per pixel
            count = 0
        boxes = frame["softboxes"][0, :count]
        boxes[:] = self.softboxes[:count]
        boxes["direction"][:, :3] = boxes["direction"][:, :3] @ view[:3, :3].T
        frame["softbox_count"][0, 0] = count
        frame["ground_light"][0, 0] = self.studio.ground_light

    def draw_shadows(
        self,
        frame: np.ndarray,
        transforms: np.ndarray,
        lights: np.ndarray,
        pose: np.ndarray,
        redraw: bool,
    ) -> None:
        """
        Draw the depth every shadow casting light sees into its layer
        of the shadow maps, which is every opaque mesh seen from the
        light with a projection fit around the scene, then turn the
        light's shadow on in the frame values.

        Parameters
        ------------
        frame : (1,) shaders.FRAME
          This frame's values, which get every light's shadow.
        transforms : (n, 4, 4) float
          Where each row of `self.table` is.
        lights : (m, 4, 4) float
          Where each light in `self.table.lights` is.
        pose : (4, 4) float
          Camera to world transform without a scale.
        redraw : bool
          Draw the maps rather than only look them up, which only
          changes with where things are and not with the camera.
        """
        table = self.table
        shadow_maps = self.context.shadow_maps
        layers = table.lights.layers
        casting = layers >= 0
        if not casting.any() or self.scene_bounds is None:
            return
        if redraw:
            self.shadow_fit = shadow_projections(
                lights[casting],
                table.lights.directional[casting],
                table.lights.cone_angles[casting],
                self.scene_bounds,
                shadow_maps.resolution,
            )
        clip, texels = self.shadow_fit
        # the shader looks shadows up from view space
        frame["shadow_matrices"][0, : len(clip)] = np.matmul(
            SHADOW_BIAS, np.matmul(clip, pose)
        )
        shadow = frame["lights"]["shadow"][0, : len(layers)]
        shadow[casting, 0] = layers[casting]
        shadow[casting, 1:3] = texels
        if not redraw:
            return

        # opaque meshes cast shadows
        casts = table.mesh & ~table.blended & ~table.fixed
        casts[len(table.nodes) :] = False
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, shadow_maps.framebuffer)
        gl.glViewport(0, 0, shadow_maps.resolution, shadow_maps.resolution)
        # push depth away from the light so a lit surface doesn't
        # shadow itself, more where it's seen at a grazing angle
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(2.0, 4.0)
        for layer, light_clip in enumerate(clip):
            gl.glFramebufferTextureLayer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, shadow_maps.texture, 0, layer
            )
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            # the instances are in world space so the light's view is
            # part of its projection
            self.draw_depth(np.flatnonzero(casts), transforms, light_clip, frame)

    def draw_depth(
        self,
        rows: np.ndarray,
        model_view: np.ndarray,
        projection: np.ndarray,
        frame: np.ndarray,
    ) -> None:
        """
        Draw only the depth of some rows of the draw table into the
        framebuffer which is bound, grouped by geometry for instancing.

        Parameters
        ------------
        rows : (k,) int
          Rows of `self.table` to draw.
        model_view : (n, 4, 4) float
          Transform of every row of `self.table`.
        projection : (4, 4) float
          After `model_view` to clip space.
        frame : (1,) shaders.FRAME
          This frame's values, which are left unchanged.
        """
        table = self.table
        context = self.context
        # nodes scaled to nothing aren't drawn
        rows = rows[np.abs(model_view[rows, :3, :3]).max(axis=(1, 2)) > 0.0]
        order = rows[np.argsort(table.geometry_index[rows], kind="stable")]
        # a draw call starts wherever the geometry changes, if anywhere
        starts = np.flatnonzero(np.diff(table.geometry_index[order], prepend=-1) != 0)
        ends = np.append(starts[1:], len(order))
        instances = np.zeros(len(order), dtype=shaders.INSTANCE)
        instances["model_view"] = model_view[order].transpose((0, 2, 1))
        instances["joint_offset"] = table.joint_offset[order]

        depth_frame = frame.copy()
        depth_frame["projection"] = projection
        opengl.upload_array(gl.GL_UNIFORM_BUFFER, context.frame_buffer, depth_frame)
        gl.glUseProgram(context.shadow_program.program)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        for start, end in zip(starts, ends):
            item = table.buffers[order[start]]
            gl.glBindVertexArray(item.vertex_array)
            opengl.upload_array(
                gl.GL_ARRAY_BUFFER, item.instance_buffer, instances[start:end]
            )
            gl.glDrawElementsInstanced(
                gl.GL_TRIANGLES, item.count, gl.GL_UNSIGNED_INT, None, end - start
            )

    def draw_ambient_occlusion(self, frame: np.ndarray, model_view: np.ndarray) -> int:
        """
        Work out how much of the environment nearby surfaces hide from
        every pixel at half resolution: draw the scene's depth, look
        around every pixel in it, then blur the noise away.

        Parameters
        ------------
        frame : (1,) shaders.FRAME
          This frame's values, which are left unchanged.
        model_view : (n, 4, 4) float
          Transform of every row of `self.table` to the camera.

        Returns
        ------------
        occlusion : int
          Texture of the visible fraction of the environment.
        """
        table = self.table
        context = self.context
        occlusion = self.occlusion
        gl.glViewport(0, 0, occlusion.width, occlusion.height)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

        # the depth of every opaque mesh, and the floor
        opaque = table.mesh & ~table.blended
        opaque &= (np.arange(len(table.buffers)) < len(table.nodes)) | table.catcher
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, occlusion.framebuffers["depth"])
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        self.draw_depth(np.flatnonzero(opaque), model_view, frame["projection"][0], frame)

        # how much of the sky every pixel can see with noise, then without
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glBindVertexArray(context.empty_vertex_array)
        for program, source, unit, target in (
            (context.occlusion_program, "depth", "scene_depth", "noisy"),
            (context.blur_program, "noisy", "noisy_occlusion", "blurred"),
        ):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, occlusion.framebuffers[target])
            gl.glUseProgram(program.program)
            gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[unit])
            gl.glBindTexture(gl.GL_TEXTURE_2D, occlusion.textures[source])
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        return occlusion.textures["blurred"]

    def bind_geometry(self, buffers, previous) -> opengl.ShaderProgram:
        """
        Set up the program, material, textures and render state to
        draw one piece of geometry, skipping anything which is the
        same as the geometry drawn before it.

        Parameters
        ------------
        buffers : opengl.GeometryBuffers
          Geometry about to be drawn.
        previous : opengl.GeometryBuffers or None
          Geometry drawn just before this, if any.

        Returns
        ------------
        program : opengl.ShaderProgram
          The program now in use.
        """
        context = self.context
        state = self.state
        gl.glBindVertexArray(buffers.vertex_array)
        changed_program = previous is None or (previous.kind == "mesh") != (
            buffers.kind == "mesh"
        )

        if buffers.kind != "mesh":
            program = context.line_program
            if changed_program:
                gl.glUseProgram(program.program)
                gl.glDisable(gl.GL_CULL_FACE)
                gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            program.set("points", buffers.kind == "points")
            if buffers.kind == "points":
                width = state.point_size
                if width is None:
                    width = self.studio.point_size
            else:
                width = state.line_width
                if width is None:
                    width = self.studio.line_width
            program.set("line_width", width)
            return program

        program = context.mesh_program
        if changed_program:
            gl.glUseProgram(program.program)
            program.set("smooth_shading", state.smooth)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            if state.wireframe:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            else:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if changed_program or previous.material_key != buffers.material_key:
            material = buffers.material
            for uniform, name in shaders.MATERIAL_UNIFORMS.items():
                program.set(uniform, getattr(material, name))
            program.set("has_normal_texture", "normal_texture" in buffers.textures)
            for name in shaders.MATERIAL_TEXTURES:
                gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[name])
                gl.glBindTexture(
                    gl.GL_TEXTURE_2D, buffers.textures.get(name, context.white_texture)
                )
            if state.cull and not material.double_sided:
                gl.glEnable(gl.GL_CULL_FACE)
            else:
                gl.glDisable(gl.GL_CULL_FACE)

        program.set("has_face_colors", buffers.face_colors != 0)
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["face_colors"])
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, buffers.face_colors)
        return program

    def render(self, changed: bool = True) -> None:
        """
        Draw the scene into `self.resolve_framebuffer` without reading
        it back, so it can be read without waiting.

        Parameters
        ------------
        changed : bool
          Check if geometry or which nodes draw it changed since the
          last render, rather than only the matrices of the scene graph
          which are read every frame anyway.
        """
        self.window.switch_to()
        if changed and self.sync_geometry():
            self.state.scene_changed = True
        self.draw()
        self.resolve()

    def resolve(self) -> None:
        """
        Copy the multisampled frame into `self.resolve_framebuffer` and
        add bloom to it: copy it at half size, blur what the tone mapper
        compressed across and then down a mip level of that, and add the
        blur back scaled up.
        """
        opengl.copy_framebuffer(self.framebuffer, self.resolve_framebuffer.framebuffer)
        bloom = self.bloom
        if bloom is None:
            return
        context = self.context
        resolved = self.resolve_framebuffer
        textures, framebuffers = bloom.textures, bloom.framebuffers
        # at exactly half size linear filtering averages every 2x2 block
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, resolved.framebuffer)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, framebuffers["picture"])
        gl.glBlitFramebuffer(
            *(0, 0, 2 * bloom.width, 2 * bloom.height),
            *(0, 0, bloom.width, bloom.height),
            gl.GL_COLOR_BUFFER_BIT,
            gl.GL_LINEAR,
        )
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["bloom_source"])
        gl.glBindTexture(gl.GL_TEXTURE_2D, textures["picture"])
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        program = context.bloom_program
        gl.glUseProgram(program.program)
        gl.glBindVertexArray(context.empty_vertex_array)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glViewport(0, 0, bloom.blur_width, bloom.blur_height)
        for source, target, direction in (
            ("picture", "across", (1.0, 0.0)),
            ("across", "blurred", (0.0, 1.0)),
        ):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffers[target])
            gl.glBindTexture(gl.GL_TEXTURE_2D, textures[source])
            program.set("bright_pass", source == "picture")
            program.set("blur_direction", direction)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        # added onto the picture
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, resolved.framebuffer)
        gl.glViewport(0, 0, resolved.width, resolved.height)
        gl.glBindTexture(gl.GL_TEXTURE_2D, textures["blurred"])
        program.set("blur_direction", (0.0, 0.0))
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def to_array(
        self, include_depth: bool = False, include_segmentation: bool = False
    ) -> RenderArrays:
        """
        Render the scene and read back the image, and if asked what is
        under the center of every pixel, which is one more draw for both.

        Parameters
        ------------
        include_depth : bool
          Also read the distance along the view axis of every pixel.
        include_segmentation : bool
          Also read which node of `scene.graph.nodes_geometry` every
          pixel shows.

        Returns
        ------------
        arrays : RenderArrays
          The image, and the depth and segmentation or None.
        """
        self.render()
        color = opengl.read_framebuffer(self.resolve_framebuffer)
        if not (include_depth or include_segmentation):
            return RenderArrays(color=color, depth=None, segmentation=None)
        # one unantialiased draw writes the depth and node of every pixel
        target = opengl.create_framebuffer(
            self.framebuffer.width,
            self.framebuffer.height,
            samples=0,
            storage=gl.GL_RG32F,
        )
        try:
            self.draw(target)
            values = opengl.read_framebuffer(target)
        finally:
            target.delete()
        depth, segmentation = None, None
        if include_depth:
            depth = values[..., 0]
        if include_segmentation:
            segmentation = values[..., 1].astype(np.int64)
        return RenderArrays(color=color, depth=depth, segmentation=segmentation)

    def save_image(self, file_obj):
        """
        Render the scene and save it as a PNG.

        Parameters
        ------------
        file_obj : str or file-like
          Where to save the PNG.

        Returns
        ------------
        file_obj : str or file-like
          The passed file object.
        """
        from PIL import Image

        Image.fromarray(self.to_array().color).save(file_obj, format="png")
        return file_obj

    def on_draw(self) -> bool:
        """
        Draw the scene if anything changed and copy it to the window.

        Returns
        ------------
        drawn : bool
          If the scene was drawn rather than just copied.
        """
        self.window.switch_to()
        drawn = self.state.dirty or self.state.scene_changed
        if drawn:
            # uploading geometry is slow once, which isn't drawing speed,
            # and once the quality is chosen there is nothing to time
            timed = not (self.state.scene_changed or self.chose_fast or self.state.fast)
            start = time.perf_counter()
            self.draw()
            if timed:
                # wait for the GPU so the time is how long drawing really takes
                gl.glFinish()
                self.draw_seconds.append(time.perf_counter() - start)
                self.check_speed()
            self.resolve()
        opengl.copy_framebuffer(self.resolve_framebuffer, 0)
        return drawn

    def check_speed(self) -> None:
        """
        Draw fast from now on if, a second after starting, the last
        second of drawing averaged slower than `SLOW_FPS`, once and
        never after the user chose a quality themselves.
        """
        if self.chose_fast or self.state.fast:
            return
        if time.perf_counter() - self.started < 1.0:
            return
        seconds = np.cumsum(self.draw_seconds[::-1])
        # how many of the most recent draws it took to fill a second
        count = int(np.searchsorted(seconds, 1.0)) + 1
        if count <= len(seconds) and count < SLOW_FPS:
            self.state.fast = True
            self.chose_fast = True
            util.log.info(
                "drawing slowly so shadows and occlusion are off, `v` restores them"
            )
        # only the last few seconds matter
        self.draw_seconds = self.draw_seconds[-200:]

    def on_resize(self, width: int, height: int) -> bool:
        """
        Match the framebuffers and camera to a new window size.

        Parameters
        ------------
        width, height : int
          Window size in pixels.

        Returns
        ------------
        handled : bool
          True, so pyglet's own handler doesn't change any state.
        """
        self.window.switch_to()
        size = self.window.get_framebuffer_size()
        if (self.framebuffer.width, self.framebuffer.height) != tuple(size):
            self.allocate(size)
        self.state.trackball.resize(np.array([width, height]))
        self.state.dirty = True
        # stop pyglet's default handler from changing any OpenGL state
        return pyglet.event.EVENT_HANDLED

    def on_mouse_press(self, x, y, buttons, modifiers) -> None:
        """
        Start a trackball motion picked by the mouse button and keys.

        Parameters
        ------------
        x, y : int
          Mouse position in window pixels.
        buttons : int
          Pressed mouse buttons.
        modifiers : int
          Held modifier keys.
        """
        trackball = self.state.trackball
        ctrl = modifiers & pyglet.window.key.MOD_CTRL
        shift = modifiers & pyglet.window.key.MOD_SHIFT
        if buttons == pyglet.window.mouse.LEFT:
            if ctrl and shift:
                trackball.set_state(Trackball.STATE_ZOOM)
            elif shift:
                trackball.set_state(Trackball.STATE_ROLL)
            elif ctrl:
                trackball.set_state(Trackball.STATE_PAN)
            else:
                trackball.set_state(Trackball.STATE_ROTATE)
        elif buttons == pyglet.window.mouse.MIDDLE:
            trackball.set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.RIGHT:
            trackball.set_state(Trackball.STATE_ZOOM)
        trackball.down(np.array([x, y]))
        self.camera_transform = trackball.pose
        self.state.dirty = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        """
        Move the camera with the trackball.

        Parameters
        ------------
        x, y : int
          Mouse position in window pixels.
        dx, dy : int
          Mouse motion since the last event.
        buttons : int
          Pressed mouse buttons.
        modifiers : int
          Held modifier keys.
        """
        self.state.trackball.drag(np.array([x, y]))
        self.camera_transform = self.state.trackball.pose
        self.state.dirty = True

    def on_mouse_scroll(self, x, y, dx, dy) -> None:
        """
        Zoom the camera.

        Parameters
        ------------
        x, y : int
          Mouse position in window pixels.
        dx, dy : float
          Horizontal and vertical scroll.
        """
        camera = self.camera
        if camera.magnification is None:
            self.state.trackball.scroll(dy)
            self.camera_transform = self.state.trackball.pose
        else:
            # moving an orthographic camera changes nothing so zoom it
            camera.magnification = camera.magnification * 0.9**dy
        self.state.dirty = True

    def on_key_press(self, symbol, modifiers) -> None:
        """
        Run the method bound to a key in `KEY_BINDINGS`, or nudge
        the camera with the arrow keys.

        Parameters
        ------------
        symbol : int
          The pressed key.
        modifiers : int
          Held modifier keys.
        """
        key = pyglet.window.key
        switches = {item.name for item in fields(ViewerState)}
        for letter, _, name in KEY_BINDINGS:
            if symbol != getattr(key, letter.upper()):
                continue
            if name in switches:
                self.toggle(name)
            else:
                getattr(self, name)()
            return
        nudge = {
            key.LEFT: [-10, 0],
            key.RIGHT: [10, 0],
            key.DOWN: [0, -10],
            key.UP: [0, 10],
        }
        if symbol in nudge:
            trackball = self.state.trackball
            trackball.down([0, 0])
            trackball.drag(nudge[symbol])
            self.camera_transform = trackball.pose
            self.state.dirty = True

    def on_callback(self, elapsed: float) -> None:
        """
        Run the user callback and draw whatever it changed.

        Parameters
        ------------
        elapsed : float
          Seconds since the last call.
        """
        self.callback(self.scene)
        self.state.scene_changed = True

    def on_animate(self, elapsed: float) -> None:
        """
        Advance `scene.animations` by the time since the last tick,
        looping back to the start.

        Parameters
        ------------
        elapsed : float
          Seconds since the last tick.
        """
        if not self.state.playing:
            return
        self.animation_time += elapsed
        if self.scene.duration > 0.0:
            self.animation_time %= self.scene.duration
        self.scene.animate(self.animation_time)
        self.state.dirty = True


def hidden_context() -> opengl.RenderContext:
    """
    Get the hidden window used for every offscreen render, creating
    it the first time.

    Returns
    ------------
    context : opengl.RenderContext
      A hidden window with compiled shaders.
    """
    try:
        return _HIDDEN_CONTEXT["context"]
    except KeyError:
        context = opengl.create_context(64, 64, visible=False, resizable=False)
        # it's the library's and not one of the user's windows, which
        # would keep `pyglet.app.run` from ever returning
        pyglet.app.windows.discard(context.window)
        _HIDDEN_CONTEXT["context"] = context
        return context
