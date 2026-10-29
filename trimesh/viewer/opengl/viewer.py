"""
viewer.py
-----------

An OpenGL 3.3 viewer and offscreen renderer for `trimesh.Scene`, using
raw OpenGL calls from pyglet 1.5, 2 or 3.
"""

import ctypes
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, fields, replace
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pyglet

from ... import util
from ...bounds import corners
from ...scene import Scene
from ...scene.cameras import DEFAULT_FOV, current_view, projection_matrices
from ...scene.lighting import (
    HARMONICS,
    DirectionalLight,
    SpotLight,
    prefilter,
    spherical_harmonics,
)
from ...transformations import adjugate
from ...typed import ArrayLike, Integer, Iterable, NDArray, Sequence
from ...visual import to_rgba
from ..conversion import (
    OCCLUSION_RADIUS,
    RENDER_DEFAULTS,
    GeometryArrays,
    geometry_to_arrays,
    render_lights,
)
from ..trackball import Trackball
from . import shaders

# headless Linux uses EGL, which must be set before importing `gl`
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pyglet.options["headless"] = True
# per-call error checks made drawing 400 meshes 50% slower
pyglet.options["debug_gl"] = False

CONTEXT_HINT = (
    "the trimesh viewer needs OpenGL 3.3 core, which without a display EGL provides "
    + "with `PYGLET_HEADLESS=1` (i.e. `apt install libegl1`), or run under `xvfb-run`"
)

try:
    try:
        # pyglet >= 3
        from pyglet.graphics.api import gl
        from pyglet.graphics.api.gl.base import ContextException
    except ImportError:
        # pyglet < 3
        from pyglet import gl
        from pyglet.gl import ContextException
except Exception as error:
    # pyglet 2 opens a window with an OpenGL 3.3 context on import
    raise RuntimeError(f"unable to create an OpenGL context: {CONTEXT_HINT}") from error

# pyglet 3.0.dev11 raises IndexError without a matching EGL config
# and ImportError without EGL
CONTEXT_ERRORS = (
    pyglet.window.NoSuchConfigException,
    ContextException,
    IndexError,
    ImportError,
)

DEPTH_FORMAT = gl.GL_DEPTH_COMPONENT24
# images a frame is drawn into, each with its own framebuffer, as name :
# (color or 0, depth or 0, size key of `allocate`, filter, layers, samples)
TARGETS = {
    "multisample": (gl.GL_RGBA8, DEPTH_FORMAT, "full", gl.GL_NEAREST, 0, 4),
    # read back and shown
    "resolved": (gl.GL_RGBA8, 0, "full", gl.GL_NEAREST, 0, 0),
    # view depth and node index per pixel
    "index": (gl.GL_RG32F, DEPTH_FORMAT, "full", gl.GL_NEAREST, 0, 0),
    # linear filtering softens the depth comparisons
    "shadow": (0, DEPTH_FORMAT, "shadow", gl.GL_LINEAR, shaders.MAX_SHADOWS, 0),
    # ambient occlusion depth, noisy, then blurred
    "scene_depth": (0, DEPTH_FORMAT, "half", gl.GL_NEAREST, 0, 0),
    "noisy": (gl.GL_R8, 0, "half", gl.GL_NEAREST, 0, 0),
    "occlusion": (gl.GL_R8, 0, "half", gl.GL_LINEAR, 0, 0),
    # bloom blurred across squares of `resolved` averaged, then down
    "across": (gl.GL_RGBA16F, 0, "bloom", gl.GL_LINEAR, 0, 0),
    "glow": (gl.GL_RGBA16F, 0, "bloom", gl.GL_LINEAR, 0, 0),
}

SHADOW_RESOLUTION = 2048
# bloom standard deviation per picture height: one degree of the default FOV
BLOOM_SPREAD = 1.0 / DEFAULT_FOV[1]
# fewest texels per bloom standard deviation: a Gaussian's spectrum at
# Nyquist is then `exp(-2 * pi^2)` of its peak, so there is no aliasing
BLOOM_TEXELS = 2.0
# uniform buffer binding of each of `shaders.BLOCKS`
# pyglet reserves binding 0 for its `WindowBlock`, rebound on every resize
BLOCK_BINDINGS = {name: index for index, name in enumerate(shaders.BLOCKS, start=1)}
# uniforms outside blocks besides `shaders.TEXTURE_UNITS`
UNIFORMS = ("pass_index", "points", "model_view", "node_index")
# color storage : (format, type, channels, numpy dtype)
READ_FORMATS = {
    gl.GL_RGBA8: (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 4, np.uint8),
    gl.GL_RG32F: (gl.GL_RG, gl.GL_FLOAT, 2, np.float32),
}
# kind : (create, free) of OpenGL names other than programs
GENERATORS = {
    "buffers": (gl.glGenBuffers, gl.glDeleteBuffers),
    "textures": (gl.glGenTextures, gl.glDeleteTextures),
    "framebuffers": (gl.glGenFramebuffers, gl.glDeleteFramebuffers),
    "vertex_arrays": (gl.glGenVertexArrays, gl.glDeleteVertexArrays),
}
# disable or enable an OpenGL capability by a bool
SWITCHES = (gl.glDisable, gl.glEnable)

IDENTITY = np.eye(4)

# (input, action) for `print_help`
MOUSE_BINDINGS = (
    *(("drag", "rotate"), ("ctrl + drag", "pan"), ("shift + drag", "roll")),
    *(("wheel", "zoom"), ("arrows", "nudge")),
)
# (key, action, `ViewerState` switch or method of the viewer or window)
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
    ("o", "toggle shadows, ambient occlusion and bloom", "quality"),
    ("h", "print help", "print_help"),
)

# deprecated methods raising after April 2028 : switch toggled or None
LEGACY_METHODS = {
    "toggle_culling": "cull",
    "toggle_wireframe": "wireframe",
    "toggle_fullscreen": "fullscreen",
    "toggle_axis": "axis",
    "toggle_grid": "grid",
    "add_geometry": None,
    "cleanup_geometries": None,
    "update_flags": None,
    "init_gl": None,
}
# deprecated keywords: `window_conf` is `config`, `line_settings` holds
# `line_width` and `point_size`, and the rest do nothing
LEGACY_KEYWORDS = ("window_conf", "line_settings", "resizable", "offset_lines")
LEGACY_KEYWORDS += ("profile", "record")

# one hidden window shared by every offscreen render
_HIDDEN_CONTEXT = {}


class RenderContext(NamedTuple):
    """
    A window with an OpenGL context and what its viewers share.
    """

    window: pyglet.window.Window
    # programs by name, and uniform locations by program or -1 if unused
    programs: dict
    uniforms: dict
    # bound in place of missing textures
    white_texture: int
    # for draws without vertex inputs
    empty_vertex_array: int
    # uniform buffer of `shaders.FRAME`
    frame_buffer: int
    # OpenGL names by kind, for `delete`
    names: dict


class GeometryBuffers(NamedTuple):
    """
    One piece of geometry on the GPU.
    """

    kind: str
    vertex_array: int
    # indices for meshes, segments for lines and points
    count: int
    # `shaders.MATERIAL` record and its uniform buffer
    material: NDArray
    material_buffer: int
    # `shaders.MATERIAL_TEXTURES` name : texture
    textures: dict
    # mesh only: face color buffer texture or zero
    face_colors: int
    # mesh only: instance rows, rewritten every draw
    instance_buffer: int
    # vertex centroid and bounding box corners, relative to `origin`
    centroid: NDArray
    corners: NDArray
    origin: NDArray
    skinned: bool
    # OpenGL names by kind, for `delete`
    names: dict


class RenderArrays(NamedTuple):
    """
    Images read by `SceneViewer.to_array`, with the first row at the top.
    """

    # (height, width, 4) uint8
    color: NDArray
    # (height, width) float32 view depth, zero for background, or None
    depth: NDArray | None
    # (height, width) int64 index into `scene.graph.nodes_geometry`,
    # -1 for background, or None
    segmentation: NDArray | None


@dataclass
class ViewerState:
    """
    Viewer switches the user can toggle.
    """

    # set once the window exists
    trackball: Trackball | None = None
    smooth: bool = True
    wireframe: bool = False
    cull: bool = True
    fullscreen: bool = False
    axis: bool = False
    grid: bool = False
    # scene graph nodes not drawn
    hidden: set = field(default_factory=set)
    # geometry names which don't move with the camera
    fixed: Iterable = field(default_factory=set)
    playing: bool = True
    # follow the scene camera's node, as renders do
    follow_camera: bool = False
    # shadows, ambient occlusion and bloom, off in a window to draw fast
    quality: bool = True
    # needs a redraw
    dirty: bool = True
    # geometry or what draws it may have changed
    scene_changed: bool = True


def generate(names: dict, kind: str) -> int:
    """
    Create an OpenGL object, recorded for `delete`.

    Parameters
    ------------
    names : dict
      {kind : [name, ...]} to record it in.
    kind : str
      Key of `GENERATORS`.

    Returns
    ------------
    name : int
      New OpenGL name.
    """
    value = gl.GLuint()
    GENERATORS[kind][0](1, ctypes.byref(value))
    names.setdefault(kind, []).append(value.value)
    return value.value


def delete(names: dict) -> None:
    """
    Free OpenGL objects on the current context.

    Parameters
    ------------
    names : dict
      {kind of `GENERATORS` or "programs" : [name, ...]}
    """
    # programs in use aren't deleted until unbound
    gl.glUseProgram(0)
    for program in names.get("programs", []):
        gl.glDeleteProgram(program)
    for kind, values in names.items():
        if kind in GENERATORS:
            GENERATORS[kind][1](len(values), (gl.GLuint * len(values))(*values))


def upload(
    target: int, buffer: int, data: NDArray, usage: int = gl.GL_STREAM_DRAW
) -> int:
    """
    Replace the contents of an OpenGL buffer with an array.

    Parameters
    ------------
    target : int
      Buffer type, i.e. `GL_ARRAY_BUFFER`, left bound.
    buffer : int
      OpenGL buffer.
    data : numpy.ndarray
      Array to copy.
    usage : int
      Usage hint, i.e. `GL_STATIC_DRAW`.

    Returns
    ------------
    buffer : int
      OpenGL buffer.
    """
    data = np.ascontiguousarray(data)
    gl.glBindBuffer(target, buffer)
    gl.glBufferData(target, data.nbytes, data.ctypes.data, usage)
    return buffer


def upload_image(names: dict, pixels: NDArray, srgb: bool) -> int:
    """
    Copy an RGBA image into a mipmapped, repeating texture.

    Parameters
    ------------
    names : dict
      Where to record the texture for `delete`.
    pixels : (h, w, 4) uint8
      Image with the first row at the top.
    srgb : bool
      Decode sRGB to linear when sampled.

    Returns
    ------------
    texture : int
      OpenGL texture.
    """
    # trimesh UV coordinates have the origin at the bottom left
    pixels = np.ascontiguousarray(pixels[::-1], dtype=np.uint8)
    texture = generate(names, "textures")
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, (gl.GL_RGBA8, gl.GL_SRGB8_ALPHA8)[srgb], pixels.shape[1],
        pixels.shape[0], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels.ctypes.data,
    )  # fmt: skip
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    # magnifying is linear by default
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR
    )
    return texture


def upload_geometry(arrays: GeometryArrays) -> GeometryBuffers:
    """
    Copy converted geometry onto the GPU.

    Parameters
    ------------
    arrays : GeometryArrays
      From `conversion.geometry_to_arrays`.

    Returns
    ------------
    buffers : GeometryBuffers
      Geometry on the GPU.
    """
    names = {}
    vertex_array = generate(names, "vertex_arrays")
    gl.glBindVertexArray(vertex_array)
    rows, count, instance_buffer = arrays.rows, len(arrays.rows), 0
    # an instance per segment, or per node drawing a mesh
    program, inputs = "line", [(rows, 1)]
    if arrays.kind == "mesh":
        points, count, program = rows["position"], arrays.faces.size, "mesh"
        inputs = [(rows, 0), (np.zeros(1, dtype=shaders.INSTANCE), 1), (arrays.skin, 0)]
        upload(gl.GL_ELEMENT_ARRAY_BUFFER, generate(names, "buffers"), arrays.faces)
    else:
        points = np.concatenate((rows["segment_start"], rows["segment_end"]))
    locations = shaders.attribute_locations(*shaders.INPUTS[program])
    for data, divisor in inputs:
        if data is None:
            continue
        buffer = upload(gl.GL_ARRAY_BUFFER, generate(names, "buffers"), data)
        if data.dtype == shaders.INSTANCE:
            instance_buffer = buffer
        # one input per field or matrix column, with bytes normalized
        for name in data.dtype.names:
            item, offset = data.dtype.fields[name][:2]
            columns, width = (1, 1, *item.shape)[-2:]
            normalized = item.base.itemsize == 1
            for column in range(columns):
                location = locations[name] + column
                gl.glEnableVertexAttribArray(location)
                gl.glVertexAttribPointer(
                    location, width, (gl.GL_FLOAT, gl.GL_UNSIGNED_BYTE)[normalized],
                    normalized, data.dtype.itemsize,
                    ctypes.c_void_p(offset + column * width * item.base.itemsize),
                )  # fmt: skip
                gl.glVertexAttribDivisor(location, divisor)
    gl.glBindVertexArray(0)

    face_colors = 0
    if arrays.face_colors is not None:
        # a buffer texture read by `gl_PrimitiveID`
        buffer = upload(
            gl.GL_TEXTURE_BUFFER, generate(names, "buffers"), arrays.face_colors
        )
        face_colors = generate(names, "textures")
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, face_colors)
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA8, buffer)
    return GeometryBuffers(
        kind=arrays.kind,
        vertex_array=vertex_array,
        count=count,
        material=arrays.material,
        material_buffer=upload(
            gl.GL_UNIFORM_BUFFER, generate(names, "buffers"), arrays.material
        ),
        textures={
            name: upload_image(
                names,
                np.asarray(image.convert("RGBA")),
                srgb=shaders.MATERIAL_TEXTURES[name][1],
            )
            for name, image in arrays.textures.items()
        },
        face_colors=face_colors,
        instance_buffer=instance_buffer,
        origin=arrays.origin,
        centroid=points.mean(axis=0),
        corners=corners([points.min(axis=0), points.max(axis=0)]),
        skinned=arrays.skin is not None,
        names=names,
    )


def create_context(visible: bool, config=None, **options) -> RenderContext:
    """
    Open a window with an OpenGL 3.3 core context and compile its programs.

    Parameters
    ------------
    visible : bool
      Show the window.
    config : None or pyglet config
      Override the context configuration.
    options : dict
      Passed to `pyglet.window.Window`.

    Returns
    ------------
    context : RenderContext
      Window and its shared GPU objects.
    """
    if config is None and hasattr(gl, "Config"):
        # pyglet >= 3 has no `gl.Config` and defaults to 3.3 core
        config = gl.Config(double_buffer=True)
    # macOS and pyglet 1.5 need a 3.3 core context requested explicitly
    versions = {"major_version": 3, "minor_version": 3, "forward_compatible": True}
    for name, value in versions.items():
        if config is not None and getattr(config, name, None) is None:
            setattr(config, name, value)
    try:
        window = pyglet.window.Window(config=config, visible=visible, **options)
    except CONTEXT_ERRORS:
        try:
            # compatibility contexts on Linux and Windows also support 3.3
            window = pyglet.window.Window(visible=visible, **options)
        except CONTEXT_ERRORS as error:
            message = f"unable to create an OpenGL context: {CONTEXT_HINT}"
            raise RuntimeError(message) from error
    names = {}
    try:
        window.switch_to()
        # version strings start with "major.minor"
        version, renderer = (
            ctypes.cast(gl.glGetString(name), ctypes.c_char_p).value.decode("utf-8")
            for name in (gl.GL_VERSION, gl.GL_RENDERER)
        )
        if tuple(int(part) for part in version.split(" ")[0].split(".")[:2]) < (3, 3):
            raise RuntimeError(
                f"`{renderer}` only has OpenGL `{version}`: {CONTEXT_HINT}"
            )

        programs, uniforms, log = {}, {}, ctypes.create_string_buffer(4096)
        for name in shaders.PROGRAMS:
            program, logs = gl.glCreateProgram(), []
            names.setdefault("programs", []).append(program)
            for stage, kind in (
                ("vert", gl.GL_VERTEX_SHADER),
                ("frag", gl.GL_FRAGMENT_SHADER),
            ):
                source = shaders.shader_source(f"{name}.{stage}").encode("utf-8")
                pointer = ctypes.cast(
                    (ctypes.c_char_p * 1)(source),
                    ctypes.POINTER(ctypes.POINTER(gl.GLchar)),
                )
                shader = gl.glCreateShader(kind)
                gl.glShaderSource(shader, 1, pointer, None)
                gl.glCompileShader(shader)
                gl.glGetShaderInfoLog(shader, len(log), None, log)
                logs.append(f"{name}.{stage}: {log.value.decode('utf-8', 'replace')}")
                gl.glAttachShader(program, shader)
                # the shader is freed with the program
                gl.glDeleteShader(shader)
            gl.glLinkProgram(program)
            status = gl.GLint()
            gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
            if not status.value:
                gl.glGetProgramInfoLog(program, len(log), None, log)
                logs.append(log.value.decode("utf-8", "replace"))
                raise ValueError(f"`{name}` failed to compile:\n" + "\n".join(logs))
            # bind each block the program uses
            for block_name, binding in BLOCK_BINDINGS.items():
                block = gl.glGetUniformBlockIndex(program, block_name.encode("utf-8"))
                if block != gl.GL_INVALID_INDEX:
                    gl.glUniformBlockBinding(program, block, binding)
            programs[name] = program
            uniforms[name] = {
                uniform: gl.glGetUniformLocation(program, uniform.encode("utf-8"))
                for uniform in (*UNIFORMS, *shaders.TEXTURE_UNITS)
            }
            # samplers use fixed texture units
            gl.glUseProgram(program)
            for sampler, unit in shaders.TEXTURE_UNITS.items():
                gl.glUniform1i(uniforms[name][sampler], unit)
        gl.glUseProgram(0)
        # filter the environment across cube map faces
        gl.glEnable(gl.GL_TEXTURE_CUBE_MAP_SEAMLESS)

        # every program shares one `shaders.FRAME` binding
        frame = np.zeros(1, dtype=shaders.FRAME)
        frame_buffer = upload(gl.GL_UNIFORM_BUFFER, generate(names, "buffers"), frame)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, BLOCK_BINDINGS["Frame"], frame_buffer)
        return RenderContext(
            window=window,
            programs=programs,
            uniforms=uniforms,
            white_texture=upload_image(names, np.full((1, 1, 4), 255), srgb=False),
            empty_vertex_array=generate(names, "vertex_arrays"),
            frame_buffer=frame_buffer,
            names=names,
        )
    except BaseException:
        # don't leave a window open on failure
        window.close()
        raise


def hidden_context() -> RenderContext:
    """
    Hidden window shared by every offscreen render, created once.

    Returns
    ------------
    context : RenderContext
      Hidden window with compiled shaders.
    """
    try:
        return _HIDDEN_CONTEXT["context"]
    except KeyError:
        context = create_context(visible=False, width=64, height=64, resizable=False)
        # so `pyglet.app.run` doesn't wait for it
        pyglet.app.windows.discard(context.window)
        _HIDDEN_CONTEXT["context"] = context
        return context


class SceneViewer:
    """
    An interactive window for a `trimesh.Scene`.

    Draws a copy of the scene camera as `camera` at `camera_transform`.
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
          Show a window drawing fast until `o`, or render offscreen with quality.
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
          Starting values for `ViewerState`.
        settings : dict
          `RENDER_DEFAULTS` or `ViewerState` values, i.e. `exposure=2.0`.
        """
        legacy = [key for key in LEGACY_KEYWORDS if key in settings]
        if len(legacy) > 0:
            message = f"`SceneViewer` keywords {legacy} will raise after April 2028"
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            config = settings.pop("window_conf", config)
            settings.update(settings.pop("line_settings", None) or {})
            for key in legacy:
                settings.pop(key, None)
        if not isinstance(scene, Scene):
            scene = Scene(scene)
        self.scene, self.callback = scene, callback
        self.closed = False
        # never stored in the scene, and unknown names raise
        self.render_settings = {
            key: settings.pop(key, value) for key, value in RENDER_DEFAULTS.items()
        }
        # a window draws fast until `o`, a render with shadows and occlusion
        defaults = ViewerState(quality=not visible)
        self.state = replace(defaults, **{**(flags or {}), **settings})
        self.background = np.ones(4)
        if background is not None:
            self.background = to_rgba(background) / 255.0

        # a copy of the scene camera
        self.camera, camera_transform = current_view(scene)
        self.place_camera(camera_transform)
        self.initial_camera = self.camera_transform.copy()
        if resolution is None:
            resolution = self.camera.resolution
        if visible:
            self.context = create_context(
                visible=True,
                config=config,
                width=int(resolution[0]),
                height=int(resolution[1]),
                resizable=True,
                fullscreen=self.state.fullscreen,
                caption=caption or "trimesh SceneViewer (`h` for help)",
            )
            resolution = self.context.window.get_framebuffer_size()
        else:
            self.context = hidden_context()
        self.window = self.context.window
        self.window.switch_to()
        # uploaded geometry and decorations, their keys, and the graph last synced
        self.geometry_buffers, self.geometry_keys, self.synced = {}, {}, (None, None)

        # default lights and environment are fixed around the starting camera
        lights, transforms = render_lights(scene, self.initial_camera)
        # frame values that only change with the scene: lights, environment and settings
        frame = np.zeros(1, dtype=shaders.FRAME)
        placed = lights[:-1]
        if len(placed) > shaders.MAX_LIGHTS:
            util.log.warning(f"drawing {shaders.MAX_LIGHTS} of {len(placed)} lights")
            placed = placed[: shaders.MAX_LIGHTS]
        directional = np.array(
            [isinstance(item, DirectionalLight) for item in placed], bool
        )
        spot = np.array([isinstance(item, SpotLight) for item in placed], dtype=bool)
        intensity = np.array([item.intensity for item in placed], dtype=np.float64)
        # a missing range is NaN, which `nan_to_num` makes zero falloff
        radius = np.array([item.radius for item in placed], dtype=np.float64)
        inner = np.array([getattr(item, "innerConeAngle", 0.0) for item in placed])
        outer = np.array([getattr(item, "outerConeAngle", 0.0) for item in placed])
        colors = np.reshape([item.color[:3] for item in placed], (-1, 3)) / 255.0
        cone_scale = np.where(
            spot, 1.0 / np.maximum(np.cos(inner) - np.cos(outer), 1e-3), 0
        )
        casts = directional | spot
        rows = frame["lights"][0, : len(placed)]
        rows["position"][:, 3] = ~directional
        rows["color"][:, :3] = colors * intensity[:, None]
        rows["falloff"][:, 0] = np.where(directional, 0.0, np.nan_to_num(1.0 / radius))
        rows["falloff"][:, 1] = cone_scale
        rows["falloff"][:, 2] = np.where(spot, -np.cos(outer) * cone_scale, 1.0)
        rows["falloff"][:, 3] = np.where(spot, outer, 0.0)
        order = np.cumsum(casts) - 1
        rows["shadow"][:, 0] = np.where(casts & (order < shaders.MAX_SHADOWS), order, -1)
        frame["light_count"][0, 0] = len(placed)

        # world to environment rotation, and irradiance over pi premultiplied
        # by `HARMONICS` so the shader only multiplies the polynomials
        transforms = np.asarray(transforms)
        light, rotation = lights[-1], util.unitize(transforms[-1, :3, :3].T)
        tint, irradiance = light.intensity, spherical_harmonics(light.cubemap)
        frame["environment_rotation"][0, :3, :3] = rotation
        frame["environment_color"][0] = (
            tint,
            tint,
            tint,
            len(prefilter(light.cubemap)) - 1,
        )
        frame["irradiance"][0, :, :3] = irradiance * HARMONICS[:, None] * tint / np.pi
        # unshadowed brightness of a white floor facing +Y: environment at +Y
        # plus `C * I / pi` from each directional light
        facing = util.unitize(transforms[: len(placed), :3, 2]) @ rotation[1]
        received = (HARMONICS * [1, 1, 0, 0, 0, 0, -1, 0, -1]) @ irradiance * tint
        received += (np.clip(facing, 0.0, None) * directional * intensity) @ colors
        frame["ground_light"][0, 0] = received @ shaders.LUMINANCE / np.pi
        frame["exposure"][0, 0] = self.render_settings["exposure"]
        frame["line_widths"][0, :2] = (
            self.render_settings["line_width"],
            self.render_settings["point_size"],
        )
        self.frame_constants = frame
        self.lights, self.light_transforms = placed, transforms[: len(placed)]
        self.environment, self.environment_transform = lights[-1], transforms[-1]
        # shadow catcher floor level with the environment, eight times as
        # wide as the scene, counterclockwise from above
        self.ground, bounds = np.zeros((0, 3)), scene.bounds
        if self.render_settings["ground_shadow"] and bounds is not None:
            right, up, back = util.unitize(self.environment_transform[:3, :3].T)
            center = bounds.mean(axis=0)
            # just below the scene so they never touch
            height = (corners(bounds) @ up).min() - 1e-4 * scene.scale
            base = center + (height - np.dot(center, up)) * up
            square = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64)
            self.ground = base + 4.0 * scene.scale * square @ np.vstack((right, back))
        self.names = {}
        self.allocate(resolution)
        self.reset_view()

        if visible:
            # methods named like window events are handlers
            self.window.push_handlers(self)
        if callback is not None:
            period = callback_period or 1.0 / 30.0
            pyglet.clock.schedule_interval(self.on_callback, period)
        # seconds into `scene.animations`, or None
        self.animation_time = None
        if visible and len(scene.animations) > 0:
            self.animation_time = 0.0
            pyglet.clock.schedule_interval(self.on_animate, 1.0 / 60.0)
        if start_loop and visible:
            self.run()

    def __getattr__(self, name: str):
        """
        Deprecated `LEGACY_METHODS`, raising after April 2028.

        Parameters
        ------------
        name : str
          Attribute not otherwise found.

        Returns
        ------------
        method : function
          Toggles a switch or redraws.
        """
        if name not in LEGACY_METHODS:
            raise AttributeError(f"`SceneViewer` has no attribute `{name}`")
        message = f"`SceneViewer.{name}` will be removed after April 2028"
        warnings.warn(message, category=DeprecationWarning, stacklevel=2)
        switch = LEGACY_METHODS[name]

        def legacy(*args, **kwargs):
            if switch is None:
                self.state.scene_changed = True
            else:
                self.toggle(switch)

        return legacy

    def run(self) -> None:
        """
        Run the event loop until the window is closed.
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
                    # idle without spinning the CPU
                    time.sleep(1.0 / 60.0)
                window.flip()
        finally:
            # even if drawing raised
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
        for buffers in self.geometry_buffers.values():
            delete(buffers.names)
        delete(self.names)
        # the hidden window is shared and stays open
        if self.context is not _HIDDEN_CONTEXT.get("context"):
            delete(self.context.names)
            self.window.close()

    def allocate(self, size: Sequence[Integer] | NDArray) -> None:
        """
        Reallocate `TARGETS` and per frame GPU objects for a size.

        Parameters
        ------------
        size : (2,) int
          Width and height in pixels.
        """
        self.window.switch_to()
        delete(self.names)
        self.names, self.framebuffers, self.textures, self.sizes = {}, {}, {}, {}
        # pixels are square, keeping the camera's vertical field of view
        self.camera.fit_resolution(size)
        width, height = int(size[0]), int(size[1])
        # blur bloom over the largest `2**level` squares keeping `BLOOM_TEXELS`
        spread = BLOOM_SPREAD * height
        level = int(max(np.floor(np.log2(spread / BLOOM_TEXELS)), 1))
        bloom = self.render_settings["bloom"]
        self.frame_constants["bloom"][0, :3] = (bloom, spread / 2**level, level)
        self.frame_constants["viewport_size"][0, :2] = (width, height)
        sizes = {
            "full": (width, height),
            "half": (max(width // 2, 1), max(height // 2, 1)),
            "bloom": (max(width >> level, 1), max(height >> level, 1)),
            "shadow": (SHADOW_RESOLUTION, SHADOW_RESOLUTION),
        }
        limits = [gl.GLint(), gl.GLint()]
        gl.glGetIntegerv(gl.GL_MAX_COLOR_TEXTURE_SAMPLES, ctypes.byref(limits[0]))
        gl.glGetIntegerv(gl.GL_MAX_DEPTH_TEXTURE_SAMPLES, ctypes.byref(limits[1]))
        for name, (color, depth, scale, filtering, layers, samples) in TARGETS.items():
            self.sizes[name] = size = sizes[scale]
            samples = min(samples, *(limit.value for limit in limits))
            kind = gl.GL_TEXTURE_2D
            if samples > 0:
                kind = gl.GL_TEXTURE_2D_MULTISAMPLE
            elif layers > 0:
                kind = gl.GL_TEXTURE_2D_ARRAY
            self.framebuffers[name] = generate(self.names, "framebuffers")
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers[name])
            # color last, so passes read color if there is any
            for internal, pixels, attachment in (
                (depth, gl.GL_DEPTH_COMPONENT, gl.GL_DEPTH_ATTACHMENT),
                (color, gl.GL_RGBA, gl.GL_COLOR_ATTACHMENT0),
            ):
                if internal == 0:
                    continue
                self.textures[name] = texture = generate(self.names, "textures")
                gl.glBindTexture(kind, texture)
                if samples > 0:
                    gl.glTexImage2DMultisample(kind, samples, internal, *size, True)
                elif layers > 0:
                    gl.glTexImage3D(
                        kind, 0, internal, *size, layers, 0, pixels, gl.GL_FLOAT, None
                    )
                else:
                    gl.glTexImage2D(
                        kind, 0, internal, *size, 0, pixels, gl.GL_FLOAT, None
                    )
                # multisamples aren't sampled
                for parameter, value in (
                    (gl.GL_TEXTURE_MIN_FILTER, filtering),
                    (gl.GL_TEXTURE_MAG_FILTER, filtering),
                    (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                    (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE),
                ):
                    if samples == 0:
                        gl.glTexParameteri(kind, parameter, value)
                gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, attachment, texture, 0)
            if color == 0:
                # only depth is ever drawn
                gl.glDrawBuffer(gl.GL_NONE)
                gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        # depth comparisons, with a white border so outside a shadow map is lit
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.textures["shadow"])
        for parameter, value in (
            (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE),
            (gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL),
            (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER),
            (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, parameter, value)
        white = (gl.GLfloat * 4)(1, 1, 1, 1)
        gl.glTexParameterfv(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_BORDER_COLOR, white)
        # every skin's joint matrices, one row per texel
        self.joint_buffer = generate(self.names, "buffers")
        upload(gl.GL_TEXTURE_BUFFER, self.joint_buffer, np.eye(4, dtype=np.float32))
        self.textures["joint_matrices"] = generate(self.names, "textures")
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.textures["joint_matrices"])
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA32F, self.joint_buffer)
        # environment cube map with mip levels prefiltered per roughness
        self.textures["environment"] = generate(self.names, "textures")
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.textures["environment"])
        levels = prefilter(self.environment.cubemap)
        for level, faces in enumerate(levels):
            size = faces.shape[1]
            for face, pixels in enumerate(faces):
                gl.glTexImage2D(
                    gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, level, gl.GL_RGB16F,
                    size, size, 0, gl.GL_RGB, gl.GL_FLOAT, pixels.ctypes.data,
                )  # fmt: skip
        # magnifying is linear by default
        for parameter, value in (
            (gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
            (gl.GL_TEXTURE_MAX_LEVEL, len(levels) - 1),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, parameter, value)
        # upload joints and draw shadows again
        self.placed = None

    def place_camera(self, camera_transform: ArrayLike) -> None:
        """
        Move the camera to a camera node, keeping any scale in `camera_scale`.

        Parameters
        ------------
        camera_transform : (4, 4) float
          Camera to world, possibly scaled.
        """
        pose = np.array(camera_transform, dtype=np.float64)
        scale = np.linalg.norm(pose[:3, :3], axis=0)
        pose[:3, :3] /= scale
        self.camera_transform = pose
        # 3ds Max exports scaled cameras, which only scales their planes
        self.camera_scale = scale.mean()
        self.state.dirty = True

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
          New `ViewerState` values, i.e. `{"wireframe": True}`.
        """
        self.state = replace(self.state, **(flags or {}))
        # mouse events are in window pixels, not framebuffer pixels
        size, scene = np.array(self.window.get_size()), self.scene
        self.state.trackball = Trackball(
            self.initial_camera, size, scene.scale, scene.centroid
        )
        self.camera_transform = self.initial_camera.copy()
        self.state.dirty = True
        # the axis and grid are uploaded with the geometry
        self.state.scene_changed = self.state.scene_changed or flags is not None

    def toggle(self, name: str) -> None:
        """
        Flip a `ViewerState` switch, i.e. "wireframe".

        Parameters
        ------------
        name : str
          Boolean field of `ViewerState`.
        """
        state = self.state
        setattr(state, name, not getattr(state, name))
        state.dirty = True
        # the axis and grid are uploaded with the geometry
        state.scene_changed = state.scene_changed or name in ("axis", "grid")
        if name == "fullscreen":
            self.window.set_fullscreen(state.fullscreen)

    def sync_geometry(self) -> None:
        """
        Upload changed geometry and rebuild the draw table if needed.
        """
        from ...base import Trimesh
        from ...creation import axis
        from ...path.creation import grid

        scene, state, graph = self.scene, self.state, self.scene.graph
        # a key per geometry and decoration, with colors not in every hash
        keys = {}
        for name, geometry in scene.geometry.items():
            colors = getattr(geometry, "colors", None)
            if colors is not None:
                colors = hash(np.asarray(colors).tobytes())
            keys[name] = (hash(geometry), hash(getattr(geometry, "visual", None)), colors)
        shown = {("axis",): state.axis, ("grid",): state.grid}
        shown[("ground",)] = len(self.ground) > 0
        keys.update((name, name) for name, show in shown.items() if show)
        changed = keys != self.geometry_keys
        for name, key in keys.items():
            if self.geometry_keys.get(name) == key:
                continue
            if name in self.geometry_buffers:
                delete(self.geometry_buffers.pop(name).names)
            if name == ("axis",):
                # built at unit size as tiny cylinders collapse when merged
                marker = axis(origin_size=1.0)
                marker.apply_scale((scene.scale or 1.0) / 100.0)
                arrays = geometry_to_arrays(marker)
            elif name == ("grid",):
                # a grid just under the scene, level with the ground
                right, up, back = util.unitize(self.environment_transform[:3, :3].T)
                box = corners(scene.bounds)
                center = box.mean(axis=0)
                center += ((box @ up).min() - center @ up - scene.scale / 100.0) * up
                side = np.ptp(box @ np.column_stack((right, back)), axis=0).max() or 1.0
                arrays = geometry_to_arrays(
                    grid(side=side, count=4, plane_origin=center, plane_normal=up)
                )
            elif name == ("ground",):
                # blended shadow catcher, copying the material cached on the mesh
                floor = Trimesh(self.ground, [[0, 1, 2], [0, 2, 3]], process=False)
                arrays = geometry_to_arrays(floor)
                material = arrays.material.copy()
                material["flags"][[0, 2]] = (shaders.ALPHA_MODES["BLEND"], 1)
                arrays = replace(arrays, material=material)
            else:
                arrays = geometry_to_arrays(scene.geometry[name])
            if arrays is not None:
                self.geometry_buffers[name] = upload_geometry(arrays)
        for name in set(self.geometry_buffers).difference(keys):
            delete(self.geometry_buffers.pop(name).names)
        self.geometry_keys = keys

        # matrices are read every frame, so only structure and visibility matter
        structure = self.synced[1]
        if hash(graph) != self.synced[0]:
            forest = graph.transforms
            structure = (
                dict(forest.parents),
                {node: data.get("geometry") for node, data in forest.node_data.items()},
            )
        structure = (*structure[:2], frozenset(state.hidden), frozenset(state.fixed))
        changed = changed or structure != self.synced[1]
        self.synced = (hash(graph), structure)
        if not changed:
            return
        self.placed = None

        # a row per visible node with geometry, then the decorations
        self.table = table = SimpleNamespace()
        node_data = graph.transforms.node_data
        drawn = {node: node_data[node].get("geometry") for node in graph.nodes_geometry}
        nodes = [node for node in drawn if node not in state.hidden]
        nodes = [node for node in nodes if drawn[node] in self.geometry_buffers]
        names = [drawn[node] for node in nodes]
        names.extend(name for name in self.geometry_buffers if isinstance(name, tuple))
        table.buffers = buffers = [self.geometry_buffers[name] for name in names]
        table.count = count = len(nodes)
        # rows sharing buffers, or material and textures
        first = {}
        table.geometry_index = np.array(
            [first.setdefault(name, len(first)) for name in names], dtype=np.int64
        )
        # numbered in order of first use, so the order coincident faces are
        # drawn in never depends on GL names or the process's hash seed
        materials = {}
        table.material_index = np.array(
            [
                materials.setdefault(
                    (item.material.tobytes(), *item.textures.values()), len(materials)
                )
                for item in buffers
            ],
            dtype=np.int64,
        )
        # meshes, translucent meshes, and rows fixed to the camera
        table.mesh = np.array([item.kind == "mesh" for item in buffers], dtype=bool)
        flags = np.reshape([item.material["flags"] for item in buffers], (-1, 4))
        table.blended = table.mesh & (flags[:, 0] == shaders.ALPHA_MODES["BLEND"])
        table.fixed = np.array([name in state.fixed for name in names], dtype=bool)
        # index into `scene.graph.nodes_geometry` or -1, centroid, box corners
        position = {node: index for index, node in enumerate(drawn)}
        table.node_index = np.full(len(buffers), -1.0)
        table.node_index[:count] = [position[node] for node in nodes]
        table.centroids = np.reshape([item.centroid for item in buffers], (-1, 3))
        table.corners = np.reshape([item.corners for item in buffers], (-1, 8, 3))
        table.origins = np.reshape([item.origin for item in buffers], (-1, 3))

        # skinned rows start at `joint_offset` in `joint_matrices`, and
        # joint nodes shared between skins are looked up once
        skinned = [row for row, node in enumerate(nodes) if node in scene.skins]
        skinned = np.array([row for row in skinned if buffers[row].skinned], int)
        skins = [scene.skins[nodes[row]] for row in skinned]
        counts = np.array([len(skin.joints) for skin in skins], dtype=np.int64)
        table.joint_offset = np.full(len(buffers), -1.0)
        table.joint_offset[skinned] = 4.0 * (np.cumsum(counts) - counts)
        table.inverse_binds = np.concatenate(
            [np.zeros((0, 4, 4)), *(skin.inverse_bind_matrices for skin in skins)]
        )
        table.bound_rows = np.repeat(skinned, counts)
        lookup = {}
        joints = [joint for skin in skins for joint in skin.joints]
        table.joint_index = np.array(
            [lookup.setdefault(joint, len(lookup)) for joint in joints], dtype=np.int64
        )
        # scene lights and a followed camera move with the graph
        lights, cameras = [], []
        if scene.has_lights:
            lights = [light.name for light in self.lights]
        if state.follow_camera and scene.has_camera:
            cameras = [scene.camera.name]
        # where lights and joints start in `world_nodes`
        table.lit = count + len(lights)
        table.skinned = table.lit + len(lookup)
        # edges from the world to every needed node, with parent rows or -1
        world_nodes = [*nodes, *lights, *lookup, *cameras]
        parents, above = graph.transforms.parents, {}
        for node in world_nodes:
            while node in parents and node not in above:
                above[node] = parents[node]
                node = parents[node]
        table.node_rows = row = {node: index for index, node in enumerate(above)}
        table.chain_edges = [(parent, node) for node, parent in above.items()]
        table.chain_parent = np.array([row.get(node, -1) for node in above.values()], int)
        table.world_rows = np.array([row.get(node, -1) for node in world_nodes], int)

    def draw(self, mode: str = "color", changed: bool = False) -> None:
        """
        Draw the scene into `TARGETS["resolved"]` or `TARGETS["index"]`.

        Parameters
        ------------
        mode : str
          "color" or "index", a key of `shaders.OUTPUT_MODES`.
        changed : bool
          Check for changed geometry first.
        """
        state, context, graph = self.state, self.context, self.scene.graph
        self.window.switch_to()
        if changed or state.scene_changed:
            self.sync_geometry()
            state.scene_changed = False
        table, count = self.table, self.table.count

        # world transforms of every node, light and joint
        placement = (hash(graph), self.animation_time)
        if self.placed != placement:
            if self.placed is None or self.placed[0] != placement[0]:
                # edge matrices, gathered once per graph hash
                edges = graph.transforms.edge_data
                self.edge_matrices = np.reshape(
                    [edges[edge].get("matrix", IDENTITY) for edge in table.chain_edges],
                    (-1, 4, 4),
                )
            self.placed, self.shadow_fit = placement, None
            chain = self.edge_matrices.copy()
            # animated edges without posing the scene, looping shorter ones
            if self.animation_time is not None:
                for animation in self.scene.animations:
                    seconds = self.animation_time
                    if animation.duration > 0.0:
                        seconds = seconds % animation.duration
                    # chain row of each animated node, or -1
                    rows = [table.node_rows.get(node, -1) for node in animation.nodes]
                    rows = np.array(rows, dtype=np.int64)
                    chain[rows[rows >= 0]] = animation.interpolate(seconds)[rows >= 0]
            # compose with parents by pointer doubling
            parent = table.chain_parent
            while (parent >= 0).any():
                rows = np.flatnonzero(parent >= 0)
                chain[rows] = np.matmul(chain[parent[rows]], chain[rows])
                parent = np.where(parent >= 0, parent[parent], -1)
            # identity for the world itself
            world = np.concatenate((chain, IDENTITY[None]))[table.world_rows]
            # identity for decorations
            transforms = np.tile(IDENTITY, (len(table.buffers), 1, 1))
            transforms[:count] = world[:count]
            # each geometry's rows are stored relative to its origin
            transforms[:, :3, 3] += np.matmul(
                transforms[:, :3, :3], table.origins[:, :, None]
            )[..., 0]
            lights = self.light_transforms
            if table.lit > count:
                lights = world[count : table.lit]
            # `world(joint) @ inverse_bind` for every joint of every skin
            joints = world[table.lit : table.skinned][table.joint_index]
            joints = np.matmul(joints, table.inverse_binds)
            upload(gl.GL_TEXTURE_BUFFER, self.joint_buffer, joints.astype(np.float32))
            # scene bounds from transformed boxes, without touching vertices
            boxes = np.concatenate((transforms[:count], joints))
            rows = np.concatenate((np.arange(count), table.bound_rows))
            points = np.matmul(table.corners[rows], np.swapaxes(boxes[:, :3, :3], 1, 2))
            points = (points + boxes[:, None, :3, 3]).reshape((-1, 3))
            self.scene_bounds = None
            if len(points) > 0:
                self.scene_bounds = np.array([points.min(axis=0), points.max(axis=0)])
            # a skinned mesh's joints put it in world space
            transforms[table.joint_offset >= 0] = IDENTITY
            self.world = (transforms, lights)
            if len(world) > table.skinned:
                # follow the camera node
                self.place_camera(world[table.skinned])
        transforms, lights = self.world
        view = np.linalg.inv(self.camera_transform)
        model_view = np.matmul(view, transforms)
        # fixed geometry is drawn as if the camera never moved
        initial_view = np.linalg.inv(self.initial_camera)
        model_view[table.fixed] = np.matmul(initial_view, transforms[table.fixed])

        # scaled camera planes, tightened around the scene for depth precision
        camera = self.camera
        near, far = camera.z_near * self.camera_scale, camera.z_far * self.camera_scale
        z_near, z_far, scene_scale = near, far, 1.0
        if self.scene_bounds is not None:
            # the scene corners, the origin for the axis, and the floor
            points = np.vstack((corners(self.scene_bounds), np.zeros(3), self.ground))
            depth = -(points @ view[2, :3] + view[2, 3])
            # `Scene.scale` of the scene as drawn now
            scene_scale = np.linalg.norm(np.ptp(self.scene_bounds, axis=0))
            margin = 0.05 * scene_scale
            fit_far = min(depth.max() + margin, far)
            # 24 bit depth resolves `far^2 / (near * 2^24)` at `far`: keep it under 1%
            fit_near = max(depth.min() - margin, near, fit_far * 100.0 / 2**24)
            # unless the scene is entirely outside the camera's planes
            if fit_near < fit_far:
                z_near, z_far = fit_near, fit_far
        # both fields of view, as `look_at` frames with both
        scale = 1.0 / np.tan(np.radians(camera.fov) / 2.0)
        orthographic = camera.magnification is not None
        if orthographic:
            scale = 1.0 / (np.asarray(camera.magnification) * self.camera_scale)

        frame = self.frame_constants.copy()
        frame["occlusion_radius"][0, 0] = OCCLUSION_RADIUS * scene_scale
        frame["projection"] = projection_matrices(scale, z_near, z_far, orthographic)
        frame["inverse_projection"] = np.linalg.inv(frame["projection"][0])
        frame["output_mode"][:, 0] = shaders.OUTPUT_MODES[mode]
        frame["smooth_shading"][:, 0] = state.smooth
        # lights in view space shining down -Z, with W zero for directional
        light_view = np.matmul(view, lights)
        towards = util.unitize(light_view[:, :3, 2])
        rows = frame["lights"][0, : len(lights)]
        directional = rows["position"][:, 3:] == 0.0
        rows["position"][:, :3] = np.where(directional, towards, light_view[:, :3, 3])
        rows["direction"][:, :3] = -towards
        # environment rotation from world to view space
        rotation = frame["environment_rotation"][0]
        rotation[:3, :3] = rotation[:3, :3] @ view[:3, :3].T
        # only depth, for shadows and ambient occlusion
        depth_frame = frame.copy()
        depth_frame["output_mode"][:, 0] = shaders.OUTPUT_MODES["depth"]

        gl.glEnable(gl.GL_DEPTH_TEST)
        # equal depth passes so line segments sharing an endpoint join
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["joint_matrices"])
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.textures["joint_matrices"])
        effects = mode == "color" and state.quality
        # opaque scene meshes, without decorations
        scene_rows = table.node_index >= 0
        opaque = table.mesh & ~table.blended & scene_rows
        # lights with a layer of `shadow_maps`
        casting = rows["shadow"][:, 0] >= 0
        if not (effects and self.scene_bounds is not None):
            rows["shadow"][:, 0] = -1.0
        elif casting.any():
            if self.shadow_fit is None:
                # shadow maps don't depend on the camera: fit each around the scene
                world = np.array(lights[casting], dtype=np.float64)
                parallel = directional[casting, 0]
                # drop node scale so light space is in world units
                world[:, :3, :3] /= np.linalg.norm(
                    world[:, :3, :3], axis=1, keepdims=True
                )
                to_light = np.linalg.inv(world)
                box = corners(self.scene_bounds)
                local = np.matmul(box, np.swapaxes(to_light[:, :3, :3], 1, 2))
                local += to_light[:, None, :3, 3]
                # distance in front of each light along -Z
                depth = -local[..., 2]
                # well past float32 rounding so nothing clips, nonzero for a point
                margin = 1e-3 * np.ptp(box, axis=0).max() + 1e-12
                far = depth.max(axis=1) + margin
                low, high = local[..., :2].min(axis=1), local[..., :2].max(axis=1)
                # nonzero for a scene seen edge on
                size = np.maximum(high - low, 1e-12)
                # center directional lights on the scene
                to_light[parallel, :2, 3] -= (high + low)[parallel] / 2.0
                # spot near plane is in front, at least `far / 1000` so a 24 bit
                # depth buffer resolves `far^2 / (near * 2^24)` under 0.01% of far
                nearest = depth.min(axis=1)
                near = np.where(
                    parallel, nearest - margin, np.maximum(nearest, far * 1e-3)
                )
                # clip the cone so the focal length is finite
                focal = 1.0 / np.tan(
                    np.clip(rows["falloff"][casting, 3], 1e-3, np.pi / 2.0 - 1e-3)
                )
                scale = np.where(parallel[:, None], 2.0 / size, focal[:, None])
                self.shadow_fit = np.matmul(
                    projection_matrices(scale, near, far, parallel), to_light
                )
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers["shadow"])
                gl.glViewport(0, 0, *self.sizes["shadow"])
                # slope scaled offset against shadow acne
                gl.glPolygonOffset(2.0, 4.0)
                shadow_frame = depth_frame.copy()
                for layer, light_clip in enumerate(self.shadow_fit):
                    gl.glFramebufferTextureLayer(
                        gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                        self.textures["shadow"], 0, layer,
                    )  # fmt: skip
                    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
                    # the light's view is part of its projection
                    shadow_frame["projection"] = light_clip
                    upload(gl.GL_UNIFORM_BUFFER, context.frame_buffer, shadow_frame)
                    self.draw_rows(opaque & ~table.fixed, transforms, "depth")
            # the shader looks shadows up from view space
            shadow_matrices = self.shadow_fit @ self.camera_transform
            frame["shadow_matrices"][0, : len(shadow_matrices)] = shadow_matrices

        # meshes are pushed back slightly so lines on their surface win
        gl.glPolygonOffset(1.0, 1.0)
        # white is no occlusion
        occlusion = context.white_texture
        if effects:
            upload(gl.GL_UNIFORM_BUFFER, context.frame_buffer, depth_frame)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers["scene_depth"])
            gl.glViewport(0, 0, *self.sizes["scene_depth"])
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            self.draw_rows(opaque, model_view, "depth")
            self.fullscreen("occlusion")
            self.fullscreen("blur")
            occlusion = self.textures["occlusion"]
        upload(gl.GL_UNIFORM_BUFFER, context.frame_buffer, frame)
        # bound every frame as the window owns the binding table too
        gl.glBindBufferBase(
            gl.GL_UNIFORM_BUFFER, BLOCK_BINDINGS["Frame"], context.frame_buffer
        )
        for sampler, kind, texture in (
            ("shadow_maps", gl.GL_TEXTURE_2D_ARRAY, self.textures["shadow"]),
            ("ambient_occlusion", gl.GL_TEXTURE_2D, occlusion),
            ("environment", gl.GL_TEXTURE_CUBE_MAP, self.textures["environment"]),
        ):
            gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[sampler])
            gl.glBindTexture(kind, texture)

        output = {"color": "multisample", "index": "index"}[mode]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers[output])
        gl.glViewport(0, 0, *self.sizes[output])
        gl.glClearColor(*self.background)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if mode == "index":
            # depth zero and node -1, which a clamped `glClearColor` may not write
            gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, -1.0, 0.0, 0.0))
        elif self.render_settings["show_environment"]:
            self.fullscreen("background")
        gl.glEnable(gl.GL_DEPTH_TEST)
        SWITCHES[mode == "color"](gl.GL_BLEND)
        # premultiplied alpha
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        # decorations are only drawn in color
        self.draw_rows(scene_rows | (mode == "color"), model_view, mode)
        gl.glBindVertexArray(context.empty_vertex_array)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDepthMask(gl.GL_TRUE)
        state.dirty = False
        if mode != "color":
            return
        # resolve antialiasing, then add bloom
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.framebuffers["multisample"])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.framebuffers["resolved"])
        area = (0, 0, *self.sizes["resolved"])
        gl.glBlitFramebuffer(*area, *area, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        if state.quality and self.render_settings["bloom"] > 0.0:
            for name in ("bloom_across", "bloom_down", "bloom_add"):
                self.fullscreen(name)

    def draw_rows(self, mask: NDArray, model_view: NDArray, mode: str) -> None:
        """
        Draw rows of `self.table` instanced by geometry, translucent last.

        Parameters
        ------------
        mask : (n,) bool
          Rows of `self.table` to draw.
        model_view : (n, 4, 4) float
          Row to view or light transforms.
        mode : str
          Key of `shaders.OUTPUT_MODES`, "depth" is double sided.
        """
        table, state, programs = self.table, self.state, self.context.programs
        blended, geometry_index = table.blended, table.geometry_index
        # glTF hides nodes by scaling them to zero
        scaled = np.abs(model_view[:, :3, :3]).max(axis=(1, 2), initial=0.0) > 0.0
        rows = np.flatnonzero(mask & scaled)
        view_z = model_view[rows, 2]
        depth = (view_z[:, :3] * table.centroids[rows]).sum(axis=1) + view_z[:, 3]
        # opaque rows by material then geometry, then translucent far to near
        blend = blended[rows]
        primary = np.where(blend, depth, geometry_index[rows])
        secondary = np.where(blend, 0, table.material_index[rows])
        order = rows[np.lexsort((primary, secondary, blend))]
        # signed adjugate: the inverse transpose up to scale, defined at zero scale
        linear = model_view[order, :3, :3]
        instances = np.zeros(len(order), dtype=shaders.INSTANCE)
        instances["model_view"] = model_view[order].transpose((0, 2, 1))
        mirror = np.sign(np.linalg.det(linear))[:, None, None]
        instances["normal_matrix"] = adjugate(linear) * mirror
        instances["node_index"] = table.node_index[order]
        instances["joint_offset"] = table.joint_offset[order]
        # a draw call per geometry, translucent mesh and line
        changes = np.diff(geometry_index[order], prepend=-1) != 0
        starts = np.flatnonzero(changes | (blended | ~table.mesh)[order])
        ends = np.append(starts[1:], len(order))

        # each draw uploads a slice of `instances`
        size, pointer = instances.itemsize, instances.ctypes.data
        translucent = (blended[order[starts]] & (mode == "color")).tolist()
        program, material = None, None
        draws = zip(starts.tolist(), ends.tolist(), order[starts].tolist(), translucent)
        for start, end, row, hide in draws:
            item = table.buffers[row]
            name = "line"
            if item.kind == "mesh":
                name = "mesh"
            uniforms = self.context.uniforms[name]
            if name != program:
                program, material = name, None
                gl.glUseProgram(programs[name])
                polygons = gl.GL_FILL
                if name == "mesh" and state.wireframe and mode != "depth":
                    polygons = gl.GL_LINE
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, polygons)
                SWITCHES[name == "mesh"](gl.GL_POLYGON_OFFSET_FILL)
                gl.glDisable(gl.GL_CULL_FACE)
            if hide:
                # translucent meshes don't write depth
                gl.glDepthMask(gl.GL_FALSE)
            gl.glBindVertexArray(item.vertex_array)
            if name == "line":
                gl.glUniform1i(uniforms["points"], item.kind == "points")
                gl.glUniform1f(uniforms["node_index"], table.node_index[row])
                matrix = (gl.GLfloat * 16)(*model_view[row].ravel())
                gl.glUniformMatrix4fv(uniforms["model_view"], 1, gl.GL_TRUE, matrix)
                gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, item.count)
                continue
            if table.material_index[row] != material:
                material = table.material_index[row]
                for texture in shaders.MATERIAL_TEXTURES:
                    gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[texture])
                    image = item.textures.get(texture, self.context.white_texture)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, image)
                binding = BLOCK_BINDINGS["Material"]
                gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, binding, item.material_buffer)
                # the fourth flag is double sided
                culled = state.cull and mode != "depth" and item.material["flags"][3] == 0
                SWITCHES[bool(culled)](gl.GL_CULL_FACE)
            gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["face_colors"])
            gl.glBindTexture(gl.GL_TEXTURE_BUFFER, item.face_colors)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, item.instance_buffer)
            length, offset = (end - start) * size, pointer + start * size
            gl.glBufferData(gl.GL_ARRAY_BUFFER, length, offset, gl.GL_STREAM_DRAW)
            gl.glDrawElementsInstanced(
                gl.GL_TRIANGLES, item.count, gl.GL_UNSIGNED_INT, None, end - start
            )

    def fullscreen(self, name: str) -> None:
        """
        Draw one of `shaders.PASSES` over the whole of its output.

        Parameters
        ------------
        name : str
          Key of `shaders.PASSES`.
        """
        inputs, output, adds = shaders.PASSES[name]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers[output])
        gl.glViewport(0, 0, *self.sizes[output])
        gl.glDisable(gl.GL_DEPTH_TEST)
        SWITCHES[adds](gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        for sampler, source in inputs.items():
            gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[sampler])
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[source])
        gl.glUseProgram(self.context.programs["fullscreen"])
        location = self.context.uniforms["fullscreen"]["pass_index"]
        gl.glUniform1i(location, list(shaders.PASSES).index(name))
        gl.glBindVertexArray(self.context.empty_vertex_array)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def read_pixels(self, name: str) -> NDArray:
        """
        Copy the color of one of `TARGETS` off the GPU.

        Parameters
        ------------
        name : str
          Key of `TARGETS` with color in `READ_FORMATS`.

        Returns
        ------------
        pixels : (height, width, channels) uint8 or float32
          Flipped view with the first row at the top.
        """
        pixel_format, kind, channels, dtype = READ_FORMATS[TARGETS[name][0]]
        width, height = self.sizes[name]
        pixels = np.zeros((height, width, channels), dtype=dtype)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.framebuffers[name])
        gl.glReadPixels(0, 0, width, height, pixel_format, kind, pixels.ctypes.data)
        # OpenGL puts the first row at the bottom
        return pixels[::-1]

    def to_array(
        self, include_depth: bool = False, include_segmentation: bool = False
    ) -> RenderArrays:
        """
        Render the scene and read back color, depth and segmentation.

        Parameters
        ------------
        include_depth : bool
          Also read view depth per pixel.
        include_segmentation : bool
          Also read the `scene.graph.nodes_geometry` index per pixel.

        Returns
        ------------
        arrays : RenderArrays
          Color, and depth and segmentation or None.
        """
        self.draw(changed=True)
        color = self.read_pixels("resolved")
        depth, segmentation = None, None
        if include_depth or include_segmentation:
            self.draw("index")
            values = self.read_pixels("index")
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
          If the scene was redrawn.
        """
        self.window.switch_to()
        drawn = self.state.dirty or self.state.scene_changed
        if drawn:
            self.draw()
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.framebuffers["resolved"])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        area = (0, 0, *self.sizes["resolved"])
        gl.glBlitFramebuffer(*area, *area, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        return drawn

    def on_resize(self, width: int, height: int) -> bool:
        """
        Match the images and camera to a new window size.

        Parameters
        ------------
        width, height : int
          Window size in pixels.

        Returns
        ------------
        handled : bool
          Skips pyglet's default handler.
        """
        size = tuple(self.window.get_framebuffer_size())
        if self.sizes["resolved"] != size:
            self.allocate(size)
        self.state.trackball.resize(np.array([width, height]))
        self.state.dirty = True
        return pyglet.event.EVENT_HANDLED

    def on_mouse_press(self, x, y, buttons, modifiers) -> None:
        """
        Start a trackball motion picked by mouse button and modifiers.

        Parameters
        ------------
        x, y : int
          Mouse position in window pixels.
        buttons : int
          Pressed mouse buttons.
        modifiers : int
          Held modifier keys.
        """
        key, mouse = pyglet.window.key, pyglet.window.mouse
        held = bool(modifiers & key.MOD_CTRL) + 2 * bool(modifiers & key.MOD_SHIFT)
        left = (Trackball.STATE_ROTATE, Trackball.STATE_PAN, Trackball.STATE_ROLL)
        motions = {
            mouse.LEFT: (*left, Trackball.STATE_ZOOM)[held],
            mouse.MIDDLE: Trackball.STATE_PAN,
            mouse.RIGHT: Trackball.STATE_ZOOM,
        }
        if buttons in motions:
            self.state.trackball.set_state(motions[buttons])
        self.state.trackball.down(np.array([x, y]))
        self.camera_transform = self.state.trackball.pose
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
            # orthographic cameras zoom by magnification
            camera.magnification = camera.magnification * 0.9**dy
        self.state.dirty = True

    def on_key_press(self, symbol, modifiers) -> None:
        """
        Run the `KEY_BINDINGS` action of a key, or nudge with arrows.

        Parameters
        ------------
        symbol : int
          The pressed key.
        modifiers : int
          Held modifier keys.
        """
        key = pyglet.window.key
        keys = {getattr(key, letter.upper()): name for letter, _, name in KEY_BINDINGS}
        name = keys.get(symbol)
        nudge = {
            key.LEFT: (-10, 0),
            key.RIGHT: (10, 0),
            key.DOWN: (0, -10),
            key.UP: (0, 10),
        }
        if name in {item.name for item in fields(ViewerState)}:
            self.toggle(name)
        elif name is not None and hasattr(type(self), name):
            getattr(self, name)()
        elif name is not None:
            getattr(self.window, name)()
        elif symbol in nudge:
            self.state.trackball.down([0, 0])
            self.on_mouse_drag(*nudge[symbol], 0, 0, 0, 0)

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
        Advance and loop `scene.animations` without changing the scene.

        Parameters
        ------------
        elapsed : float
          Seconds since the last tick.
        """
        if not self.state.playing:
            return
        self.animation_time = (self.animation_time or 0.0) + elapsed
        if self.scene.duration > 0.0:
            self.animation_time %= self.scene.duration
        self.state.dirty = True
