"""
viewer.py
-----------

An OpenGL 3.3 viewer for `trimesh.Scene` which also renders offscreen,
moving the arrays `conversion.py` makes with raw OpenGL calls which
pyglet 1.5, 2.x and 3.x all expose. Every frame is drawn into images
in `TARGETS` and copied out, so a screenshot is the same with visible,
hidden, high-DPI and headless windows.
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
from ...scene.cameras import DEFAULT_FOV, current_view
from ...scene.lighting import prefilter
from ...transformations import adjugate
from ...typed import ArrayLike, Integer, Iterable, NDArray, Sequence
from ...visual import to_rgba
from ..conversion import (
    RENDER_DEFAULTS,
    GeometryArrays,
    frame_constants,
    geometry_to_arrays,
    render_lights,
    shadow_projections,
)
from ..trackball import Trackball
from . import shaders

# with no display on Linux use EGL to get a context: this has to
# be set before importing `gl` which connects to a display
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pyglet.options["headless"] = True
# pyglet checks for an error after every call unless this is off when
# `gl` is first imported, which made drawing 400 meshes 50% slower, 3.5
# to 5.3 ms a frame, and the check can't be turned off for one context
pyglet.options["debug_gl"] = False

# what to tell anyone whose machine can't give the viewer a context
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

# a window couldn't be given the context it asked for, which pyglet
# 3.0.dev11 raises as an IndexError when EGL has no matching config, and
# as an ImportError when making the window finds no EGL library at all
CONTEXT_ERRORS = (
    pyglet.window.NoSuchConfigException,
    ContextException,
    IndexError,
    ImportError,
)

# Every image a frame is drawn into, each with a framebuffer of its own:
# name : (color format or zero, has depth, size, filtering, layers, samples)
# where the size is a key of the sizes `SceneViewer.allocate` works out
TARGETS = {
    # the antialiased frame
    "multisample": (gl.GL_RGBA8, True, "full", gl.GL_NEAREST, 0, 4),
    # the frame resolved, which is read back and shown, with mip levels for bloom
    "resolved": (gl.GL_RGBA8, False, "full", gl.GL_LINEAR_MIPMAP_LINEAR, 0, 0),
    # the view depth and node index of every pixel, unantialiased
    "index": (gl.GL_RG32F, True, "full", gl.GL_NEAREST, 0, 0),
    # the depth each shadow casting light sees, which is compared against
    # with linear filtering blending the comparisons into a soft edge
    "shadow": (0, True, "shadow", gl.GL_LINEAR, shaders.MAX_SHADOWS, 0),
    # the scene's depth for ambient occlusion, which is noisy then blurred
    "scene_depth": (0, True, "half", gl.GL_NEAREST, 0, 0),
    "noisy": (gl.GL_R8, False, "half", gl.GL_NEAREST, 0, 0),
    "occlusion": (gl.GL_R8, False, "half", gl.GL_LINEAR, 0, 0),
    # the bloom blurred across a mip level of `resolved` then down
    "across": (gl.GL_RGBA16F, False, "bloom", gl.GL_LINEAR, 0, 0),
    "glow": (gl.GL_RGBA16F, False, "bloom", gl.GL_LINEAR, 0, 0),
}

# width and height of every shadow map in pixels
SHADOW_RESOLUTION = 2048
# the standard deviation of the glow of bloom as a fraction of the
# picture's height, so it looks the same at every resolution: one degree
# of the default camera's vertical field of view
BLOOM_SPREAD = 1.0 / DEFAULT_FOV[1]
# the fewest texels per standard deviation bloom is blurred at, where a
# Gaussian's spectrum at the Nyquist frequency is `exp(-2 * pi^2)`, or
# 3e-9 of its peak, so sampling it that coarsely adds no aliasing
BLOOM_TEXELS = 2.0
# a visible viewer drawing slower than this switches to fast drawing
SLOW_FPS = 15.0
# the uniform buffer binding each of `shaders.BLOCKS` is read from
BLOCK_BINDINGS = {name: index for index, name in enumerate(shaders.BLOCKS)}
# color storage : (format, type, channels, numpy dtype) to read it with
READ_FORMATS = {
    gl.GL_RGBA8: (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 4, np.uint8),
    gl.GL_RG32F: (gl.GL_RG, gl.GL_FLOAT, 2, np.float32),
}
# how to create and free each kind of OpenGL name but programs
GENERATORS = {
    "buffers": (gl.glGenBuffers, gl.glDeleteBuffers),
    "textures": (gl.glGenTextures, gl.glDeleteTextures),
    "framebuffers": (gl.glGenFramebuffers, gl.glDeleteFramebuffers),
    "vertex_arrays": (gl.glGenVertexArrays, gl.glDeleteVertexArrays),
}
# turn an OpenGL capability off or on, by a bool
SWITCHES = (gl.glDisable, gl.glEnable)
# how to set the uniforms which aren't in a block by their OpenGL type,
# where booleans and samplers which are a texture unit are integers
UNIFORM_SETTERS = {
    gl.GL_FLOAT: gl.glUniform1f,
    gl.GL_FLOAT_MAT4: lambda location, value: gl.glUniformMatrix4fv(
        location, 1, gl.GL_TRUE, (gl.GLfloat * 16)(*np.ravel(value))
    ),
}

IDENTITY = np.eye(4)
# maps clip space of -1.0 to 1.0 to texture coordinates of 0.0 to 1.0
SHADOW_BIAS = np.array(
    [[0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5], [0, 0, 0, 1.0]]
)

# (input, action) for the help message
MOUSE_BINDINGS = (
    *(("drag", "rotate"), ("ctrl + drag", "pan"), ("shift + drag", "roll")),
    *(("wheel", "zoom"), ("arrows", "nudge")),
)
# (key, action, `ViewerState` switch, or method of the viewer or its window)
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

# `main`'s `SceneViewer` methods, which warn and will raise after April
# 2028: the `ViewerState` switch they toggle, or None where geometry and
# the axis and grid follow the scene and state before every draw anyway
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
# and its keywords, of which `window_conf` is `config` and `line_settings`
# holds `line_width` and `point_size`, and the rest do nothing
LEGACY_KEYWORDS = ("window_conf", "line_settings", "resizable", "offset_lines")
LEGACY_KEYWORDS += ("profile", "record")

# one hidden window shared by every offscreen render, so rendering
# thousands of images doesn't create thousands of OpenGL contexts
_HIDDEN_CONTEXT = {}


class ShaderProgram(NamedTuple):
    """
    A linked program and the (location, OpenGL type) of each uniform.
    """

    program: int
    uniforms: dict

    def set(self, name: str, value) -> None:
        """
        Set a uniform on this program while it is in use, ignoring
        uniforms the compiler optimized out.

        Parameters
        ------------
        name : str
          Uniform name in the shader source.
        value : float, int, bool or (4, 4) float
          Value of the uniform's type.
        """
        if name in self.uniforms:
            location, kind = self.uniforms[name]
            UNIFORM_SETTERS.get(kind, gl.glUniform1i)(location, value)


class RenderContext(NamedTuple):
    """
    A window with an OpenGL context and what every viewer drawing
    in it shares.
    """

    window: pyglet.window.Window
    # `shaders.PROGRAMS` by name
    programs: dict
    # a 1x1 white texture bound in place of any missing texture
    white_texture: int
    # a vertex array with no attributes for drawing without inputs
    empty_vertex_array: int
    # the uniform buffer holding one `shaders.FRAME`
    frame_buffer: int
    # every OpenGL name above by kind, for `delete`
    names: dict


class GeometryBuffers(NamedTuple):
    """
    One piece of geometry on the GPU and what drawing it needs.
    """

    kind: str
    vertex_array: int
    # indices for meshes, segments for lines and points
    count: int
    # the `shaders.MATERIAL` record, which meshes have a uniform buffer of
    material: NDArray
    material_buffer: int
    # `shaders.MATERIAL_TEXTURES` name to texture
    textures: dict
    # mesh only: the buffer texture with face colors or zero
    face_colors: int
    # mesh only: per-instance rows, rewritten before every draw
    instance_buffer: int
    # the material record and texture ids, so consecutive meshes with
    # the same one skip binding it again
    material_key: bytes
    # the center of the vertices, and the corners of the box around them
    centroid: NDArray
    corners: NDArray
    skinned: bool
    # every OpenGL name above by kind, for `delete`
    names: dict


class RenderArrays(NamedTuple):
    """
    What `SceneViewer.to_array` read back, with the first row at the top.
    """

    # (height, width, 4) uint8 antialiased image
    color: NDArray
    # (height, width) float32 distance along the view axis under the center
    # of every pixel, zero for the background, or None if not asked for
    depth: NDArray | None
    # (height, width) int64 index into `scene.graph.nodes_geometry` under the
    # center of every pixel, -1 for the background, or None if not asked for
    segmentation: NDArray | None


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
    # scene graph nodes which aren't drawn
    hidden: set = field(default_factory=set)
    # geometry names which don't move with the camera
    fixed: Iterable = field(default_factory=set)
    # play `scene.animations` if there are any
    playing: bool = True
    # the camera rides the node of the scene's camera, as renders do,
    # rather than staying where the trackball puts it
    follow_camera: bool = False
    # directional and spot lights cast shadows
    shadows: bool = True
    # nearby surfaces hide the environment from creases and corners
    ambient_occlusion: bool = True
    # skip shadows and ambient occlusion for slow machines,
    # which a visible viewer turns on by itself if drawing is too slow
    fast: bool = False
    # the scene needs to be drawn again
    dirty: bool = True
    # geometry or what draws it may have changed, which also needs a draw
    scene_changed: bool = True


def generate(names: dict, kind: str) -> int:
    """
    Create one OpenGL object of a kind, recorded to be freed by `delete`.

    Parameters
    ------------
    names : dict
      {kind : [name, ...]} to record it in.
    kind : str
      A key of `GENERATORS`.

    Returns
    ------------
    name : int
      The new OpenGL name.
    """
    value = gl.GLuint()
    GENERATORS[kind][0](1, ctypes.byref(value))
    names.setdefault(kind, []).append(value.value)
    return value.value


def delete(names: dict) -> None:
    """
    Free OpenGL objects on the context which is current.

    Parameters
    ------------
    names : dict
      {kind of `GENERATORS` or "programs" : [name, ...]}
    """
    # a program in use is only deleted once it no longer is
    gl.glUseProgram(0)
    for kind, values in names.items():
        if kind == "programs":
            for program in values:
                gl.glDeleteProgram(program)
            continue
        GENERATORS[kind][1](len(values), (gl.GLuint * len(values))(*values))


def upload(
    target: int, buffer: int, data: NDArray, usage: int = gl.GL_STREAM_DRAW
) -> int:
    """
    Replace the contents of an OpenGL buffer with an array, which by
    default changes every frame like `shaders.INSTANCE` rows.

    Parameters
    ------------
    target : int
      Buffer type, i.e. `GL_ARRAY_BUFFER`, which is left bound.
    buffer : int
      The OpenGL buffer.
    data : numpy.ndarray
      Array to copy.
    usage : int
      How often it changes, i.e. `GL_STATIC_DRAW` for never.

    Returns
    ------------
    buffer : int
      The OpenGL buffer.
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
      Colors are sRGB and should be decoded to linear when sampled.

    Returns
    ------------
    texture : int
      The OpenGL texture.
    """
    # trimesh UV coordinates have the origin at the bottom left
    pixels = np.ascontiguousarray(pixels[::-1], dtype=np.uint8)
    texture = generate(names, "textures")
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    internal = gl.GL_RGBA8
    if srgb:
        internal = gl.GL_SRGB8_ALPHA8
    height, width = pixels.shape[:2]
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, internal, width, height, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels.ctypes.data,
    )  # fmt: skip
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    for parameter, value in (
        (gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
        (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
    ):
        gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, value)
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
      What drawing and later deleting the geometry needs.
    """
    names = {}
    vertex_array = generate(names, "vertex_arrays")
    gl.glBindVertexArray(vertex_array)
    rows, count = arrays.rows, len(arrays.rows)
    # each segment is one instance which the shader makes a quad around,
    # and every node drawing a mesh is one instance rewritten every frame
    program, inputs = "line", [(rows, 1)]
    if arrays.kind == "mesh":
        points, count, program = rows["position"], arrays.faces.size, "mesh"
        inputs = [(rows, 0), (np.zeros(1, dtype=shaders.INSTANCE), 1), (arrays.skin, 0)]
        upload(gl.GL_ELEMENT_ARRAY_BUFFER, generate(names, "buffers"), arrays.faces)
    else:
        points = np.concatenate((rows["segment_start"], rows["segment_end"]))
    locations = shaders.attribute_locations(*shaders.INPUTS[program])
    buffers, instance_buffer = [], 0
    for data, divisor in inputs:
        if data is None:
            continue
        buffers.append(upload(gl.GL_ARRAY_BUFFER, generate(names, "buffers"), data))
        if data.dtype == shaders.INSTANCE:
            instance_buffer = buffers[-1]
        # one input per field, where a matrix is one input per column each
        # the size of a row as matrices are stored transposed, and bytes
        # are normalized to 0.0-1.0 while everything else is float32
        for name in data.dtype.names:
            item, offset = data.dtype.fields[name][:2]
            columns, width = (1, 1, *item.shape)[-2:]
            normalized = item.base.itemsize == 1
            kind = gl.GL_FLOAT
            if normalized:
                kind = gl.GL_UNSIGNED_BYTE
            for column in range(columns):
                location = locations[name] + column
                gl.glEnableVertexAttribArray(location)
                start = ctypes.c_void_p(offset + column * width * item.base.itemsize)
                gl.glVertexAttribPointer(
                    location, width, kind, normalized, data.dtype.itemsize, start
                )
                gl.glVertexAttribDivisor(location, divisor)
    gl.glBindVertexArray(0)

    face_colors = 0
    if arrays.face_colors is not None:
        # face colors are looked up with `gl_PrimitiveID` in a buffer texture
        buffer = upload(
            gl.GL_TEXTURE_BUFFER, generate(names, "buffers"), arrays.face_colors
        )
        face_colors = generate(names, "textures")
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, face_colors)
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA8, buffer)
    textures = {
        name: upload_image(names, np.asarray(image.convert("RGBA")), srgb=srgb)
        for name, image in arrays.textures.items()
        for srgb in [shaders.MATERIAL_TEXTURES[name]]
    }
    texture_ids = [textures.get(name, 0) for name in shaders.MATERIAL_TEXTURES]
    return GeometryBuffers(
        kind=arrays.kind,
        vertex_array=vertex_array,
        count=count,
        material=arrays.material,
        material_buffer=upload(
            gl.GL_UNIFORM_BUFFER, generate(names, "buffers"), arrays.material
        ),
        textures=textures,
        face_colors=face_colors,
        instance_buffer=instance_buffer,
        material_key=arrays.material.tobytes() + np.array(texture_ids).tobytes(),
        centroid=points.mean(axis=0),
        corners=corners([points.min(axis=0), points.max(axis=0)]),
        skinned=arrays.skin is not None,
        names=names,
    )


def create_context(visible: bool, config=None, **options) -> RenderContext:
    """
    Open a window with an OpenGL 3.3 core context, then compile
    `shaders.PROGRAMS` and allocate what every viewer drawing in it shares.

    Parameters
    ------------
    visible : bool
      Show the window on screen.
    config : None or pyglet config
      Override the context configuration.
    options : dict
      Passed to `pyglet.window.Window`, i.e. `width` or `caption`.

    Returns
    ------------
    context : RenderContext
      The window and everything allocated in its context.
    """
    if config is None:
        try:
            config = gl.Config(double_buffer=True)
        except AttributeError:
            # pyglet >= 3 has no `gl.Config` and defaults to 3.3 core
            pass
    if config is not None:
        # pyglet < 3: macOS and pyglet 1.5 only hand out a modern
        # context if one is explicitly requested
        for name, value in (
            ("major_version", 3),
            ("minor_version", 3),
            ("forward_compatible", True),
        ):
            if getattr(config, name, None) is None:
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
        # every OpenGL version starts its version string with "major.minor"
        version, renderer = (
            ctypes.cast(gl.glGetString(name), ctypes.c_char_p).value.decode("utf-8")
            for name in (gl.GL_VERSION, gl.GL_RENDERER)
        )
        if tuple(int(part) for part in version.split(" ")[0].split(".")[:2]) < (3, 3):
            raise RuntimeError(
                f"`{renderer}` only has OpenGL `{version}`: {CONTEXT_HINT}"
            )

        programs = {}
        for name in shaders.PROGRAMS:
            program = gl.glCreateProgram()
            names.setdefault("programs", []).append(program)
            log = ctypes.create_string_buffer(4096)
            logs = []
            for stage, kind in (
                ("vert", gl.GL_VERTEX_SHADER),
                ("frag", gl.GL_FRAGMENT_SHADER),
            ):
                source = shaders.shader_source(f"{name}.{stage}").encode("utf-8")
                pointer = ctypes.cast(
                    ctypes.pointer(ctypes.pointer(ctypes.create_string_buffer(source))),
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
            # read every block from its binding unless the program never uses it
            for block_name, binding in BLOCK_BINDINGS.items():
                block = gl.glGetUniformBlockIndex(program, block_name.encode("utf-8"))
                if block != gl.GL_INVALID_INDEX:
                    gl.glUniformBlockBinding(program, block, binding)
            # the location and type of every uniform, looked up once
            count, uniforms = gl.GLint(), {}
            gl.glGetProgramiv(program, gl.GL_ACTIVE_UNIFORMS, ctypes.byref(count))
            for index in range(count.value):
                size, kind = gl.GLint(), gl.GLenum()
                gl.glGetActiveUniform(
                    program, index, len(log), None, ctypes.byref(size), ctypes.byref(kind), log
                )  # fmt: skip
                location = gl.glGetUniformLocation(program, log)
                uniforms[log.value.decode("utf-8")] = (location, kind.value)
            programs[name] = ShaderProgram(program=program, uniforms=uniforms)
            # samplers read from a fixed texture unit for the life of a program
            gl.glUseProgram(program)
            for sampler, unit in shaders.TEXTURE_UNITS.items():
                programs[name].set(sampler, unit)
        gl.glUseProgram(0)
        # the environment's mip levels blend across the edges of its faces
        gl.glEnable(gl.GL_TEXTURE_CUBE_MAP_SEAMLESS)

        # every program reads `shaders.FRAME` from the same binding
        frame = np.zeros(1, dtype=shaders.FRAME)
        frame_buffer = upload(gl.GL_UNIFORM_BUFFER, generate(names, "buffers"), frame)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, BLOCK_BINDINGS["Frame"], frame_buffer)
        return RenderContext(
            window=window,
            programs=programs,
            white_texture=upload_image(names, np.full((1, 1, 4), 255), srgb=False),
            empty_vertex_array=generate(names, "vertex_arrays"),
            frame_buffer=frame_buffer,
            names=names,
        )
    except BaseException:
        # a failed context never leaves a window open
        window.close()
        raise


def hidden_context() -> RenderContext:
    """
    The hidden window every offscreen render draws in, created once.

    Returns
    ------------
    context : RenderContext
      A hidden window with compiled shaders.
    """
    try:
        return _HIDDEN_CONTEXT["context"]
    except KeyError:
        context = create_context(visible=False, width=64, height=64, resizable=False)
        # it's the library's and not one of the user's windows, which
        # would keep `pyglet.app.run` from ever returning
        pyglet.app.windows.discard(context.window)
        _HIDDEN_CONTEXT["context"] = context
        return context


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
          Changes to `RENDER_DEFAULTS` like `exposure=2.0`, or starting
          values for `ViewerState` like `smooth=False` or `fixed=["axis"]`:
          anything else raises.
        """
        legacy = [key for key in LEGACY_KEYWORDS if key in settings]
        if len(legacy) > 0:
            warnings.warn(
                f"`SceneViewer` keywords {legacy} will raise after April 2028",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config = settings.pop("window_conf", config)
            settings.update(settings.pop("line_settings", None) or {})
            for key in legacy:
                settings.pop(key, None)
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
        # how the scene is drawn, which is never stored in the scene, and
        # everything the user can toggle, where a misspelled name raises
        self.render_settings = {
            key: settings.pop(key, value) for key, value in RENDER_DEFAULTS.items()
        }
        self.state = replace(ViewerState(), **{**(flags or {}), **settings})
        self.background = np.ones(4)
        if background is not None:
            self.background = to_rgba(background) / 255.0

        # the viewer's own camera, starting where the scene's camera is
        self.camera, camera_transform = current_view(scene)
        self.place_camera(camera_transform)
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

        # geometry name to `GeometryBuffers` and the key it was built from,
        # and the axis, grid and floor which are drawn where they are
        self.geometry_buffers, self.geometry_keys = {}, {}
        self.decoration_buffers = {}
        # the scene graph hash and what it drew where when last checked
        self.synced_graph, self.graph_structure = None, None
        # the box around the scene from where every node is, placed for
        # the scene graph hash and animation time in `placed`, with the
        # shadow maps drawn for the placement `shadowed`
        self.scene_bounds, self.placed, self.shadowed = None, None, None

        self.initial_camera = self.camera_transform.copy()
        # fixed geometry is drawn as if the camera never moved
        self.initial_view = np.linalg.inv(self.initial_camera)
        # the default lights and environment are fixed in the world
        # around the camera the viewer starts at
        lights, transforms = render_lights(scene, self.initial_camera)
        self.lights, self.light_transforms = lights[:-1], transforms[:-1]
        self.environment, self.environment_transform = lights[-1], transforms[-1]
        # a square floor under the lowest corner of the scene which only
        # shows shadows, level with the environment and eight times as wide
        # as the scene, counterclockwise seen from above
        self.ground, bounds = None, scene.bounds
        if self.render_settings["ground_shadow"] and bounds is not None:
            right, up, back = util.unitize(self.environment_transform[:3, :3].T)
            center = bounds.mean(axis=0)
            # a ten thousandth of the scene under it so the scene never touches it
            height = (corners(bounds) @ up).min() - 1e-4 * scene.scale
            base = center + (height - np.dot(center, up)) * up
            square = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64)
            self.ground = base + 4.0 * scene.scale * square @ np.vstack((right, back))
        # what every frame starts from: the environment in world space
        # and the render settings
        self.frame_constants = frame_constants(
            scene, lights, transforms, self.render_settings
        )
        self.light_count = int(self.frame_constants["light_count"][0, 0])
        self.names = {}
        self.allocate(resolution)
        self.reset_view()

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
        # seconds into `scene.animations` which are drawn without posing the
        # scene, or None to draw the scene graph as it is
        self.animation_time = None
        if visible and len(scene.animations) > 0:
            self.animation_time = 0.0
            pyglet.clock.schedule_interval(self.on_animate, 1.0 / 60.0)
        if start_loop and visible:
            self.run()

    def __getattr__(self, name: str):
        """
        The methods of `main`'s `SceneViewer` in `LEGACY_METHODS`,
        which warn and will raise after April 2028.

        Parameters
        ------------
        name : str
          Attribute which isn't otherwise found.

        Returns
        ------------
        method : function
          Toggles a switch or redraws, taking any arguments.
        """
        if name not in LEGACY_METHODS:
            raise AttributeError(f"`SceneViewer` has no attribute `{name}`")
        warnings.warn(
            f"`SceneViewer.{name}` will be removed after April 2028",
            category=DeprecationWarning,
            stacklevel=2,
        )
        switch = LEGACY_METHODS[name]

        def legacy(*args, **kwargs):
            if switch is None:
                self.state.scene_changed = True
            else:
                self.toggle(switch)

        return legacy

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
        for buffers in (
            *self.geometry_buffers.values(),
            *self.decoration_buffers.values(),
        ):
            delete(buffers.names)
        delete(self.names)
        self.geometry_buffers.clear()
        self.decoration_buffers.clear()
        # the hidden window is shared and stays open
        if self.context is not _HIDDEN_CONTEXT.get("context"):
            delete(self.context.names)
            self.window.close()

    def allocate(self, size: Sequence[Integer] | NDArray) -> None:
        """
        Allocate every image in `TARGETS`, the buffers drawing reads every
        frame and the environment's cube map, freeing any from before, and
        fit the camera to a size.

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
        # bloom is blurred at the smallest mip level of the frame below full
        # size with `BLOOM_TEXELS` per standard deviation, so it costs the
        # same at every resolution
        spread = BLOOM_SPREAD * height
        level = int(max(np.floor(np.log2(spread / BLOOM_TEXELS)), 1))
        bloom = self.render_settings["bloom"]
        self.frame_constants["bloom"][0, :3] = (bloom, spread / 2**level, level)
        sizes = {
            "full": (width, height),
            "half": (max(width // 2, 1), max(height // 2, 1)),
            "bloom": (max(width >> level, 1), max(height >> level, 1)),
            "shadow": (SHADOW_RESOLUTION, SHADOW_RESOLUTION),
        }
        limit = []
        for name in (gl.GL_MAX_COLOR_TEXTURE_SAMPLES, gl.GL_MAX_DEPTH_TEXTURE_SAMPLES):
            value = gl.GLint()
            gl.glGetIntegerv(name, ctypes.byref(value))
            limit.append(value.value)
        for name, (color, depth, scale, filtering, layers, samples) in TARGETS.items():
            target_width, target_height = sizes[scale]
            samples = min(samples, *limit)
            kind = gl.GL_TEXTURE_2D
            if samples > 0:
                kind = gl.GL_TEXTURE_2D_MULTISAMPLE
            elif layers > 0:
                kind = gl.GL_TEXTURE_2D_ARRAY
            framebuffer = generate(self.names, "framebuffers")
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
            # the color texture is what other passes read if there is one
            for internal, pixels, attachment in (
                (
                    gl.GL_DEPTH_COMPONENT24 * depth,
                    gl.GL_DEPTH_COMPONENT,
                    gl.GL_DEPTH_ATTACHMENT,
                ),
                (color, gl.GL_RGBA, gl.GL_COLOR_ATTACHMENT0),
            ):
                if internal == 0:
                    continue
                texture = generate(self.names, "textures")
                gl.glBindTexture(kind, texture)
                if samples > 0:
                    gl.glTexImage2DMultisample(
                        kind, samples, internal, target_width, target_height, True
                    )
                elif layers > 0:
                    gl.glTexImage3D(
                        kind, 0, internal, target_width, target_height, layers, 0,
                        pixels, gl.GL_FLOAT, None,
                    )  # fmt: skip
                else:
                    gl.glTexImage2D(
                        kind, 0, internal, target_width, target_height, 0,
                        pixels, gl.GL_FLOAT, None,
                    )  # fmt: skip
                if samples == 0:
                    # magnifying never uses mip levels
                    magnify = filtering
                    if filtering == gl.GL_LINEAR_MIPMAP_LINEAR:
                        magnify = gl.GL_LINEAR
                    for parameter, value in (
                        (gl.GL_TEXTURE_MIN_FILTER, filtering),
                        (gl.GL_TEXTURE_MAG_FILTER, magnify),
                        (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                        (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE),
                    ):
                        gl.glTexParameteri(kind, parameter, value)
                gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, attachment, texture, 0)
                self.textures[name] = texture
            if color == 0:
                # only depth is ever drawn
                gl.glDrawBuffer(gl.GL_NONE)
                gl.glReadBuffer(gl.GL_NONE)
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise ValueError(f"unable to allocate `{name}`: status `{status}`")
            self.framebuffers[name] = framebuffer
            self.sizes[name] = (target_width, target_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        # shadow maps are sampled comparing against their depth, and
        # anything outside of what a light saw is lit
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.textures["shadow"])
        for parameter, value in (
            (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE),
            (gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL),
            (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER),
            (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, parameter, value)
        gl.glTexParameterfv(
            gl.GL_TEXTURE_2D_ARRAY,
            gl.GL_TEXTURE_BORDER_COLOR,
            (gl.GLfloat * 4)(1, 1, 1, 1),
        )
        # every skin's joint matrices, one row per texel, rewritten when
        # anything moves, and two pixel buffers so one frame is copied off
        # the GPU while the next is drawn
        self.joint_buffer = generate(self.names, "buffers")
        upload(gl.GL_TEXTURE_BUFFER, self.joint_buffer, np.eye(4, dtype=np.float32))
        self.textures["joint_matrices"] = generate(self.names, "textures")
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.textures["joint_matrices"])
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA32F, self.joint_buffer)
        self.pixel_buffers = [generate(self.names, "buffers") for _ in range(2)]
        # the environment's cube map with its mip levels blurred like
        # reflections off each roughness, as linear radiance
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
        for parameter, value in (
            (gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
            (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
            (gl.GL_TEXTURE_MAX_LEVEL, len(levels) - 1),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, parameter, value)
        # so the joints and shadows are drawn again
        self.placed, self.shadowed = None, None

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
        self.state.trackball = Trackball(
            pose=self.initial_camera,
            # mouse events are in window pixels, not framebuffer pixels
            size=np.array(self.window.get_size()),
            scale=self.scene.scale,
            target=self.scene.centroid,
        )
        self.camera_transform = self.initial_camera.copy()
        self.state.dirty = True
        # the axis and grid are uploaded with the geometry
        self.state.scene_changed = self.state.scene_changed or flags is not None

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
        # the axis and grid are uploaded with the geometry
        state.scene_changed = state.scene_changed or name in ("axis", "grid")
        if name == "fullscreen":
            self.window.set_fullscreen(state.fullscreen)
        elif name == "fast":
            # the user decided so never switch on the user's behalf
            self.chose_fast = True

    def sync_geometry(self, force: bool = False) -> None:
        """
        Upload geometry which is new or changed since the last draw,
        free any removed from the scene, match the axis, grid and floor
        to the state, and work out what to draw if any of it or which
        nodes draw it changed.

        Parameters
        ------------
        force : bool
          Work out what to draw anyway, i.e. after hiding a node.
        """
        from ...base import Trimesh
        from ...creation import axis
        from ...path.creation import grid
        from ...transformations import translation_matrix

        scene, state, graph = self.scene, self.state, self.scene.graph
        # a key which changes whenever geometry would draw differently,
        # including colors which aren't part of every geometry's hash
        keys = {}
        for name, geometry in scene.geometry.items():
            colors = getattr(geometry, "colors", None)
            if colors is not None:
                colors = hash(np.asarray(colors).tobytes())
            keys[name] = (hash(geometry), hash(getattr(geometry, "visual", None)), colors)
        changed = force or keys != self.geometry_keys
        for name, key in keys.items():
            if self.geometry_keys.get(name) == key:
                continue
            if name in self.geometry_buffers:
                delete(self.geometry_buffers.pop(name).names)
            arrays = geometry_to_arrays(scene.geometry[name])
            if arrays is not None:
                self.geometry_buffers[name] = upload_geometry(arrays)
        for name in set(self.geometry_buffers).difference(keys):
            delete(self.geometry_buffers.pop(name).names)
        self.geometry_keys = keys

        shown = {
            "axis": state.axis,
            "grid": state.grid,
            "ground": self.ground is not None,
        }
        for name, show in shown.items():
            if bool(show) == (name in self.decoration_buffers):
                continue
            changed = True
            if not show:
                delete(self.decoration_buffers.pop(name).names)
                continue
            if name == "axis":
                arrays = geometry_to_arrays(axis(origin_size=scene.scale / 100.0))
            elif name == "grid":
                # a grid just under the scene
                bounds = scene.bounds
                center = bounds.mean(axis=0)
                center[2] = bounds[0][2] - np.ptp(bounds[:, 2]) / 100.0
                side = np.ptp(bounds, axis=0)[:2].max()
                arrays = geometry_to_arrays(
                    grid(side=side, count=4, transform=translation_matrix(center))
                )
            else:
                # a floor which only shows shadows, blended over the
                # background, in a copy of the record cached on the mesh
                floor = Trimesh(self.ground, [[0, 1, 2], [0, 2, 3]], process=False)
                arrays = geometry_to_arrays(floor)
                material = arrays.material.copy()
                material["flags"][[0, 2]] = (shaders.ALPHA_MODES["BLEND"], 1)
                arrays = replace(arrays, material=material)
            self.decoration_buffers[name] = upload_geometry(arrays)

        # every matrix is read every frame, so only the parents and the
        # geometry of every node matter, which only change with the hash
        if hash(graph) != self.synced_graph:
            self.synced_graph = hash(graph)
            forest = graph.transforms
            structure = (
                dict(forest.parents),
                {node: data.get("geometry") for node, data in forest.node_data.items()},
            )
            changed = changed or structure != self.graph_structure
            self.graph_structure = structure
        if not changed:
            return
        self.placed = None

        # what to draw, which only changes with the scene, so drawing a
        # frame only has to look up where each node is: one row per node
        # with geometry and then one for each of the axis, grid and floor
        self.table = table = SimpleNamespace()
        node_data = graph.transforms.node_data
        drawn = {node: node_data[node].get("geometry") for node in graph.nodes_geometry}
        table.nodes = [
            node
            for node, name in drawn.items()
            if name in self.geometry_buffers and node not in state.hidden
        ]
        nodes, names = table.nodes, [drawn[node] for node in table.nodes]
        buffers = [self.geometry_buffers[name] for name in names]
        table.buffers = buffers = buffers + list(self.decoration_buffers.values())
        count = len(nodes)
        # rows sharing an index draw the same buffers, numbered by first
        # appearance so the order doesn't depend on where they are in memory,
        # or the same material
        first = {}
        table.geometry_index = np.array(
            [first.setdefault(id(item), len(first)) for item in buffers], dtype=np.int64
        )
        table.material_index = np.unique(
            [hash(item.material_key) for item in buffers], return_inverse=True
        )[1]
        # rows which are meshes rather than lines or points, translucent meshes
        # drawn last from back to front, the invisible floor, and rows which
        # stay put relative to the camera
        table.mesh = np.array([item.kind == "mesh" for item in buffers], dtype=bool)
        flags = np.reshape([item.material["flags"] for item in buffers], (-1, 4))
        table.blended = table.mesh & (flags[:, 0] == shaders.ALPHA_MODES["BLEND"])
        table.catcher = flags[:, 2] == 1
        table.fixed = np.zeros(len(buffers), dtype=bool)
        table.fixed[:count] = [name in state.fixed for name in names]
        # each row's index into `scene.graph.nodes_geometry` or -1, the
        # center of its vertices for sorting and the corners of their box
        position = {node: index for index, node in enumerate(graph.nodes_geometry)}
        table.node_index = np.full(len(buffers), -1.0)
        table.node_index[:count] = [position[node] for node in nodes]
        table.centroids = np.reshape([item.centroid for item in buffers], (-1, 3))
        table.corners = np.reshape([item.corners for item in buffers], (-1, 8, 3))

        # skinned meshes look up their joints from where their skin's
        # four rows per joint start, with a skin shared by nodes once
        skinned = [
            (row, scene.skins[node])
            for row, node in enumerate(nodes)
            if node in scene.skins and buffers[row].skinned
        ]
        skins = list({id(skin): skin for _, skin in skinned}.values())
        order = {id(skin): index for index, skin in enumerate(skins)}
        skin_index = np.full(len(buffers), -1)
        skin_index[[row for row, _ in skinned]] = [order[id(skin)] for _, skin in skinned]
        starts = np.cumsum([0] + [len(skin.joints) for skin in skins])
        table.joint_offset = np.where(skin_index >= 0, 4.0 * starts[skin_index], -1.0)
        table.inverse_binds = np.concatenate(
            [np.zeros((0, 4, 4))] + [skin.inverse_bind_matrices for skin in skins]
        )
        # a skinned row is anywhere one of its skin's joints could put it,
        # so its box is bounded by every corner under every one of them
        rows = np.flatnonzero(skin_index >= 0)
        table.bound_rows = np.repeat(rows, np.diff(starts)[skin_index[rows]])
        table.bound_joints = np.concatenate(
            [np.zeros(0, dtype=np.int64)]
            + [np.arange(starts[s], starts[s + 1]) for s in skin_index[rows]]
        )
        # skins usually share a skeleton, so each joint is looked up once
        joint_nodes = list(
            dict.fromkeys(joint for skin in skins for joint in skin.joints)
        )
        lookup = {node: index for index, node in enumerate(joint_nodes)}
        table.joint_index = np.array(
            [lookup[joint] for skin in skins for joint in skin.joints], dtype=np.int64
        )
        table.joint_count = len(joint_nodes)
        # the scene's own lights move with the scene graph, and the
        # default lights stay where the viewer started
        scene_lights = []
        if scene.has_lights:
            scene_lights = [light.name for light in self.lights[: self.light_count]]
        table.scene_light_count = len(scene_lights)
        # a render looks through the scene's camera, which may be animated
        cameras = []
        if state.follow_camera and scene.has_camera:
            cameras = [scene.camera.name]
        # every edge between the world and the nodes whose transforms are
        # needed, parents before children, with the row of each edge's
        # parent or -1, the rows of each depth, and the row of each node
        # or -1 for the world itself
        world_nodes = [*nodes, *scene_lights, *joint_nodes, *cameras]
        parents = graph.transforms.parents
        depth = {}
        for node in world_nodes:
            walked = []
            while node in parents and node not in depth:
                walked.append(node)
                node = parents[node]
            above = depth.get(node, -1)
            depth.update(
                (item, above + level) for level, item in enumerate(walked[::-1], 1)
            )
        ordered = sorted(depth, key=depth.get)
        row = {node: index for index, node in enumerate(ordered)}
        table.chain_edges = [(parents[node], node) for node in ordered]
        table.chain_parent = np.array(
            [row.get(parents[node], -1) for node in ordered], dtype=np.int64
        )
        levels = np.array([depth[node] for node in ordered], dtype=np.int64)
        table.chain_levels = [
            np.flatnonzero(levels == level) for level in range(levels.max(initial=-1) + 1)
        ]
        table.world_rows = np.array(
            [row.get(node, -1) for node in world_nodes], dtype=np.int64
        )
        # the row of every node an animation can drive
        table.node_rows = {
            node: index for index, (_, node) in enumerate(table.chain_edges)
        }

    def draw(self, mode: str = "color") -> None:
        """
        Draw the scene into `TARGETS["multisample"]` and resolve it with
        bloom into `resolved`, or draw the view depth and node index of
        every pixel into `index`.

        Parameters
        ------------
        mode : str
          "color" or "index", a key of `shaders.OUTPUT_MODES`.
        """
        state, context = self.state, self.context
        if state.scene_changed:
            self.sync_geometry(force=True)
            state.scene_changed = False
        table, graph = self.table, self.scene.graph
        count = len(table.nodes)

        # the only per-node work: where every node, light and joint is,
        # which only changes with the scene graph and not the camera
        placement = (hash(graph), self.animation_time)
        if self.placed != placement:
            self.placed = placement
            # every edge's matrix, then the world transform of every node
            # from its parent's, one depth of the tree at a time
            edge_data = graph.transforms.edge_data
            chain = np.reshape(
                [edge_data[edge].get("matrix", IDENTITY) for edge in table.chain_edges],
                (-1, 4, 4),
            )
            if self.animation_time is not None:
                # animated nodes where they are now, leaving the scene as it is
                for animation in self.scene.animations:
                    # an animation shorter than the others loops
                    seconds = self.animation_time
                    if animation.duration > 0.0:
                        seconds = seconds % animation.duration
                    # where each is in the chain, or -1 if it isn't drawn
                    rows = np.array(
                        [table.node_rows.get(node, -1) for node in animation.nodes],
                        dtype=np.int64,
                    )
                    chain[rows[rows >= 0]] = animation.at(seconds)[rows >= 0]
            for rows in table.chain_levels[1:]:
                chain[rows] = np.matmul(chain[table.chain_parent[rows]], chain[rows])
            # with the identity for anything which is the world itself
            world = np.concatenate((chain, IDENTITY[None]))[table.world_rows]
            lit = count + table.scene_light_count
            skinned = lit + table.joint_count
            # with an identity for the decorations
            transforms = np.tile(IDENTITY, (len(table.buffers), 1, 1))
            transforms[:count] = world[:count]
            lights = self.light_transforms[: self.light_count]
            if table.scene_light_count > 0:
                lights = world[count:lit]
            # `world(joint) @ inverse_bind` for every joint of every skin
            joints = np.matmul(world[lit:skinned][table.joint_index], table.inverse_binds)
            upload(gl.GL_TEXTURE_BUFFER, self.joint_buffer, joints.astype(np.float32))
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
            self.world = (transforms, lights)
            if len(world) > skinned:
                # the camera rides its node
                self.place_camera(world[skinned])
        transforms, lights = self.world
        view = np.linalg.inv(self.camera_transform)
        model_view = np.matmul(view, transforms)
        model_view[table.fixed] = np.matmul(self.initial_view, transforms[table.fixed])

        # the camera with its planes moved in to just contain the scene so
        # no depth precision is wasted, which only changes depth, where the
        # scale of the camera's node scales its planes and magnification
        camera = self.camera.copy()
        near, far = camera.z_near * self.camera_scale, camera.z_far * self.camera_scale
        camera.z_near, camera.z_far = near, far
        if camera.magnification is not None:
            camera.magnification = camera.magnification * self.camera_scale
        if self.scene_bounds is not None:
            # the scene corners, the origin for the axis, and the floor
            points = np.vstack((corners(self.scene_bounds), np.zeros(3)))
            if self.ground is not None:
                points = np.vstack((points, self.ground))
            depth = -(points @ view[:3, :3].T + view[:3, 3])[:, 2]
            # the same as `Scene.scale` without measuring the scene again
            margin = 0.05 * np.linalg.norm(np.ptp(self.scene_bounds, axis=0))
            z_far = min(depth.max() + margin, far)
            # a 24 bit depth buffer tells apart depths `far^2 / (near * 2^24)`
            # apart at the far plane, which this keeps under 1% of `far`
            z_near = max(depth.min() - margin, near, z_far * 1e-5)
            # unless the scene is entirely outside the camera's planes
            if z_near < z_far:
                camera.z_near, camera.z_far = z_near, z_far

        # everything every program reads once per frame
        frame = self.frame_constants.copy()
        frame["projection"] = camera.projection
        frame["inverse_projection"] = np.linalg.inv(frame["projection"][0])
        frame["viewport_size"][:, :2] = self.sizes["resolved"]
        frame["output_mode"][:, 0] = shaders.OUTPUT_MODES[mode]
        frame["smooth_shading"][:, 0] = state.smooth
        frame["line_widths"][0, :2] = [
            self.render_settings[key] for key in ("line_width", "point_size")
        ]
        # every light in view space, where a light shines down its -Z so
        # its +Z points back towards it, and a position W of zero is a
        # light infinitely far away
        light_view = np.matmul(view, lights)
        towards = util.unitize(light_view[:, :3, 2])
        rows = frame["lights"][0, : self.light_count]
        directional = rows["position"][:, 3:] == 0.0
        rows["position"][:, :3] = np.where(directional, towards, light_view[:, :3, 3])
        rows["direction"][:, :3] = -towards
        # nothing is shadowed until the shadows are drawn
        layers = rows["shadow"][:, 0].copy()
        rows["shadow"][:, 0] = -1.0
        # the environment, which `frame_constants` has from world space
        rotation = frame["environment_rotation"][0]
        rotation[:3, :3] = rotation[:3, :3] @ view[:3, :3].T

        gl.glEnable(gl.GL_DEPTH_TEST)
        # equal depth passes so line segments sharing an endpoint join
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["joint_matrices"])
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.textures["joint_matrices"])
        effects = mode == "color" and not state.fast
        casting = layers >= 0
        if effects and state.shadows and casting.any() and self.scene_bounds is not None:
            if self.shadowed is not self.world:
                # every opaque mesh seen from each shadow casting light with
                # a projection fit around the scene, which only changes with
                # where things are and not with the camera
                self.shadowed = self.world
                self.shadow_fit = shadow_projections(
                    lights[casting],
                    directional[casting, 0],
                    rows["falloff"][casting, 3],
                    self.scene_bounds,
                    SHADOW_RESOLUTION,
                )
                casts = table.mesh & ~table.blended & ~table.fixed
                casts[count:] = False
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers["shadow"])
                gl.glViewport(0, 0, *self.sizes["shadow"])
                # push depth away from the light so a lit surface doesn't
                # shadow itself, more where it's seen at a grazing angle
                gl.glPolygonOffset(2.0, 4.0)
                shadow_frame = frame.copy()
                shadow_frame["output_mode"][:, 0] = shaders.OUTPUT_MODES["depth"]
                for layer, light_clip in enumerate(self.shadow_fit[0]):
                    gl.glFramebufferTextureLayer(
                        gl.GL_FRAMEBUFFER,
                        gl.GL_DEPTH_ATTACHMENT,
                        self.textures["shadow"],
                        0,
                        layer,
                    )
                    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
                    # the light's view is part of its projection
                    shadow_frame["projection"] = light_clip
                    upload(gl.GL_UNIFORM_BUFFER, context.frame_buffer, shadow_frame)
                    self.draw_rows(np.flatnonzero(casts), transforms, "depth")
            clip, texels = self.shadow_fit
            # the shader looks shadows up from view space
            frame["shadow_matrices"][0, : len(clip)] = np.matmul(
                SHADOW_BIAS, np.matmul(clip, self.camera_transform)
            )
            rows["shadow"][casting, 0] = layers[casting]
            rows["shadow"][casting, 1:3] = texels

        # meshes are pushed back slightly so lines on their surface win
        gl.glPolygonOffset(1.0, 1.0)
        # a white image is no occlusion at all
        occlusion = context.white_texture
        if effects and state.ambient_occlusion:
            # how much of the environment nearby surfaces hide from every
            # pixel at half size, from the depth of every opaque mesh
            depth_frame = frame.copy()
            depth_frame["output_mode"][:, 0] = shaders.OUTPUT_MODES["depth"]
            upload(gl.GL_UNIFORM_BUFFER, context.frame_buffer, depth_frame)
            opaque = table.mesh & ~table.blended
            opaque &= (np.arange(len(table.buffers)) < count) | table.catcher
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers["scene_depth"])
            gl.glViewport(0, 0, *self.sizes["scene_depth"])
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            self.draw_rows(np.flatnonzero(opaque), model_view, "depth")
            self.fullscreen("occlusion")
            self.fullscreen("blur")
            occlusion = self.textures["occlusion"]
        upload(gl.GL_UNIFORM_BUFFER, context.frame_buffer, frame)
        for sampler, kind, texture in (
            ("shadow_maps", gl.GL_TEXTURE_2D_ARRAY, self.textures["shadow"]),
            ("ambient_occlusion", gl.GL_TEXTURE_2D, occlusion),
            ("environment", gl.GL_TEXTURE_CUBE_MAP, self.textures["environment"]),
        ):
            gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[sampler])
            gl.glBindTexture(kind, texture)

        output = "index"
        if mode == "color":
            output = "multisample"
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers[output])
        gl.glViewport(0, 0, *self.sizes[output])
        gl.glClearColor(*self.background)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        rows = np.arange(len(table.buffers))
        if mode == "index":
            # background is depth zero and node -1, which a clamped
            # `glClearColor` might not write, and the decorations aren't
            # part of the scene
            gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, -1.0, 0.0, 0.0))
            rows = rows[:count]
        elif self.render_settings["show_environment"]:
            self.fullscreen("background")
        gl.glEnable(gl.GL_DEPTH_TEST)
        SWITCHES[mode == "color"](gl.GL_BLEND)
        # every program writes color already multiplied by its alpha, so
        # a transparent surface can still add its reflections
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.draw_rows(rows, model_view, mode)
        gl.glBindVertexArray(context.empty_vertex_array)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDepthMask(gl.GL_TRUE)
        self.frame_count += 1
        state.dirty = False
        if mode != "color":
            return
        # resolve the antialiasing, then add the bloom
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.framebuffers["multisample"])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.framebuffers["resolved"])
        rectangle = (0, 0, *self.sizes["resolved"])
        gl.glBlitFramebuffer(
            *rectangle, *rectangle, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST
        )
        if self.render_settings["bloom"] > 0.0:
            for name in ("bloom_across", "bloom_down", "bloom_add"):
                self.fullscreen(name)

    def draw_rows(self, rows: NDArray, model_view: NDArray, mode: str) -> None:
        """
        Draw rows of the draw table into the bound framebuffer: opaque
        meshes grouped by material and geometry so every node sharing a
        mesh is one instanced draw, then translucent meshes one at a time
        from the farthest to the nearest.

        Parameters
        ------------
        rows : (k,) int
          Rows of `self.table` to draw.
        model_view : (n, 4, 4) float
          Transform of every row of `self.table` to the view or light.
        mode : str
          A key of `shaders.OUTPUT_MODES`, where "depth" draws meshes
          from both sides as shadows and ambient occlusion need.
        """
        table, state, programs = self.table, self.state, self.context.programs
        blended, mesh, geometry_index = table.blended, table.mesh, table.geometry_index
        # nodes scaled to nothing aren't drawn, which is how glTF hides
        # things and otherwise points and lines would leave a dot
        rows = rows[np.abs(model_view[rows, :3, :3]).max(axis=(1, 2), initial=0.0) > 0.0]
        view_z = model_view[rows, 2]
        depth = (view_z[:, :3] * table.centroids[rows]).sum(axis=1) + view_z[:, 3]
        order = rows[
            np.lexsort(
                (
                    np.where(blended[rows], depth, geometry_index[rows]),
                    np.where(blended[rows], 0, table.material_index[rows]),
                    blended[rows],
                )
            )
        ]
        # normals transform by the inverse transpose, which up to a scale
        # the shader normalizes away is the cofactor matrix, which unlike
        # an inverse exists even for a node animated to zero scale, and
        # a mirror flips the sign: stored transposed for OpenGL, which
        # for the cofactor is the adjugate
        linear = model_view[order, :3, :3]
        instances = np.zeros(len(order), dtype=shaders.INSTANCE)
        instances["model_view"] = model_view[order].transpose((0, 2, 1))
        instances["normal_matrix"] = (
            adjugate(linear) * np.sign(np.linalg.det(linear))[:, None, None]
        )
        instances["node_index"] = table.node_index[order]
        instances["joint_offset"] = table.joint_offset[order]
        # a new draw call whenever the geometry changes and for every
        # translucent mesh and every line
        single = (blended | ~mesh)[order]
        starts = np.flatnonzero(
            (np.diff(geometry_index[order], prepend=-1) != 0) | single
        )
        ends = np.append(starts[1:], len(order))

        # the instances of each draw are a slice of `instances` from here
        size, pointer = instances.itemsize, instances.ctypes.data
        face_colors = gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS["face_colors"]
        translucent = (blended[order[starts]] & (mode == "color")).tolist()
        program, material = None, None
        for start, end, row, hide in zip(
            starts.tolist(), ends.tolist(), order[starts].tolist(), translucent
        ):
            item = table.buffers[row]
            name = "line"
            if item.kind == "mesh":
                name = "mesh"
            if name != program:
                program, material = name, None
                gl.glUseProgram(programs[name].program)
                polygons = gl.GL_FILL
                if name == "mesh" and state.wireframe and mode != "depth":
                    polygons = gl.GL_LINE
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, polygons)
                SWITCHES[name == "mesh"](gl.GL_POLYGON_OFFSET_FILL)
                gl.glDisable(gl.GL_CULL_FACE)
            if hide:
                # translucent meshes don't hide what is behind them
                gl.glDepthMask(gl.GL_FALSE)
            gl.glBindVertexArray(item.vertex_array)
            if name == "line":
                programs[name].set("points", item.kind == "points")
                programs[name].set("model_view", model_view[row])
                programs[name].set("node_index", table.node_index[row])
                gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, item.count)
                continue
            if item.material_key != material:
                material = item.material_key
                for texture in shaders.MATERIAL_TEXTURES:
                    gl.glActiveTexture(gl.GL_TEXTURE0 + shaders.TEXTURE_UNITS[texture])
                    gl.glBindTexture(
                        gl.GL_TEXTURE_2D,
                        item.textures.get(texture, self.context.white_texture),
                    )
                # the fourth flag is a material seen from both sides
                culled = state.cull and mode != "depth" and item.material["flags"][3] == 0
                SWITCHES[bool(culled)](gl.GL_CULL_FACE)
            gl.glBindBufferBase(
                gl.GL_UNIFORM_BUFFER, BLOCK_BINDINGS["Material"], item.material_buffer
            )
            gl.glActiveTexture(face_colors)
            gl.glBindTexture(gl.GL_TEXTURE_BUFFER, item.face_colors)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, item.instance_buffer)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                (end - start) * size,
                pointer + start * size,
                gl.GL_STREAM_DRAW,
            )
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
            if TARGETS[source][3] == gl.GL_LINEAR_MIPMAP_LINEAR:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        program = self.context.programs["fullscreen"]
        gl.glUseProgram(program.program)
        program.set("pass_index", list(shaders.PASSES).index(name))
        gl.glBindVertexArray(self.context.empty_vertex_array)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def render(self, changed: bool = True) -> None:
        """
        Draw the scene into `TARGETS["resolved"]` without reading it
        back, so it can be read without waiting.

        Parameters
        ------------
        changed : bool
          Check if geometry or which nodes draw it changed since the
          last render, rather than only the matrices of the scene graph
          which are read every frame anyway.
        """
        self.window.switch_to()
        if changed:
            self.sync_geometry()
        self.draw()

    def read_pixels(
        self, name: str, slot: int = 0, start: bool = True, finish: bool = True
    ) -> NDArray | None:
        """
        Start copying the color of one of `TARGETS` into a pixel buffer
        without waiting, and wait for that copy and return it, so the
        next frame can be drawn while a frame is copied.

        Parameters
        ------------
        name : str
          A key of `TARGETS` with color in `READ_FORMATS`.
        slot : int
          Which of the two pixel buffers to copy into or out of.
        start : bool
          Start the copy.
        finish : bool
          Wait for the copy started before and return it.

        Returns
        ------------
        pixels : (height, width, channels) uint8 or float32 or None
          RGBA bytes or two floats per pixel with the first row at the
          top as a flipped view, if `finish`.
        """
        pixel_format, kind, channels, dtype = READ_FORMATS[TARGETS[name][0]]
        width, height = self.sizes[name]
        pixels = np.zeros((height, width, channels), dtype=dtype)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pixel_buffers[slot])
        if start:
            gl.glBufferData(
                gl.GL_PIXEL_PACK_BUFFER, pixels.nbytes, None, gl.GL_STREAM_READ
            )
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.framebuffers[name])
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            # into the bound pixel buffer, where a null pointer is offset zero
            gl.glReadPixels(0, 0, width, height, pixel_format, kind, None)
        if finish:
            gl.glGetBufferSubData(
                gl.GL_PIXEL_PACK_BUFFER,
                0,
                pixels.nbytes,
                pixels.ctypes.data_as(ctypes.c_void_p),
            )
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        if finish:
            # OpenGL puts the first row at the bottom
            return pixels[::-1]
        return None

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
        Draw the scene if anything changed and copy it to the window,
        switching to fast drawing once if, a second after starting, the
        last second of drawing averaged slower than `SLOW_FPS`.

        Returns
        ------------
        drawn : bool
          If the scene was drawn rather than just copied.
        """
        self.window.switch_to()
        state = self.state
        drawn = state.dirty or state.scene_changed
        if drawn:
            # uploading geometry is slow once, which isn't drawing speed,
            # and once the quality is chosen there is nothing to time
            timed = not (state.scene_changed or self.chose_fast or state.fast)
            start = time.perf_counter()
            self.draw()
            if timed:
                # wait for the GPU so the time is how long drawing really takes
                gl.glFinish()
                self.draw_seconds.append(time.perf_counter() - start)
                # only the last few seconds matter
                self.draw_seconds = self.draw_seconds[-200:]
                seconds = np.cumsum(self.draw_seconds[::-1])
                # how many of the most recent draws it took to fill a second
                count = int(np.searchsorted(seconds, 1.0)) + 1
                started = time.perf_counter() - self.started > 1.0
                if started and count <= len(seconds) and count < SLOW_FPS:
                    state.fast = self.chose_fast = True
                    util.log.info("drawing slowly so shadows and occlusion are off")
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.framebuffers["resolved"])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        rectangle = (0, 0, *self.sizes["resolved"])
        gl.glBlitFramebuffer(
            *rectangle, *rectangle, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST
        )
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
          True, so pyglet's own handler doesn't change any state.
        """
        size = tuple(self.window.get_framebuffer_size())
        if self.sizes["resolved"] != size:
            self.allocate(size)
        self.state.trackball.resize(np.array([width, height]))
        self.state.dirty = True
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
        Run what `KEY_BINDINGS` binds to a key, or nudge the camera
        with the arrow keys.

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
            elif hasattr(type(self), name):
                getattr(self, name)()
            else:
                getattr(self.window, name)()
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
        looping back to the start, which is drawn without changing
        the scene.

        Parameters
        ------------
        elapsed : float
          Seconds since the last tick.
        """
        if not self.state.playing:
            return
        if self.animation_time is None:
            self.animation_time = 0.0
        self.animation_time += elapsed
        if self.scene.duration > 0.0:
            self.animation_time %= self.scene.duration
        self.state.dirty = True
