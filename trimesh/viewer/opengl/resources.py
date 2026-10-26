"""
resources.py
--------------

Everything that talks to OpenGL. What the bytes in every buffer mean
is declared once in `shaders/__init__.py`, and this module only moves
them with raw OpenGL 3.3 core calls, which pyglet exposes identically
on pyglet 1.5, 2.x and 3.x, using a window from pyglet for the context.

Nothing is ever drawn into the window directly: every frame is drawn
into our own multisampled framebuffer and copied out, which means a
screenshot is a plain `glReadPixels` of memory we allocated and is
the same with visible, hidden, high-DPI and headless windows.
"""

import ctypes
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass

import numpy as np
import pyglet

from ... import util
from ...bounds import corners
from ...scene.cameras import DEFAULT_FOV
from ...typed import Integer, NDArray
from ..conversion import GeometryArrays
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
    "the trimesh viewer needs OpenGL 3.3 core, which without a working display "
    + "on Linux EGL provides with `PYGLET_HEADLESS=1` (i.e. `apt install libegl1`), "
    + "or run under `xvfb-run`"
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

# how many samples to antialias the framebuffer with
MULTISAMPLE_COUNT = 4

# the uniform buffer binding each of `shaders.BLOCKS` is read from
BLOCK_BINDINGS = {name: index for index, name in enumerate(shaders.BLOCKS)}
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

# color storage : (format, type, channels, numpy dtype) to read it with
READ_FORMATS = {
    gl.GL_RGBA8: (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 4, np.uint8),
    gl.GL_RG32F: (gl.GL_RG, gl.GL_FLOAT, 2, np.float32),
}


# a dataclass field holding OpenGL names of a kind, which `delete` frees
NAMES = {
    kind: {"names": kind}
    for kind in (
        "programs",
        "buffers",
        "textures",
        "framebuffers",
        "renderbuffers",
        "vertex_arrays",
    )
}


@dataclass
class ShaderProgram:
    """
    A linked shader program and the location and type of each
    uniform, looked up once when it is linked.
    """

    program: int = field(metadata=NAMES["programs"])
    # uniform name to (location, OpenGL type)
    uniforms: dict = field(default_factory=dict)

    def set(self, name: str, value) -> None:
        """
        Set a uniform on this program, which must be in use.
        Uniforms the compiler optimized out are silently ignored.

        Parameters
        ------------
        name : str
          Uniform name in the shader source.
        value : float, int, bool or array
          Value matching the uniform's type.
        """
        try:
            location, kind = self.uniforms[name]
        except KeyError:
            return
        UNIFORM_SETTERS[kind](location, value)


@dataclass
class Framebuffer:
    """
    A framebuffer allocated by the viewer with a color and a depth
    renderbuffer, which every frame is drawn into.
    """

    framebuffer: int = field(metadata=NAMES["framebuffers"])
    color: int = field(metadata=NAMES["renderbuffers"])
    depth: int = field(metadata=NAMES["renderbuffers"])
    width: int
    height: int
    samples: int
    # the internal format of `color`, a key of `READ_FORMATS`
    storage: int


@dataclass
class GeometryBuffers:
    """
    Handles to one piece of geometry uploaded to the GPU, and what
    is needed to draw it.
    """

    kind: str
    vertex_array: int = field(metadata=NAMES["vertex_arrays"])
    # every buffer: rows, indices, instances, skin, face colors, material
    buffers: list = field(metadata=NAMES["buffers"])
    # indices for meshes, segments for lines and points
    count: int
    # the `shaders.MATERIAL` record, which meshes also have on the GPU
    material: NDArray
    # mesh only: the uniform buffer holding `material`
    material_buffer: int = 0
    # texture name ("base_color_texture", ...) to texture id
    textures: dict = field(default_factory=dict, metadata=NAMES["textures"])
    # mesh only: the buffer texture with face colors or zero
    face_colors: int = field(default=0, metadata=NAMES["textures"])
    # mesh only: per-instance matrices, rewritten before every draw
    instance_buffer: int = 0
    # the material record and texture ids, so consecutive meshes with the
    # same material skip binding textures again
    material_key: bytes = b""
    # the center of the vertices for sorting transparent geometry
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # (8, 3) corners of the box around the vertices
    corners: np.ndarray = field(default_factory=lambda: np.zeros((8, 3)))
    # mesh only: the vertices have joints and weights
    skinned: bool = False


@dataclass
class TextureTargets:
    """
    Textures which are each drawn into through a framebuffer of their
    own, attached once when they are allocated, or for a layered texture
    a layer at a time.
    """

    # name to texture
    textures: dict = field(metadata=NAMES["textures"])
    # name to the framebuffer which draws into that texture
    framebuffers: dict = field(metadata=NAMES["framebuffers"])
    # name to the (width, height) of that texture in pixels
    sizes: dict


@dataclass
class RenderContext:
    """
    A window with an OpenGL context and everything compiled and
    allocated once per context.
    """

    window: pyglet.window.Window
    mesh_program: ShaderProgram
    line_program: ShaderProgram
    # a 1x1 white texture bound in place of any missing texture
    white_texture: int = field(metadata=NAMES["textures"])
    # a vertex array with no attributes for drawing nothing
    empty_vertex_array: int = field(metadata=NAMES["vertex_arrays"])
    # the uniform buffer holding one `shaders.FRAME`
    frame_buffer: int = field(metadata=NAMES["buffers"])
    # draws only depth, from `mesh.vert` and `shadow.frag`
    shadow_program: ShaderProgram
    # screen space ambient occlusion and its blur over full screen triangles
    occlusion_program: ShaderProgram
    blur_program: ShaderProgram
    # the environment behind the scene, when it is seen
    background_program: ShaderProgram
    # light too bright for the display glowing onto its neighbors
    bloom_program: ShaderProgram
    # `depth` with one layer per shadow casting light
    shadow_maps: TextureTargets
    # the joint matrices of every skin, as a buffer and a buffer texture
    joint_buffer: int = field(metadata=NAMES["buffers"])
    joint_texture: int = field(metadata=NAMES["textures"])
    # the `SceneViewer.world` whose joints and shadows are uploaded, which
    # every viewer drawing into a shared hidden window has to check
    placement: object = None


def delete(*items) -> None:
    """
    Free every OpenGL object held by some of the dataclasses here, which
    are the fields marked with a kind of `NAMES`, including those of any
    dataclass they hold, on the context which is current.

    Parameters
    ------------
    items : Framebuffer, GeometryBuffers, TextureTargets or RenderContext
      What to free, which can't be drawn with afterwards.
    """
    names = {kind: [] for kind in NAMES}
    pending = list(items)
    while len(pending) > 0:
        item = pending.pop()
        for item_field in fields(item):
            value = getattr(item, item_field.name)
            if is_dataclass(value):
                pending.append(value)
                continue
            kind = item_field.metadata.get("names")
            if kind is None:
                continue
            if isinstance(value, dict):
                value = list(value.values())
            names[kind].extend(np.ravel(value).astype(np.int64).tolist())
    # a program in use is only deleted once it no longer is
    gl.glUseProgram(0)
    for program in names.pop("programs"):
        gl.glDeleteProgram(program)
    for kind, values in names.items():
        DELETERS[kind](len(values), (gl.GLuint * len(values))(*values))


# how to free each kind of OpenGL name in `NAMES` but programs, which
# are freed one at a time
DELETERS = {
    "buffers": gl.glDeleteBuffers,
    "textures": gl.glDeleteTextures,
    "framebuffers": gl.glDeleteFramebuffers,
    "renderbuffers": gl.glDeleteRenderbuffers,
    "vertex_arrays": gl.glDeleteVertexArrays,
}


def set_matrix(location: int, value) -> None:
    """
    Set a 4x4 matrix uniform on the program in use.

    Parameters
    ------------
    location : int
      Location of the uniform.
    value : (4, 4) float
      Row-major matrix, which is transposed while uploading.
    """
    matrix = np.ascontiguousarray(value, dtype=np.float32)
    pointer = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gl.glUniformMatrix4fv(location, 1, gl.GL_TRUE, pointer)


# how to set a uniform of each OpenGL type the programs set one at a time,
# which is looked up once at link time: every other value is in a block
UNIFORM_SETTERS = {
    gl.GL_FLOAT: gl.glUniform1f,
    gl.GL_FLOAT_VEC2: lambda location, value: gl.glUniform2f(location, *value),
    gl.GL_FLOAT_MAT4: set_matrix,
    # booleans and samplers which are a texture unit
    **dict.fromkeys(
        (
            gl.GL_BOOL,
            gl.GL_SAMPLER_2D,
            gl.GL_SAMPLER_BUFFER,
            gl.GL_SAMPLER_2D_ARRAY_SHADOW,
        ),
        gl.glUniform1i,
    ),
}


def create_context(
    width: int,
    height: int,
    visible: bool = True,
    resizable: bool = True,
    fullscreen: bool = False,
    caption: str = "trimesh",
    config=None,
) -> RenderContext:
    """
    Open a window with an OpenGL 3.3 core context and compile
    the shaders.

    Parameters
    ------------
    width, height : int
      Window size in pixels.
    visible : bool
      Show the window on screen.
    resizable : bool
      Allow the window to be resized.
    fullscreen : bool
      Open the window fullscreen.
    caption : str
      Window title.
    config : None or pyglet config
      Override the context configuration.

    Returns
    ------------
    context : RenderContext
      The window and the compiled shaders.
    """
    options = {
        "width": int(width),
        "height": int(height),
        "visible": visible,
        "resizable": resizable,
        "fullscreen": fullscreen,
        "caption": caption,
    }
    if config is None:
        try:
            config = gl.Config(double_buffer=True)
        except AttributeError:
            # pyglet >= 3 has no `gl.Config` and defaults to 3.3 core
            config = None
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
        window = pyglet.window.Window(config=config, **options)
    except CONTEXT_ERRORS:
        try:
            # compatibility contexts on Linux and Windows also support 3.3
            window = pyglet.window.Window(**options)
        except CONTEXT_ERRORS as error:
            message = f"unable to create an OpenGL context: {CONTEXT_HINT}"
            raise RuntimeError(message) from error
    try:
        return create_render_context(window)
    except BaseException:
        # a failed context never leaves a window open
        window.close()
        raise


def create_render_context(window: pyglet.window.Window) -> RenderContext:
    """
    Check a new window's context is OpenGL 3.3 or newer, then compile
    the shaders and allocate everything shared by every frame.

    Parameters
    ------------
    window : pyglet.window.Window
      A window with a new context.

    Returns
    ------------
    context : RenderContext
      The window and the compiled shaders.
    """
    window.switch_to()
    # every OpenGL version starts its version string with "major.minor"
    version, renderer = (
        ctypes.cast(gl.glGetString(name), ctypes.c_char_p).value.decode("utf-8")
        for name in (gl.GL_VERSION, gl.GL_RENDERER)
    )
    if tuple(int(part) for part in version.split(" ")[0].split(".")[:2]) < (3, 3):
        raise RuntimeError(f"`{renderer}` only has OpenGL `{version}`: {CONTEXT_HINT}")

    white_texture = upload_image(np.full((1, 1, 4), 255, dtype=np.uint8), srgb=False)
    empty_vertex_array = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(empty_vertex_array))

    # every program reads `shaders.FRAME` from the same binding
    frame_buffer = create_buffer(gl.GL_UNIFORM_BUFFER, np.zeros(1, dtype=shaders.FRAME))
    gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, BLOCK_BINDINGS["Frame"], frame_buffer)

    # every skin's joint matrices, one row per texel, rewritten each frame
    joint_buffer = create_buffer(gl.GL_TEXTURE_BUFFER, np.eye(4, dtype=np.float32))
    joint_texture = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(joint_texture))
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, joint_texture)
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA32F, joint_buffer)

    return RenderContext(
        window=window,
        joint_buffer=joint_buffer,
        joint_texture=joint_texture.value,
        mesh_program=compile_program("mesh"),
        line_program=compile_program("line"),
        white_texture=white_texture,
        empty_vertex_array=empty_vertex_array.value,
        frame_buffer=frame_buffer,
        shadow_program=compile_program("mesh", fragment="shadow"),
        occlusion_program=compile_program("fullscreen", fragment="occlusion"),
        blur_program=compile_program("fullscreen", fragment="blur"),
        background_program=compile_program("fullscreen", fragment="background"),
        bloom_program=compile_program("fullscreen", fragment="bloom"),
        shadow_maps=create_shadow_maps(SHADOW_RESOLUTION),
    )


def compile_program(name: str, fragment: str | None = None) -> ShaderProgram:
    """
    Compile and link `shaders/{name}.vert.glsl` and `shaders/{name}.frag.glsl`
    after filling them in with `shaders.shader_source`.

    Parameters
    ------------
    name : str
      Name of the shader pair in `trimesh/viewer/opengl/shaders`.
    fragment : str or None
      Use `shaders/{fragment}.frag.glsl` instead.

    Returns
    ------------
    program : ShaderProgram
      The linked program and its uniform locations.
    """
    if fragment is None:
        fragment = name
    program = gl.glCreateProgram()
    for file_name, kind in (
        (f"{name}.vert", gl.GL_VERTEX_SHADER),
        (f"{fragment}.frag", gl.GL_FRAGMENT_SHADER),
    ):
        source = shaders.shader_source(file_name).encode("utf-8")
        shader = gl.glCreateShader(kind)
        buffer = ctypes.create_string_buffer(source)
        pointer = ctypes.cast(
            ctypes.pointer(ctypes.pointer(buffer)),
            ctypes.POINTER(ctypes.POINTER(gl.GLchar)),
        )
        gl.glShaderSource(shader, 1, pointer, None)
        gl.glCompileShader(shader)
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            message = ctypes.create_string_buffer(4096)
            gl.glGetShaderInfoLog(shader, 4096, None, message)
            log = message.value.decode("utf-8", errors="replace")
            raise ValueError(f"`{file_name}` failed to compile:\n{log}")
        gl.glAttachShader(program, shader)
        # the shader is freed when the program is
        gl.glDeleteShader(shader)

    gl.glLinkProgram(program)
    status = gl.GLint()
    gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        message = ctypes.create_string_buffer(4096)
        gl.glGetProgramInfoLog(program, 4096, None, message)
        log = message.value.decode("utf-8", errors="replace")
        raise ValueError(f"`{name}` failed to link:\n{log}")

    # read every block from its binding unless the program never uses it
    for block_name, binding in BLOCK_BINDINGS.items():
        block = gl.glGetUniformBlockIndex(
            program, ctypes.create_string_buffer(block_name.encode("utf-8"))
        )
        if block != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, block, binding)

    # look up the location and type of every uniform once
    count = gl.GLint()
    gl.glGetProgramiv(program, gl.GL_ACTIVE_UNIFORMS, ctypes.byref(count))
    uniforms = {}
    for index in range(count.value):
        size, kind = gl.GLint(), gl.GLenum()
        uniform_name = ctypes.create_string_buffer(256)
        gl.glGetActiveUniform(
            program,
            index,
            256,
            None,
            ctypes.byref(size),
            ctypes.byref(kind),
            uniform_name,
        )
        decoded = uniform_name.value.decode("utf-8")
        uniforms[decoded] = (gl.glGetUniformLocation(program, uniform_name), kind.value)
    result = ShaderProgram(program=program, uniforms=uniforms)
    # samplers read from a fixed texture unit for the life of the program
    gl.glUseProgram(program)
    for sampler, unit in shaders.TEXTURE_UNITS.items():
        result.set(sampler, unit)
    gl.glUseProgram(0)
    return result


def create_buffer(target: int, data: np.ndarray) -> int:
    """
    Copy a numpy array into a new OpenGL buffer.

    Parameters
    ------------
    target : int
      Buffer type, i.e. `GL_ARRAY_BUFFER`.
    data : numpy.ndarray
      Values to copy.

    Returns
    ------------
    buffer : int
      The OpenGL buffer, which is left bound to `target`.
    """
    buffer = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(buffer))
    upload_array(target, buffer.value, data, usage=gl.GL_STATIC_DRAW)
    return buffer.value


def attribute_pointers(dtype: np.dtype, locations: dict, per_instance: bool) -> None:
    """
    Point vertex shader inputs at the rows of a structured array in the
    bound `GL_ARRAY_BUFFER`, one input per field of its dtype.

    Parameters
    ------------
    dtype : numpy.dtype
      A dtype from `layout`, or a subset of one.
    locations : dict
      {field name : location} from `shaders.attribute_locations`.
    per_instance : bool
      Advance once per instance rather than once per vertex.
    """
    for name in dtype.names:
        field, offset = dtype.fields[name][:2]
        # a matrix is one input per column, each the size of a row
        # because `layout` stores matrices transposed, and a scalar is
        # one column of width one
        columns, width = (1, 1, *field.shape)[-2:]
        # bytes are normalized to 0.0-1.0 and everything else is float32
        normalized = field.base.itemsize == 1
        if normalized:
            kind = gl.GL_UNSIGNED_BYTE
        else:
            kind = gl.GL_FLOAT
        for column in range(columns):
            location = locations[name] + column
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(
                location,
                width,
                kind,
                normalized,
                dtype.itemsize,
                ctypes.c_void_p(offset + column * width * field.base.itemsize),
            )
            gl.glVertexAttribDivisor(location, int(per_instance))


def upload_array(
    target: int, buffer: int, data: np.ndarray, usage: int = gl.GL_STREAM_DRAW
) -> None:
    """
    Replace the contents of a buffer, which by default changes every
    frame, such as `shaders.INSTANCE` rows or one `shaders.FRAME`.

    Parameters
    ------------
    target : int
      Buffer type, i.e. `GL_ARRAY_BUFFER` or `GL_UNIFORM_BUFFER`.
    buffer : int
      The OpenGL buffer, which is left bound to `target`.
    data : numpy.ndarray
      Structured array to copy.
    usage : int
      How often it changes, i.e. `GL_STATIC_DRAW` for never.
    """
    data = np.ascontiguousarray(data)
    gl.glBindBuffer(target, buffer)
    gl.glBufferData(target, data.nbytes, data.ctypes.data, usage)


def upload_geometry(arrays: GeometryArrays) -> GeometryBuffers:
    """
    Copy converted geometry onto the GPU.

    Parameters
    ------------
    arrays : GeometryArrays
      Output of `conversion.geometry_to_arrays`.

    Returns
    ------------
    buffers : GeometryBuffers
      Handles to draw and later delete the geometry.
    """
    vertex_array = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(vertex_array))
    gl.glBindVertexArray(vertex_array)
    rows = arrays.rows
    if arrays.kind != "mesh":
        # each segment is one instance and the shader makes a quad around it
        points = np.concatenate((rows["segment_start"], rows["segment_end"]))
        buffers = [create_buffer(gl.GL_ARRAY_BUFFER, rows)]
        attribute_pointers(
            shaders.SEGMENT,
            shaders.attribute_locations(shaders.SEGMENT),
            per_instance=True,
        )
        gl.glBindVertexArray(0)
        return GeometryBuffers(
            kind=arrays.kind,
            vertex_array=vertex_array.value,
            buffers=buffers,
            count=len(rows),
            material=arrays.material,
            centroid=points.mean(axis=0),
            corners=corners([points.min(axis=0), points.max(axis=0)]),
        )

    points = rows["position"]
    locations = shaders.attribute_locations(
        shaders.VERTEX, shaders.SKIN, shaders.INSTANCE
    )
    buffers = [create_buffer(gl.GL_ARRAY_BUFFER, rows)]
    attribute_pointers(shaders.VERTEX, locations, per_instance=False)
    buffers.append(create_buffer(gl.GL_ELEMENT_ARRAY_BUFFER, arrays.faces))
    # every node drawing this mesh is one instance, rewritten every frame
    buffers.append(create_buffer(gl.GL_ARRAY_BUFFER, np.zeros(1, dtype=shaders.INSTANCE)))
    attribute_pointers(shaders.INSTANCE, locations, per_instance=True)
    if arrays.skin is not None:
        buffers.append(create_buffer(gl.GL_ARRAY_BUFFER, arrays.skin))
        attribute_pointers(shaders.SKIN, locations, per_instance=False)
    gl.glBindVertexArray(0)

    face_colors = 0
    if arrays.face_colors is not None:
        # face colors are looked up with `gl_PrimitiveID` in a buffer texture
        buffers.append(create_buffer(gl.GL_TEXTURE_BUFFER, arrays.face_colors))
        texture = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(texture))
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, texture)
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA8, buffers[-1])
        face_colors = texture.value

    textures = {
        name: upload_image(np.asarray(image.convert("RGBA")), srgb=srgb)
        for name, image in arrays.textures.items()
        for srgb in [shaders.MATERIAL_TEXTURES[name]]
    }
    # the material in a uniform buffer of its own, bound for every draw
    buffers.append(create_buffer(gl.GL_UNIFORM_BUFFER, arrays.material))
    texture_ids = [textures.get(name, 0) for name in shaders.MATERIAL_TEXTURES]
    return GeometryBuffers(
        kind="mesh",
        vertex_array=vertex_array.value,
        buffers=buffers,
        count=arrays.faces.size,
        material=arrays.material,
        material_buffer=buffers[-1],
        textures=textures,
        face_colors=face_colors,
        instance_buffer=buffers[2],
        material_key=arrays.material.tobytes() + np.array(texture_ids).tobytes(),
        centroid=points.mean(axis=0),
        corners=corners([points.min(axis=0), points.max(axis=0)]),
        skinned=arrays.skin is not None,
    )


def upload_image(pixels: np.ndarray, srgb: bool) -> int:
    """
    Copy an RGBA image into a mipmapped, repeating texture.

    Parameters
    ------------
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
    texture = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(texture))
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    if srgb:
        internal = gl.GL_SRGB8_ALPHA8
    else:
        internal = gl.GL_RGBA8
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        internal,
        pixels.shape[1],
        pixels.shape[0],
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        pixels.ctypes.data_as(ctypes.c_void_p),
    )
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR
    )
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    return texture.value


def create_shadow_maps(resolution: int) -> TextureTargets:
    """
    Allocate a depth texture with one layer per shadow casting light,
    which the mesh shader compares depths against with filtering so
    shadow edges are smooth.

    Parameters
    ------------
    resolution : int
      Width and height of each layer in pixels.

    Returns
    ------------
    shadow_maps : TextureTargets
      `depth`, a texture array, and a framebuffer to attach each layer to.
    """
    texture = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(texture))
    target = gl.GL_TEXTURE_2D_ARRAY
    gl.glBindTexture(target, texture)
    gl.glTexImage3D(
        target,
        0,
        gl.GL_DEPTH_COMPONENT24,
        resolution,
        resolution,
        shaders.MAX_SHADOWS,
        0,
        gl.GL_DEPTH_COMPONENT,
        gl.GL_FLOAT,
        None,
    )
    # sampling compares against the stored depth, and linear filtering
    # blends four of those comparisons into a soft edge
    for name, value in (
        (gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR),
        (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
        (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE),
        (gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL),
        # anything outside of what a light saw is lit
        (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER),
        (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER),
    ):
        gl.glTexParameteri(target, name, value)
    gl.glTexParameterfv(
        target, gl.GL_TEXTURE_BORDER_COLOR, (gl.GLfloat * 4)(1.0, 1.0, 1.0, 1.0)
    )

    framebuffer = gl.GLuint()
    gl.glGenFramebuffers(1, ctypes.byref(framebuffer))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
    # only depth is ever drawn
    gl.glDrawBuffer(gl.GL_NONE)
    gl.glReadBuffer(gl.GL_NONE)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    return TextureTargets(
        textures={"depth": texture.value},
        framebuffers={"depth": framebuffer.value},
        sizes={"depth": (resolution, resolution)},
    )


def create_texture_targets(formats: dict) -> TextureTargets:
    """
    Allocate 2D textures clamped at their edges, each with a framebuffer
    drawing into it: depth textures as its depth and others as its color.

    Parameters
    ------------
    formats : dict
      {name : (internal format, (width, height), filtering)}

    Returns
    ------------
    targets : TextureTargets
      Every texture and the framebuffer drawing into it.
    """
    textures, framebuffers = {}, {}
    for name, (internal, size, filtering) in formats.items():
        texture, framebuffer = gl.GLuint(), gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(texture))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        depth = internal == gl.GL_DEPTH_COMPONENT24
        if depth:
            pixel_format, attachment = gl.GL_DEPTH_COMPONENT, gl.GL_DEPTH_ATTACHMENT
        else:
            pixel_format, attachment = gl.GL_RGBA, gl.GL_COLOR_ATTACHMENT0
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, internal, *size, 0, pixel_format, gl.GL_FLOAT, None
        )
        for parameter, value in (
            (gl.GL_TEXTURE_MIN_FILTER, filtering),
            (gl.GL_TEXTURE_MAG_FILTER, filtering),
            (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
            (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, value)
        gl.glGenFramebuffers(1, ctypes.byref(framebuffer))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, attachment, gl.GL_TEXTURE_2D, texture, 0
        )
        if depth:
            # only depth is ever drawn
            gl.glDrawBuffer(gl.GL_NONE)
            gl.glReadBuffer(gl.GL_NONE)
        textures[name], framebuffers[name] = texture.value, framebuffer.value
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    sizes = {name: tuple(size) for name, (_, size, _) in formats.items()}
    return TextureTargets(textures=textures, framebuffers=framebuffers, sizes=sizes)


def create_ambient_occlusion(width: Integer, height: Integer) -> TextureTargets:
    """
    Allocate the half resolution images for screen space ambient
    occlusion: the scene's depth, the occlusion with noise, and blurred.

    Parameters
    ------------
    width, height : int
      Size of the full resolution framebuffer in pixels.

    Returns
    ------------
    occlusion : TextureTargets
      The `depth`, `noisy` and `blurred` images.
    """
    size = max(int(width) // 2, 1), max(int(height) // 2, 1)
    formats = {
        "depth": (gl.GL_DEPTH_COMPONENT24, size, gl.GL_NEAREST),
        "noisy": (gl.GL_R8, size, gl.GL_NEAREST),
        # the blurred occlusion is scaled up to the full image smoothly
        "blurred": (gl.GL_R8, size, gl.GL_LINEAR),
    }
    return create_texture_targets(formats)


def bloom_level(height: Integer) -> tuple[int, float]:
    """
    The mip level of the half size picture bloom is blurred at, the
    smallest with `BLOOM_TEXELS` texels per standard deviation of the
    glow, so blurring costs the same at every resolution.

    Parameters
    ------------
    height : int
      Height of the full resolution framebuffer in pixels.

    Returns
    ------------
    level : int
      Mip level of the half size picture which is blurred.
    spread : float
      Standard deviation of the glow in texels of that level.
    """
    spread = BLOOM_SPREAD * max(int(height) // 2, 1)
    level = int(max(np.floor(np.log2(spread / BLOOM_TEXELS)), 0))
    return level, spread / 2**level


def create_bloom(width: Integer, height: Integer) -> TextureTargets:
    """
    Allocate the images bloom is drawn in: a half size `picture` with
    mip levels down to `bloom_level`, blurred `across` and then into
    `blurred` at the size of that level.

    Parameters
    ------------
    width, height : int
      Size of the full resolution framebuffer in pixels.

    Returns
    ------------
    bloom : TextureTargets
      The images and their sizes.
    """
    level, _ = bloom_level(height)
    width, height = max(int(width) // 2, 1), max(int(height) // 2, 1)
    blur = max(width >> level, 1), max(height >> level, 1)
    targets = create_texture_targets(
        {
            "picture": (gl.GL_RGBA8, (width, height), gl.GL_LINEAR),
            "across": (gl.GL_RGBA16F, blur, gl.GL_LINEAR),
            "blurred": (gl.GL_RGBA16F, blur, gl.GL_LINEAR),
        }
    )
    # only the mip levels down to the one which is blurred are made
    gl.glBindTexture(gl.GL_TEXTURE_2D, targets.textures["picture"])
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR
    )
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, level)
    return targets


def create_framebuffer(
    width: Integer,
    height: Integer,
    samples: Integer = MULTISAMPLE_COUNT,
    storage: int = gl.GL_RGBA8,
) -> Framebuffer:
    """
    Allocate a framebuffer with a color buffer and a depth buffer.

    Parameters
    ------------
    width, height : int
      Size in pixels.
    samples : int
      Antialiasing samples per pixel, or zero for none. If the driver
      can't multisample this falls back to zero.
    storage : int
      Color format, `GL_RGBA8` for images or `GL_RG32F` for two floats.

    Returns
    ------------
    framebuffer : Framebuffer
      The complete framebuffer.
    """
    maximum = gl.GLint()
    gl.glGetIntegerv(gl.GL_MAX_SAMPLES, ctypes.byref(maximum))
    samples = int(min(samples, maximum.value))

    framebuffer = gl.GLuint()
    gl.glGenFramebuffers(1, ctypes.byref(framebuffer))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
    renderbuffers = (gl.GLuint * 2)()
    gl.glGenRenderbuffers(2, renderbuffers)
    for renderbuffer, internal, attachment in (
        (renderbuffers[0], storage, gl.GL_COLOR_ATTACHMENT0),
        (renderbuffers[1], gl.GL_DEPTH_COMPONENT24, gl.GL_DEPTH_ATTACHMENT),
    ):
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, renderbuffer)
        gl.glRenderbufferStorageMultisample(
            gl.GL_RENDERBUFFER, samples, internal, int(width), int(height)
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, attachment, gl.GL_RENDERBUFFER, renderbuffer
        )

    result = Framebuffer(
        framebuffer=framebuffer.value,
        color=renderbuffers[0],
        depth=renderbuffers[1],
        width=int(width),
        height=int(height),
        samples=samples,
        storage=storage,
    )
    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        delete(result)
        if samples > 0:
            util.log.debug("multisampled framebuffer unsupported: retrying without")
            return create_framebuffer(width, height, samples=0, storage=storage)
        raise ValueError(f"unable to create a framebuffer: status `{status}`")
    return result


def copy_framebuffer(source: Framebuffer, destination: int) -> None:
    """
    Copy color from our framebuffer into another framebuffer, which
    also resolves multisampling.

    Parameters
    ------------
    source : Framebuffer
      Framebuffer to copy from.
    destination : int
      Framebuffer to copy into, where zero is the window.
    """
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, source.framebuffer)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, destination)
    # the same rectangle in both: copy every pixel one to one
    rectangle = (0, 0, source.width, source.height)
    gl.glBlitFramebuffer(*rectangle, *rectangle, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)


def read_framebuffer(framebuffer: Framebuffer) -> np.ndarray:
    """
    Read the color of a single-sample framebuffer into numpy, waiting
    for the GPU to finish drawing it.

    Parameters
    ------------
    framebuffer : Framebuffer
      Framebuffer with zero samples.

    Returns
    ------------
    pixels : (height, width, channels) uint8 or float32
      RGBA bytes or two floats per pixel, with the first row at the top.
    """
    buffer = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(buffer))
    try:
        read_framebuffer_start(framebuffer, buffer.value)
        return read_framebuffer_finish(framebuffer, buffer.value)
    finally:
        gl.glDeleteBuffers(1, ctypes.byref(buffer))


def read_framebuffer_start(framebuffer: Framebuffer, buffer: int) -> None:
    """
    Start copying the color of a single-sample framebuffer into a pixel
    buffer on the GPU without waiting for it, so the next frame can be
    drawn while this one is copied.

    Parameters
    ------------
    framebuffer : Framebuffer
      Framebuffer with zero samples.
    buffer : int
      OpenGL buffer to copy into, resized to fit.
    """
    pixel_format, kind, channels, dtype = READ_FORMATS[framebuffer.storage]
    size = framebuffer.width * framebuffer.height * channels * np.dtype(dtype).itemsize
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, buffer)
    gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, size, None, gl.GL_STREAM_READ)
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, framebuffer.framebuffer)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    # into the bound pixel buffer, where a null pointer is offset zero
    gl.glReadPixels(0, 0, framebuffer.width, framebuffer.height, pixel_format, kind, None)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)


def read_framebuffer_finish(framebuffer: Framebuffer, buffer: int) -> np.ndarray:
    """
    Wait for a copy started with `read_framebuffer_start` and return it.

    Parameters
    ------------
    framebuffer : Framebuffer
      The framebuffer which was copied.
    buffer : int
      The OpenGL buffer it was copied into.

    Returns
    ------------
    pixels : (height, width, channels) uint8 or float32
      RGBA bytes or two floats per pixel, with the first row at the top
      as a flipped view so no time is spent copying it here.
    """
    _, _, channels, dtype = READ_FORMATS[framebuffer.storage]
    pixels = np.zeros((framebuffer.height, framebuffer.width, channels), dtype=dtype)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, buffer)
    gl.glGetBufferSubData(
        gl.GL_PIXEL_PACK_BUFFER, 0, pixels.nbytes, pixels.ctypes.data_as(ctypes.c_void_p)
    )
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
    # OpenGL puts the first row at the bottom
    return pixels[::-1]
