"""
opengl
-----------

Everything that talks to OpenGL. What the bytes in every buffer mean
is declared once in `shaders.py`, and this module only moves them. It only uses raw OpenGL 3.3 core
calls, which pyglet exposes identically on pyglet 1.5, 2.x and 3.x,
and a window from pyglet to own the context.

Nothing is ever drawn into the window directly: every frame is drawn
into our own multisampled framebuffer and copied out, which means a
screenshot is a plain `glReadPixels` of memory we allocated and is
the same with visible, hidden, high-DPI and headless windows.
"""

import ctypes
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pyglet

from ... import util
from ...bounds import corners
from ...typed import Integer
from ..conversion import GeometryArrays, MaterialUniforms
from . import shaders

# with no display on Linux use EGL to get a context: this has to
# be set before importing `gl` which connects to a display
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pyglet.options["headless"] = True
# checking for an error after every call doubles the cost of drawing
pyglet.options["debug_gl"] = False

try:
    # pyglet >= 3
    from pyglet.graphics.api import gl
except ImportError:
    # pyglet < 3
    from pyglet import gl

# how many samples to antialias the framebuffer with
MULTISAMPLE_COUNT = 4

# the uniform buffer binding `shaders.FRAME` is read from
FRAME_BINDING = 0
# width and height of every shadow map in pixels
SHADOW_RESOLUTION = 2048

# color storage : (format, type, channels, numpy dtype) to read it with
READ_FORMATS = {
    gl.GL_RGBA8: (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 4, np.uint8),
    gl.GL_RG32F: (gl.GL_RG, gl.GL_FLOAT, 2, np.float32),
}


@dataclass
class ShaderProgram:
    """
    A linked shader program and the location and type of each
    uniform, looked up once when it is linked.
    """

    program: int
    # uniform name to (location, OpenGL type)
    uniforms: dict

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

    framebuffer: int
    color: int
    depth: int
    width: int
    height: int
    samples: int
    # the internal format of `color`, a key of `READ_FORMATS`
    storage: int

    def delete(self) -> None:
        """
        Free the GPU memory for this framebuffer.
        """
        gl.glDeleteFramebuffers(1, ctypes.byref(gl.GLuint(self.framebuffer)))
        renderbuffers = (gl.GLuint * 2)(self.color, self.depth)
        gl.glDeleteRenderbuffers(2, renderbuffers)


@dataclass
class GeometryBuffers:
    """
    Handles to one piece of geometry uploaded to the GPU, and what
    is needed to draw it.
    """

    kind: str
    vertex_array: int
    # every buffer to delete: vertex attributes, indices, face colors
    buffers: list
    # indices for meshes, segments for lines and points
    count: int
    material: MaterialUniforms
    # texture name ("base_color_texture", ...) to texture id
    textures: dict = field(default_factory=dict)
    # mesh only: the buffer texture with face colors or zero
    face_colors: int = 0
    # mesh only: per-instance matrices, rewritten before every draw
    instance_buffer: int = 0
    # mesh only: every material value and texture, so consecutive
    # meshes with the same material skip setting it again
    material_key: tuple = ()
    # the center of the vertices for sorting transparent geometry
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # (8, 3) corners of the box around the vertices
    corners: np.ndarray = field(default_factory=lambda: np.zeros((8, 3)))
    # mesh only: the vertices have joints and weights
    skinned: bool = False

    def delete(self) -> None:
        """
        Free the GPU memory for this geometry.
        """
        gl.glDeleteVertexArrays(1, ctypes.byref(gl.GLuint(self.vertex_array)))
        buffers = (gl.GLuint * len(self.buffers))(*self.buffers)
        gl.glDeleteBuffers(len(self.buffers), buffers)
        textures = [*self.textures.values(), self.face_colors]
        gl.glDeleteTextures(len(textures), (gl.GLuint * len(textures))(*textures))


@dataclass
class ShadowMaps:
    """
    One depth texture layer per shadow casting light, and the
    framebuffer each layer is rendered into from its light.
    """

    framebuffer: int
    # a `GL_TEXTURE_2D_ARRAY` of depth with `shaders.MAX_SHADOWS` layers
    texture: int
    resolution: int


@dataclass
class AmbientOcclusion:
    """
    Half resolution images for screen space ambient occlusion: the
    scene's depth, the occlusion with noise, and the occlusion blurred,
    all drawn through one framebuffer.
    """

    framebuffer: int
    depth: int
    noisy: int
    blurred: int
    width: int
    height: int

    def delete(self) -> None:
        """
        Free the GPU memory for these images.
        """
        gl.glDeleteFramebuffers(1, ctypes.byref(gl.GLuint(self.framebuffer)))
        textures = (gl.GLuint * 3)(self.depth, self.noisy, self.blurred)
        gl.glDeleteTextures(3, textures)


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
    white_texture: int
    # a vertex array with no attributes for drawing nothing
    empty_vertex_array: int
    # the uniform buffer holding one `shaders.FRAME`
    frame_buffer: int
    # draws only depth, from `mesh.vert` and `shadow.frag`
    shadow_program: ShaderProgram
    # screen space ambient occlusion and its blur over full screen triangles
    occlusion_program: ShaderProgram
    blur_program: ShaderProgram
    # the environment behind the scene, when it is seen
    background_program: ShaderProgram
    shadow_maps: ShadowMaps
    # the joint matrices of every skin, as a buffer and a buffer texture
    joint_buffer: int
    joint_texture: int


def set_matrix(location: int, value) -> None:
    """
    Set a 3x3 or 4x4 matrix uniform on the program in use.

    Parameters
    ------------
    location : int
      Location of the uniform.
    value : (3, 3) or (4, 4) float
      Row-major matrix, which is transposed while uploading.
    """
    matrix = np.ascontiguousarray(value, dtype=np.float32)
    pointer = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if len(matrix) == 4:
        gl.glUniformMatrix4fv(location, 1, gl.GL_TRUE, pointer)
    else:
        gl.glUniformMatrix3fv(location, 1, gl.GL_TRUE, pointer)


# how to set a uniform of each OpenGL type, which is looked up once at link time
UNIFORM_SETTERS = {
    gl.GL_FLOAT: gl.glUniform1f,
    gl.GL_FLOAT_VEC2: lambda location, value: gl.glUniform2f(location, *value),
    gl.GL_FLOAT_VEC3: lambda location, value: gl.glUniform3f(location, *value),
    gl.GL_FLOAT_VEC4: lambda location, value: gl.glUniform4f(location, *value),
    gl.GL_FLOAT_MAT3: set_matrix,
    gl.GL_FLOAT_MAT4: set_matrix,
    # integers, booleans and samplers which are a texture unit
    **dict.fromkeys(
        (
            gl.GL_INT,
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
    width : int
      Window width in pixels.
    height : int
      Window height in pixels.
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
            # pyglet < 3: macOS and pyglet 1.5 only hand out a
            # modern context if one is explicitly requested
            config = gl.Config(
                major_version=3,
                minor_version=3,
                forward_compatible=True,
                double_buffer=True,
            )
        except AttributeError:
            # pyglet >= 3 has no `gl.Config` and defaults to 3.3 core
            config = None
    try:
        window = pyglet.window.Window(config=config, **options)
    except pyglet.window.NoSuchConfigException:
        # compatibility contexts on Linux and Windows also support 3.3
        window = pyglet.window.Window(**options)
    window.switch_to()

    white_texture = upload_image(np.full((1, 1, 4), 255, dtype=np.uint8), srgb=False)
    empty_vertex_array = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(empty_vertex_array))

    # every program reads `shaders.FRAME` from the same binding
    frame_buffer = create_buffer(gl.GL_UNIFORM_BUFFER, np.zeros(1, dtype=shaders.FRAME))
    gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, FRAME_BINDING, frame_buffer)

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
            raise ValueError(f"`{file_name}` failed to compile:\n{shader_log(shader)}")
        gl.glAttachShader(program, shader)
        # the shader is freed when the program is
        gl.glDeleteShader(shader)

    gl.glLinkProgram(program)
    status = gl.GLint()
    gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        raise ValueError(f"`{name}` failed to link:\n{shader_log(program, program=True)}")

    # read `shaders.FRAME` from its binding unless the program never uses it
    block = gl.glGetUniformBlockIndex(program, ctypes.create_string_buffer(b"Frame"))
    if block != gl.GL_INVALID_INDEX:
        gl.glUniformBlockBinding(program, block, FRAME_BINDING)

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


def shader_log(handle: int, program: bool = False) -> str:
    """
    Get the compiler or linker message for a shader or program.

    Parameters
    ------------
    handle : int
      Shader or program.
    program : bool
      If `handle` is a program rather than a shader.

    Returns
    ------------
    log : str
      The message from the driver.
    """
    log = ctypes.create_string_buffer(4096)
    if program:
        gl.glGetProgramInfoLog(handle, 4096, None, log)
    else:
        gl.glGetShaderInfoLog(handle, 4096, None, log)
    return log.value.decode("utf-8", errors="replace")


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
    gl.glBufferData(target, data.nbytes, data.ctypes.data_as(ctypes.c_void_p), usage)


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

    if arrays.kind != "mesh":
        # each segment is one instance and the shader makes a quad around it
        segments = np.zeros(len(arrays.vertices), dtype=shaders.SEGMENT)
        segments["segment_start"] = arrays.vertices[:, 0]
        segments["segment_end"] = arrays.vertices[:, 1]
        segments["color"] = arrays.vertex_colors
        buffers = [create_buffer(gl.GL_ARRAY_BUFFER, segments)]
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
            count=len(segments),
            material=arrays.material,
            centroid=arrays.vertices.reshape((-1, 3)).mean(axis=0),
            corners=box_corners(arrays.vertices),
        )

    locations = shaders.attribute_locations(
        shaders.VERTEX, shaders.SKIN, shaders.INSTANCE
    )
    vertices = np.zeros(len(arrays.vertices), dtype=shaders.VERTEX)
    vertices["position"] = arrays.vertices
    vertices["normal"] = arrays.normals
    vertices["uv"] = arrays.uv
    vertices["color"] = arrays.vertex_colors
    buffers = [create_buffer(gl.GL_ARRAY_BUFFER, vertices)]
    attribute_pointers(shaders.VERTEX, locations, per_instance=False)
    buffers.append(create_buffer(gl.GL_ELEMENT_ARRAY_BUFFER, arrays.faces))
    # every node drawing this mesh is one instance, rewritten every frame
    buffers.append(create_buffer(gl.GL_ARRAY_BUFFER, np.zeros(1, dtype=shaders.INSTANCE)))
    attribute_pointers(shaders.INSTANCE, locations, per_instance=True)
    skinned = arrays.joints is not None
    if skinned:
        skin = np.zeros(len(arrays.vertices), dtype=shaders.SKIN)
        skin["joints"] = arrays.joints
        skin["weights"] = arrays.weights
        buffers.append(create_buffer(gl.GL_ARRAY_BUFFER, skin))
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

    material = arrays.material
    textures = {}
    for name, srgb in shaders.MATERIAL_TEXTURES.items():
        image = getattr(material, name)
        if image is not None:
            textures[name] = upload_image(np.asarray(image.convert("RGBA")), srgb=srgb)

    return GeometryBuffers(
        kind="mesh",
        vertex_array=vertex_array.value,
        buffers=buffers,
        count=arrays.faces.size,
        material=material,
        textures=textures,
        face_colors=face_colors,
        instance_buffer=buffers[2],
        material_key=(
            *material.base_color.tolist(),
            material.metallic,
            material.roughness,
            *material.emissive.tolist(),
            material.alpha_mode,
            material.alpha_cutoff,
            material.double_sided,
            material.shadow_catcher,
            material.diffuse_transmission,
            *material.diffuse_transmission_color.tolist(),
            *(textures.get(name, 0) for name in shaders.MATERIAL_TEXTURES),
        ),
        centroid=arrays.vertices.mean(axis=0),
        corners=box_corners(arrays.vertices),
        skinned=skinned,
    )


def box_corners(vertices: np.ndarray) -> np.ndarray:
    """
    The corners of the axis aligned box around some vertices.

    Parameters
    ------------
    vertices : (..., 3) float
      Points.

    Returns
    ------------
    corners : (8, 3) float
      Every corner of their bounding box.
    """
    points = np.reshape(vertices, (-1, 3))
    return corners([points.min(axis=0), points.max(axis=0)])


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


def create_shadow_maps(resolution: int) -> ShadowMaps:
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
    shadow_maps : ShadowMaps
      The texture and a framebuffer to render each layer into.
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
    return ShadowMaps(
        framebuffer=framebuffer.value, texture=texture.value, resolution=resolution
    )


def create_ambient_occlusion(width: Integer, height: Integer) -> AmbientOcclusion:
    """
    Allocate the half resolution images ambient occlusion is drawn in.

    Parameters
    ------------
    width : int
      Width of the full resolution framebuffer in pixels.
    height : int
      Height of the full resolution framebuffer in pixels.

    Returns
    ------------
    occlusion : AmbientOcclusion
      Depth, noisy and blurred images and a framebuffer.
    """
    width, height = max(int(width) // 2, 1), max(int(height) // 2, 1)
    textures = (gl.GLuint * 3)()
    gl.glGenTextures(3, textures)
    for texture, internal, pixel_format, kind, smooth in (
        (textures[0], gl.GL_DEPTH_COMPONENT24, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, False),
        (textures[1], gl.GL_R8, gl.GL_RED, gl.GL_UNSIGNED_BYTE, False),
        (textures[2], gl.GL_R8, gl.GL_RED, gl.GL_UNSIGNED_BYTE, True),
    ):
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, internal, width, height, 0, pixel_format, kind, None
        )
        # the blurred occlusion is scaled up to the full image smoothly
        filtering = gl.GL_NEAREST
        if smooth:
            filtering = gl.GL_LINEAR
        for name, value in (
            (gl.GL_TEXTURE_MIN_FILTER, filtering),
            (gl.GL_TEXTURE_MAG_FILTER, filtering),
            (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
            (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, name, value)
    framebuffer = gl.GLuint()
    gl.glGenFramebuffers(1, ctypes.byref(framebuffer))
    return AmbientOcclusion(
        framebuffer=framebuffer.value,
        depth=textures[0],
        noisy=textures[1],
        blurred=textures[2],
        width=width,
        height=height,
    )


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
    width : int
      Width in pixels.
    height : int
      Height in pixels.
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
        result.delete()
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
    gl.glReadPixels(
        0,
        0,
        framebuffer.width,
        framebuffer.height,
        pixel_format,
        kind,
        ctypes.c_void_p(0),
    )
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
