"""
shaders
-----------

The GLSL shaders beside this file, and every byte the GPU reads
declared once as a numpy structured dtype.

Python fills arrays of these dtypes by field name, and the GLSL that
reads them is generated from the same dtypes and filled into the
shader templates here, so the two can never disagree about a name,
a type, an offset or an attribute location. Nothing in this module
imports OpenGL.
"""

import os
from string import Template

import numpy as np

from ....scene.metadata import DIFFUSE_SHARPNESS, SPECULAR_SHARPNESS
from ...conversion import ALPHA_MODES

# the directory holding the `.glsl` templates
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

# the most lights a frame can be lit by
MAX_LIGHTS = 8
# the most directional and spot lights which cast shadows
MAX_SHADOWS = 4
# the most softboxes an environment can have
MAX_SOFTBOXES = 4

# the texture unit each sampler reads from for the life of a program
TEXTURE_UNITS = {
    "base_color_texture": 0,
    "metallic_roughness_texture": 1,
    "emissive_texture": 2,
    "normal_texture": 3,
    "occlusion_texture": 4,
    "face_colors": 5,
    "shadow_maps": 6,
    "joint_matrices": 7,
    "scene_depth": 8,
    "noisy_occlusion": 9,
    "ambient_occlusion": 10,
    "diffuse_transmission_texture": 11,
    "diffuse_transmission_color_texture": 12,
}

# material texture sampler : if it holds sRGB colors rather than data
MATERIAL_TEXTURES = {
    "base_color_texture": True,
    "metallic_roughness_texture": False,
    "emissive_texture": True,
    "normal_texture": False,
    "occlusion_texture": False,
    "diffuse_transmission_texture": False,
    "diffuse_transmission_color_texture": True,
}

# one row per mesh vertex
VERTEX = np.dtype(
    [
        ("position", "<f4", 3),
        ("normal", "<f4", 3),
        ("uv", "<f4", 2),
        # uint8 RGBA which the shader reads as 0.0-1.0
        ("color", "u1", 4),
    ]
)

# one row per vertex of a skinned mesh, in a buffer of its own so meshes
# without a skin don't carry it
SKIN = np.dtype(
    [
        # which four joints move the vertex, as floats which are exact
        ("joints", "<f4", 4),
        # how much each of them moves it
        ("weights", "<f4", 4),
    ]
)

# one row per scene graph node drawing a mesh, which is one instance
INSTANCE = np.dtype(
    [
        # mesh to camera, stored transposed: OpenGL reads a matrix
        # input one column at a time, which is one row of the transpose
        ("model_view", "<f4", (4, 4)),
        # the cofactor of `model_view`, also stored transposed
        ("normal_matrix", "<f4", (3, 3)),
        # which of `scene.graph.nodes_geometry` this is, for segmentation
        ("node_index", "<f4"),
        # the first row of this node's joints in `joint_matrices`, or
        # -1 if its mesh isn't skinned
        ("joint_offset", "<f4"),
    ]
)

# one row per line segment or point, which is one instance
SEGMENT = np.dtype(
    [
        ("segment_start", "<f4", 3),
        ("segment_end", "<f4", 3),
        ("color", "u1", 4),
    ]
)

# one `KHR_lights_punctual` light, where every kind of light is the
# same math: directional lights are infinitely far away, and point
# lights are spot lights with a cone that covers everything
LIGHT = np.dtype(
    [
        # view space position, or for a directional light a unit
        # vector towards it with a zero W so its distance is one
        ("position", "<f4", 4),
        # view space unit vector the light shines along, its node's -Z
        ("direction", "<f4", 4),
        # linear RGB times intensity in lux or candela
        ("color", "<f4", 4),
        # one over the range or zero, the spot cone scale and offset
        # which are zero and one without a cone, then zero
        ("falloff", "<f4", 4),
        # the layer of `shadow_maps` with the light's shadow or -1 for
        # none, then the width of a shadow map pixel in view units as a
        # constant plus a multiple of the distance to the light, then zero
        ("shadow", "<f4", 4),
    ],
    metadata={"glsl": "Light"},
)

# a large soft light infinitely far away, from `metadata.Softbox`
SOFTBOX = np.dtype(
    [
        # view space unit vector towards it, then its sharpness
        ("direction", "<f4", 4),
        # linear RGB radiance at its center, then zero
        ("color", "<f4", 4),
    ],
    metadata={"glsl": "Softbox"},
)

# set once per frame and shared by every program as `frame.name`: a
# uniform block only holds 16 byte vectors and matrices so numpy's
# packing is exactly the std140 layout OpenGL reads
FRAME = np.dtype(
    [
        # camera to clip space and back
        ("projection", "<f4", (4, 4)),
        ("inverse_projection", "<f4", (4, 4)),
        # framebuffer width and height in pixels, then zeros
        ("viewport_size", "<f4", 4),
        # one to draw view depth and node index instead of color, then zeros
        ("write_depth_and_index", "<i4", 4),
        # how many of `lights` are in use, then zeros
        ("light_count", "<i4", 4),
        ("lights", LIGHT, MAX_LIGHTS),
        # view space to the 0.0-1.0 texture coordinates and depth of
        # each layer of `shadow_maps`
        ("shadow_matrices", "<f4", (MAX_SHADOWS, 4, 4)),
        # the environment from `metadata.Environment`: linear RGB from
        # straight up and down, the view space up, and the softboxes
        ("sky_color", "<f4", 4),
        ("ground_color", "<f4", 4),
        ("up_direction", "<f4", 4),
        ("softbox_count", "<i4", 4),
        ("softboxes", SOFTBOX, MAX_SOFTBOXES),
        # how bright a white floor facing up looks with nothing in its
        # way, which shadows on the floor are darker than, then zeros
        ("ground_light", "<f4", 4),
        # how far away surfaces hide the environment, then zeros
        ("occlusion_radius", "<f4", 4),
    ]
)

# numpy scalar kind : (GLSL scalar, GLSL vector prefix)
GLSL_SCALARS = {"f": ("float", "vec"), "i": ("int", "ivec"), "u": ("uint", "uvec")}

# filled shader sources keyed by file name
_SHADER_SOURCE = {}


def glsl_type(field: np.dtype) -> tuple[str, tuple]:
    """
    The GLSL type of one dtype field, where the last one or two
    dimensions of the shape are the vector or matrix and anything
    before them is an array: `(4, 4)` is a `mat4` and `(2, 4, 4)`
    is `mat4[2]`.

    Parameters
    ------------
    field : numpy.dtype
      Type of the field, i.e. `dtype.fields[name][0]`.

    Returns
    ------------
    name : str
      GLSL type, i.e. `vec3` or `mat4`.
    array : tuple
      Leading dimensions which are a GLSL array, or empty.
    """
    base, shape = field.base, field.shape
    if base.names is not None:
        # a nested structured dtype is a GLSL struct
        return base.metadata["glsl"], shape
    scalar, vector = GLSL_SCALARS[base.kind]
    if base.itemsize == 1:
        # bytes are normalized to 0.0-1.0 floats as they're read
        scalar, vector = GLSL_SCALARS["f"]
    if len(shape) >= 2 and shape[-1] == shape[-2] and base.kind == "f":
        return f"mat{shape[-1]}", shape[:-2]
    if len(shape) >= 1:
        return f"{vector}{shape[-1]}", shape[:-1]
    return scalar, ()


def attribute_locations(*dtypes: np.dtype) -> dict:
    """
    Number the vertex shader inputs of one or more dtypes in order,
    where a matrix takes one location per column.

    Parameters
    ------------
    dtypes : numpy.dtype
      Per-vertex then per-instance dtypes of one shader.

    Returns
    ------------
    locations : dict
      {field name : first location}
    """
    locations = {}
    location = 0
    for dtype in dtypes:
        for name in dtype.names:
            locations[name] = location
            # a matrix takes a location per column and anything else one
            location += int(np.prod(dtype.fields[name][0].shape[:-1]))
    return locations


def shader_source(name: str) -> str:
    """
    Fill a shader template from this directory with the declarations
    and constants in this module.

    Parameters
    ------------
    name : str
      Program and stage, i.e. `mesh.vert` for `mesh.vert.glsl`.

    Returns
    ------------
    source : str
      GLSL ready to compile.
    """
    try:
        return _SHADER_SOURCE[name]
    except KeyError:
        pass
    with open(os.path.join(DIRECTORY, f"{name}.glsl")) as f:
        template = Template(f.read())
    with open(os.path.join(DIRECTORY, "shared.glsl")) as f:
        shared = f.read()

    # the vertex inputs of each program, numbered in order
    inputs = {}
    for key, dtypes in {
        "mesh_inputs": (VERTEX, SKIN, INSTANCE),
        "segment_inputs": (SEGMENT,),
    }.items():
        locations = attribute_locations(*dtypes)
        lines = []
        for dtype in dtypes:
            for field in dtype.names:
                kind, _ = glsl_type(dtype.fields[field][0])
                lines.append(f"layout(location = {locations[field]}) in {kind} {field};")
        inputs[key] = "\n".join(lines)

    # the frame's std140 block after the structs its fields use, with
    # matrices `row_major` so numpy arrays upload untransposed: every field
    # has to be whole 16 byte vectors or matrices, which is the only case
    # where numpy and std140 pack a dtype identically
    structs = {
        field.base.metadata["glsl"]: field.base
        for field, *_ in FRAME.fields.values()
        if field.base.names is not None
    }
    parts = [(f"struct {kind}", fields, "};") for kind, fields in structs.items()]
    parts.append(("layout(std140, row_major) uniform Frame", FRAME, "} frame;"))
    block = []
    for header, fields, closing in parts:
        block.append(f"{header} {{")
        for field_name in fields.names:
            field, offset = fields.fields[field_name][:2]
            kind, array = glsl_type(field)
            element = field.base.itemsize * int(np.prod(field.shape[len(array) :]))
            if element % 16 != 0 or offset % 16 != 0:
                raise ValueError(
                    f"`{field_name}` must be made of 16 byte vectors for std140"
                )
            dimensions = "".join(f"[{size}]" for size in array)
            block.append(f"    {kind} {field_name}{dimensions};")
        block.append(closing)

    source = template.substitute(
        frame_block="\n".join(block),
        shared_functions=shared,
        **inputs,
        alpha_modes="\n".join(
            f"const int ALPHA_{name} = {value};" for name, value in ALPHA_MODES.items()
        ),
        DIFFUSE_SHARPNESS=repr(float(DIFFUSE_SHARPNESS)),
        SPECULAR_SHARPNESS="vec4({})".format(
            ", ".join(repr(float(value)) for value in SPECULAR_SHARPNESS)
        ),
    )
    _SHADER_SOURCE[name] = source
    return source
