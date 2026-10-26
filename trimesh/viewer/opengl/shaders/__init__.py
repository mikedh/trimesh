"""
shaders
-----------

The GLSL shaders beside this file, and every byte the GPU reads
declared once as a numpy structured dtype: Python fills arrays of them
by field name, and the GLSL reading them is generated from them, so the
two never disagree about a name, a type, an offset or an attribute
location. Nothing in this module imports OpenGL.
"""

import os

import numpy as np

# Rec.709 weights of linear RGB for how bright a color looks
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
# the directory holding the `.glsl` files
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

# How a surface's alpha is used, which is glTF's `alphaMode`, and the
# integer the mesh shader compares against as `ALPHA_{name}`: OPAQUE
# ignores alpha, MASK draws only where alpha is at least `alpha_cutoff`,
# and BLEND draws over whatever is behind, sorted far to near. Clear
# glass from `KHR_materials_transmission` is BLEND with an alpha of one
# minus how much light passes through: every shader writes color already
# multiplied by alpha and the framebuffer adds what is behind times one
# minus alpha, so reflections, which aren't, still show on clear glass.
ALPHA_MODES = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}

# what a draw writes as `frame.output_mode`, compared against `OUTPUT_{name}`:
# shaded color, the view depth and node index of every pixel, or only depth
OUTPUT_MODES = {"color": 0, "index": 1, "depth": 2}

# the most lights a frame can be lit by
MAX_LIGHTS = 8
# the most directional and spot lights which cast shadows
MAX_SHADOWS = 4

# every program as `{name}.vert.glsl` and `{name}.frag.glsl`
PROGRAMS = ("mesh", "line", "fullscreen")

# material texture sampler : if it holds sRGB colors rather than data
MATERIAL_TEXTURES = {
    "base_color_texture": True,
    "metallic_roughness_texture": False,
    "emissive_texture": True,
    "normal_texture": False,
    "occlusion_texture": False,
    "transmission_texture": False,
    "diffuse_transmission_texture": False,
    "diffuse_transmission_color_texture": True,
}

# The passes of `fullscreen.frag`, which it tells apart by comparing
# `pass_index` against `PASS_{name}`, in the order they are drawn:
# name : ({sampler : image it reads}, image it writes, adds onto it)
# where the images are the viewer's `TARGETS`
PASSES = {
    # the environment behind the scene
    "background": ({}, "multisample", False),
    # how much of the sky nearby surfaces hide, with noise and then without
    "occlusion": ({"scene_depth": "scene_depth"}, "noisy", False),
    "blur": (
        {"scene_depth": "scene_depth", "noisy_occlusion": "noisy"},
        "occlusion",
        False,
    ),
    # what the tone mapper compressed glows: blurred across a mip level
    # of the frame, then down, then added onto the frame
    "bloom_across": ({"bloom_source": "resolved"}, "across", False),
    "bloom_down": ({"bloom_source": "across"}, "glow", False),
    "bloom_add": ({"bloom_source": "glow"}, "resolved", True),
}

# the texture unit each sampler reads from for the life of a program:
# the material's, then the frame's, then those `PASSES` read
SAMPLERS = (
    *MATERIAL_TEXTURES, "face_colors", "shadow_maps", "joint_matrices",
    "ambient_occlusion", "environment",
    *(sampler for inputs, _, _ in PASSES.values() for sampler in inputs),
)  # fmt: skip
TEXTURE_UNITS = {name: unit for unit, name in enumerate(dict.fromkeys(SAMPLERS))}

# one row per mesh vertex, with uint8 RGBA the shader reads as 0.0-1.0
VERTEX = np.dtype(
    [("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2), ("color", "u1", 4)]
)

# one row per vertex of a skinned mesh in a buffer of its own: the four
# joints moving it as floats which are exact, and how much each moves it
SKIN = np.dtype([("joints", "<f4", 4), ("weights", "<f4", 4)])

# one row per scene graph node drawing a mesh, which is one instance
INSTANCE = np.dtype(
    [
        # mesh to camera and its cofactor, stored transposed as OpenGL
        # reads a matrix input one column at a time
        ("model_view", "<f4", (4, 4)),
        ("normal_matrix", "<f4", (3, 3)),
        # which of `scene.graph.nodes_geometry` this is, for segmentation
        ("node_index", "<f4"),
        # the first row of its joints in `joint_matrices`, or -1 unskinned
        ("joint_offset", "<f4"),
    ]
)

# one row per line segment or point, which is one instance
SEGMENT = np.dtype(
    [("segment_start", "<f4", 3), ("segment_end", "<f4", 3), ("color", "u1", 4)]
)

# the vertex shader inputs of each program, numbered in this order
INPUTS = {"mesh": (VERTEX, SKIN, INSTANCE), "line": (SEGMENT,)}

# one `KHR_lights_punctual` light, where every kind is the same math:
# directional lights are infinitely far away, and point lights are spot
# lights with a cone covering everything
LIGHT = np.dtype(
    [
        # view space position, or a unit vector towards a directional light
        # with a zero W so its distance is one
        ("position", "<f4", 4),
        # view space unit vector the light shines along, its node's -Z
        ("direction", "<f4", 4),
        # linear RGB times intensity in lux or candela
        ("color", "<f4", 4),
        # one over the range or zero, the spot cone scale and offset which are
        # zero and one without a cone, then the outer cone angle for the CPU
        ("falloff", "<f4", 4),
        # the layer of `shadow_maps` with its shadow or -1, then the width of
        # a shadow map pixel in view units as a constant plus a multiple of
        # the distance to the light, then zero
        ("shadow", "<f4", 4),
    ],
    metadata={"glsl": "Light"},
)

# set once per frame and shared by every program as `frame.name`, of only
# 16 byte vectors and matrices so numpy packs it like std140
FRAME = np.dtype(
    [
        # camera to clip space and back
        ("projection", "<f4", (4, 4)),
        ("inverse_projection", "<f4", (4, 4)),
        # framebuffer width and height in pixels, then zeros
        ("viewport_size", "<f4", 4),
        # the value in `OUTPUT_MODES`, then zeros
        ("output_mode", "<i4", 4),
        # how many of `lights` are in use, then zeros
        ("light_count", "<i4", 4),
        ("lights", LIGHT, MAX_LIGHTS),
        # view space to the 0.0-1.0 texture coordinates and depth of
        # each layer of `shadow_maps`
        ("shadow_matrices", "<f4", (MAX_SHADOWS, 4, 4)),
        # the `EnvironmentLight`: view space to its cube map's frame, the
        # linear RGB the cube map is multiplied by then the mip level of
        # roughness one, and its irradiance over pi as the coefficients of
        # `lighting.spherical_harmonics` times their constants
        ("environment_rotation", "<f4", (4, 4)),
        ("environment_color", "<f4", 4),
        ("irradiance", "<f4", (9, 4)),
        # how bright a white floor facing up looks with nothing in its
        # way, which shadows on the floor are darker than, then zeros
        ("ground_light", "<f4", 4),
        # how far away surfaces hide the environment, then zeros
        ("occlusion_radius", "<f4", 4),
        # what light is multiplied by before tone mapping, then zeros
        ("exposure", "<f4", 4),
        # how much of what tone mapping compressed glows onto its neighbors,
        # its standard deviation in texels of the mip level it is blurred
        # at, that level, then zero
        ("bloom", "<f4", 4),
        # pixels across lines, then across points, then zeros
        ("line_widths", "<f4", 4),
        # one to shade with vertex normals rather than flat, then zeros
        ("smooth_shading", "<i4", 4),
    ]
)

# one glTF metallic-roughness material, which every mesh has a uniform
# buffer of, read by the mesh shader as `material.name`
MATERIAL = np.dtype(
    [
        # linear RGBA multiplied into every pixel
        ("base_color", "<f4", 4),
        # linear RGB light given off, then zero
        ("emissive", "<f4", 4),
        # metallic, roughness, the alpha below which `MASK` draws nothing,
        # and the fraction of diffuse light a thin surface scatters out of
        # its other side from `KHR_materials_diffuse_transmission`
        ("factors", "<f4", 4),
        # linear RGB of that scattered light, then the fraction of light
        # passing through like clear glass from `KHR_materials_transmission`
        ("transmission_color", "<f4", 4),
        # the value in `ALPHA_MODES`, then one for `KHR_materials_unlit`, for
        # a floor only showing shadows, and for both sides which the CPU reads
        ("flags", "<i4", 4),
        # one where a normal map and a color per triangle are bound, then zeros
        ("has_textures", "<i4", 4),
    ]
)

# every uniform block by its GLSL name, read as the lower case name, from
# the uniform buffer binding of its index here
BLOCKS = {"Frame": FRAME, "Material": MATERIAL}

# numpy scalar kind : (GLSL scalar, GLSL vector prefix)
GLSL_SCALARS = {"f": ("float", "vec"), "i": ("int", "ivec"), "u": ("uint", "uvec")}


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
      Per-vertex then per-instance dtypes of one program.

    Returns
    ------------
    locations : dict
      {field name : first location}
    """
    fields = [(name, dtype.fields[name][0]) for dtype in dtypes for name in dtype.names]
    columns = [int(np.prod(item.shape[:-1])) for _, item in fields]
    return dict(zip([name for name, _ in fields], np.cumsum([0, *columns]).tolist()))


def shader_source(name: str) -> str:
    """
    A shader from this directory after everything generated from this
    module: the constants, the vertex inputs, the uniform blocks and
    `shared.glsl`.

    Parameters
    ------------
    name : str
      Program and stage, i.e. `mesh.vert` for `mesh.vert.glsl`.

    Returns
    ------------
    source : str
      GLSL ready to compile.
    """
    # every enumeration as integer constants, i.e. `ALPHA_BLEND`
    enumerations = {"ALPHA": ALPHA_MODES, "OUTPUT": OUTPUT_MODES}
    enumerations["PASS"] = dict(zip(PASSES, range(len(PASSES))))
    luminance = ", ".join(str(value) for value in LUMINANCE.tolist())
    lines = ["#version 330 core", f"const vec3 LUMINANCE = vec3({luminance});"]
    lines.extend(
        f"const int {prefix}_{key.upper()} = {value};"
        for prefix, values in enumerations.items()
        for key, value in values.items()
    )
    # the inputs of a vertex shader
    program, stage = name.split(".")
    dtypes = ()
    if stage == "vert":
        dtypes = INPUTS.get(program, ())
    locations = attribute_locations(*dtypes)
    for dtype in dtypes:
        for field in dtype.names:
            kind = glsl_type(dtype.fields[field][0])[0]
            lines.append(f"layout(location = {locations[field]}) in {kind} {field};")
    # each std140 block after the structs its fields use, with matrices
    # `row_major` so numpy arrays upload untransposed: every field has to
    # be whole 16 byte vectors or matrices, which is the only case where
    # numpy and std140 pack a dtype identically
    for block_name, dtype in BLOCKS.items():
        parts = {
            f"struct {item.base.metadata['glsl']}": (item.base, "};")
            for item, *_ in dtype.fields.values()
            if item.base.names is not None
        }
        header = f"layout(std140, row_major) uniform {block_name}"
        parts[header] = (dtype, f"}} {block_name.lower()};")
        for header, (fields, closing) in parts.items():
            lines.append(f"{header} {{")
            for field_name in fields.names:
                field, offset = fields.fields[field_name][:2]
                kind, array = glsl_type(field)
                if (field.itemsize // int(np.prod(array))) % 16 or offset % 16:
                    raise ValueError(
                        f"`{field_name}` must be made of 16 byte vectors for std140"
                    )
                dimensions = "".join(f"[{size}]" for size in array)
                lines.append(f"    {kind} {field_name}{dimensions};")
            lines.append(closing)
    for file_name in ("shared", name):
        with open(os.path.join(DIRECTORY, f"{file_name}.glsl")) as f:
            lines.append(f.read())
    return "\n".join(lines)
