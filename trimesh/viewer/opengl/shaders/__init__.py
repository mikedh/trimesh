"""
shaders
-----------

GLSL shaders and the numpy dtypes of everything the GPU reads, which
the GLSL declarations are generated from.
"""

import os

import numpy as np

# Rec.709 luminance weights of linear RGB
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

# glTF `alphaMode` as `ALPHA_{name}`: transmission is BLEND with alpha of
# `1 - transmission`, and color is premultiplied so glass keeps reflections
ALPHA_MODES = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}

# `frame.output_mode` as `OUTPUT_{name}`: color, depth and node index, or depth
OUTPUT_MODES = {"color": 0, "index": 1, "depth": 2}

# most lights per frame
MAX_LIGHTS = 8
# most directional and spot lights casting shadows
MAX_SHADOWS = 4

# programs as `{name}.vert.glsl` and `{name}.frag.glsl`
PROGRAMS = ("mesh", "line", "fullscreen")

# material sampler : (`PBRMaterial` attribute, if it holds sRGB color)
MATERIAL_TEXTURES = {
    "base_color_texture": ("baseColorTexture", True),
    "metallic_roughness_texture": ("metallicRoughnessTexture", False),
    "emissive_texture": ("emissiveTexture", True),
    "normal_texture": ("normalTexture", False),
    "occlusion_texture": ("occlusionTexture", False),
    "transmission_texture": ("transmissionTexture", False),
    "diffuse_transmission_texture": ("diffuseTransmissionTexture", False),
    "diffuse_transmission_color_texture": ("diffuseTransmissionColorTexture", True),
}

# `fullscreen.frag` passes as `PASS_{name}` in draw order, with `TARGETS` as
# name : ({sampler : input}, output, adds onto output)
PASSES = {
    "background": ({}, "multisample", False),
    # ambient occlusion, noisy then blurred
    "occlusion": ({"scene_depth": "scene_depth"}, "noisy", False),
    "blur": (
        {"scene_depth": "scene_depth", "noisy_occlusion": "noisy"},
        "occlusion",
        False,
    ),
    # bloom blurred across squares of pixels, then down, then added to the frame
    "bloom_across": ({"bloom_source": "resolved"}, "across", False),
    "bloom_down": ({"bloom_source": "across"}, "glow", False),
    "bloom_add": ({"bloom_source": "glow"}, "resolved", True),
}

# fixed texture unit of each sampler
SAMPLERS = (
    *MATERIAL_TEXTURES, "face_colors", "shadow_maps", "joint_matrices",
    "ambient_occlusion", "environment",
    *(sampler for inputs, _, _ in PASSES.values() for sampler in inputs),
)  # fmt: skip
TEXTURE_UNITS = {name: unit for unit, name in enumerate(dict.fromkeys(SAMPLERS))}

# one row per mesh vertex, with color read as 0.0-1.0
VERTEX = np.dtype(
    [("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2), ("color", "u1", 4)]
)

# one row per skinned vertex in its own buffer, with joint indices as floats
SKIN = np.dtype([("joints", "<f4", 4), ("weights", "<f4", 4)])

# one row per scene graph node drawing a mesh
INSTANCE = np.dtype(
    [
        # transposed as OpenGL reads matrix inputs by column
        ("model_view", "<f4", (4, 4)),
        ("normal_matrix", "<f4", (3, 3)),
        # index into `scene.graph.nodes_geometry`
        ("node_index", "<f4"),
        # first row in `joint_matrices`, or -1 if unskinned
        ("joint_offset", "<f4"),
    ]
)

# one row per line segment or point
SEGMENT = np.dtype(
    [("segment_start", "<f4", 3), ("segment_end", "<f4", 3), ("color", "u1", 4)]
)

# vertex shader inputs of each program, numbered in order
INPUTS = {"mesh": (VERTEX, SKIN, INSTANCE), "line": (SEGMENT,)}

# one `KHR_lights_punctual` light, where a point light is a spot without a cone
LIGHT = np.dtype(
    [
        # view space position, or unit vector to a directional light with W zero
        ("position", "<f4", 4),
        # view space unit vector it shines along
        ("direction", "<f4", 4),
        # linear RGB times intensity in lux or candela
        ("color", "<f4", 4),
        # 1 / range or 0, spot cone scale and offset, outer cone angle
        ("falloff", "<f4", 4),
        # layer of `shadow_maps` or -1
        ("shadow", "<f4", 4),
    ],
    metadata={"glsl": "Light"},
)

# per frame uniforms as `frame.name`: only 16 byte fields so it packs as std140
FRAME = np.dtype(
    [
        ("projection", "<f4", (4, 4)),
        ("inverse_projection", "<f4", (4, 4)),
        # framebuffer size in pixels
        ("viewport_size", "<f4", 4),
        # value of `OUTPUT_MODES`
        ("output_mode", "<i4", 4),
        ("light_count", "<i4", 4),
        ("lights", LIGHT, MAX_LIGHTS),
        # view to clip space of each layer of `shadow_maps`
        ("shadow_matrices", "<f4", (MAX_SHADOWS, 4, 4)),
        # `EnvironmentLight`: view to cube map rotation, RGB scale then mip level
        # of roughness one, and irradiance over pi as spherical harmonics
        ("environment_rotation", "<f4", (4, 4)),
        ("environment_color", "<f4", 4),
        ("irradiance", "<f4", (9, 4)),
        # luminance of an unshadowed white floor, for the shadow catcher
        ("ground_light", "<f4", 4),
        ("occlusion_radius", "<f4", 4),
        # multiplier before tone mapping
        ("exposure", "<f4", 4),
        # strength, standard deviation in texels, log2 of the square's side
        ("bloom", "<f4", 4),
        # line then point widths in pixels
        ("line_widths", "<f4", 4),
        ("smooth_shading", "<i4", 4),
    ]
)

# one glTF metallic-roughness material as `material.name`
MATERIAL = np.dtype(
    [
        # linear RGBA
        ("base_color", "<f4", 4),
        # linear RGB
        ("emissive", "<f4", 4),
        # metallic, roughness, alpha cutoff, diffuse transmission
        ("factors", "<f4", 4),
        # diffuse transmission RGB, then `KHR_materials_transmission`
        ("transmission_color", "<f4", 4),
        # `ALPHA_MODES`, unlit, shadow catcher, double sided
        ("flags", "<i4", 4),
        # normal map, face colors
        ("has_textures", "<i4", 4),
    ]
)

# uniform blocks by GLSL name, bound to their index here
BLOCKS = {"Frame": FRAME, "Material": MATERIAL}

# numpy scalar kind : (GLSL scalar, GLSL vector prefix)
GLSL_SCALARS = {"f": ("float", "vec"), "i": ("int", "ivec"), "u": ("uint", "uvec")}


def glsl_type(field: np.dtype) -> tuple[str, tuple]:
    """
    GLSL type of a dtype field, with leading dimensions as an array.

    Parameters
    ------------
    field : numpy.dtype
      Field type, i.e. `dtype.fields[name][0]`.

    Returns
    ------------
    name : str
      GLSL type, i.e. `vec3` or `mat4`.
    array : tuple
      Array dimensions, or empty.
    """
    base, shape = field.base, field.shape
    if base.names is not None:
        # a nested structured dtype is a GLSL struct
        return base.metadata["glsl"], shape
    scalar, vector = GLSL_SCALARS[base.kind]
    if base.itemsize == 1:
        # bytes are read normalized to floats
        scalar, vector = GLSL_SCALARS["f"]
    if len(shape) >= 2 and shape[-1] == shape[-2] and base.kind == "f":
        return f"mat{shape[-1]}", shape[:-2]
    if len(shape) >= 1:
        return f"{vector}{shape[-1]}", shape[:-1]
    return scalar, ()


def attribute_locations(*dtypes: np.dtype) -> dict:
    """
    Vertex shader input locations, one per matrix column.

    Parameters
    ------------
    dtypes : numpy.dtype
      Per-vertex then per-instance dtypes.

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
    Shader source after generated constants, inputs and blocks.

    Parameters
    ------------
    name : str
      Program and stage, i.e. `mesh.vert` for `mesh.vert.glsl`.

    Returns
    ------------
    source : str
      GLSL ready to compile.
    """
    # enumerations as int constants, i.e. `ALPHA_BLEND`
    enumerations = {"ALPHA": ALPHA_MODES, "OUTPUT": OUTPUT_MODES}
    enumerations["PASS"] = dict(zip(PASSES, range(len(PASSES))))
    luminance = ", ".join(str(value) for value in LUMINANCE.tolist())
    lines = ["#version 330 core", f"const vec3 LUMINANCE = vec3({luminance});"]
    lines.extend(
        f"const int {prefix}_{key.upper()} = {value};"
        for prefix, values in enumerations.items()
        for key, value in values.items()
    )
    program, stage = name.split(".")
    dtypes = ()
    if stage == "vert":
        dtypes = INPUTS.get(program, ())
    locations = attribute_locations(*dtypes)
    for dtype in dtypes:
        for field in dtype.names:
            kind = glsl_type(dtype.fields[field][0])[0]
            lines.append(f"layout(location = {locations[field]}) in {kind} {field};")
    # `row_major` so numpy uploads untransposed, and numpy only packs like
    # std140 when every field is whole 16 byte vectors or matrices
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
                kind, array = glsl_type(fields.fields[field_name][0])
                dimensions = "".join(f"[{size}]" for size in array)
                lines.append(f"    {kind} {field_name}{dimensions};")
            lines.append(closing)
    for file_name in ("shared", name):
        with open(os.path.join(DIRECTORY, f"{file_name}.glsl")) as source_file:
            lines.append(source_file.read())
    return "\n".join(lines)
