try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.lighting import (
    DirectionalLight,
    EnvironmentLight,
    PointLight,
    SpotLight,
    cubemap_sample,
    spherical_harmonics,
    studio_cubemap,
)
from trimesh.viewer.conversion import (
    OCCLUSION_RADIUS,
    environment_levels,
    render_lights,
)

tf = g.trimesh.transformations
# a straight line segment along X through the origin
SEGMENT = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
# an environment with no light at all
DARK = EnvironmentLight(name="environment")
# lights in these tests are white, rather than the default dim gray
WHITE = [255, 255, 255, 255]
# a sun at an angle, so a box's shadow falls beside it
SUN_POSE = tf.euler_matrix(0.5, -0.4, 0.0)


def test_all_exists():
    # every name in `__all__` should be an attribute: a stale
    # entry breaks `from trimesh.viewer import *`
    from trimesh import viewer

    # and the deprecated `trimesh.rendering` warns once when used, but not
    # for the names tools like `inspect` probe every module for
    script = [
        "import warnings, trimesh.rendering as rendering",
        "with warnings.catch_warnings(record=True) as caught:",
        "    warnings.simplefilter('always')",
        "    probed = hasattr(rendering, '__wrapped__'), len(caught)",
        "    rendering.mesh_to_vertexlist",
        "print(*probed, len(caught))",
    ]
    result = g.subprocess.run(
        [g.sys.executable, "-c", "\n".join(script)], capture_output=True, text=True
    )
    checks = {
        "every name exists": all(hasattr(viewer, name) for name in viewer.__all__),
        "rendering warns once": result.stdout.split() == ["False", "0", "1"],
    }
    for name, ok in checks.items():
        assert ok, name


def colored_box(color, extents=None):
    box = g.trimesh.creation.box(extents=extents)
    box.visual.face_colors = color
    return box


def mean_color(image):
    """The mean RGB of non-background pixels as a unit vector."""
    mean = image[(image[..., :3] < 250).any(axis=-1)][:, :3].mean(axis=0)
    return mean / g.np.linalg.norm(mean)


def ambient(color):
    """A constant environment, which is an ambient light."""
    cubemap = g.np.tile(color, (6, 1, 1, 1))
    return EnvironmentLight(name="environment", cubemap=cubemap)


def studio(sky, ground, softboxes=()):
    """An environment of a gradient from `ground` to `sky` and softboxes."""
    cubemap = studio_cubemap(sky, ground, softboxes)
    return EnvironmentLight(name="environment", cubemap=cubemap)


def clear_glass(thickness, height, roughness):
    """A 3 by 3 sheet of clear glass `height` above the origin."""
    glass = g.trimesh.creation.box(extents=[3.0, 3.0, thickness])
    glass.apply_translation([0.0, 0.0, height])
    glass.visual = g.trimesh.visual.TextureVisuals(
        material=g.trimesh.visual.material.PBRMaterial(
            transmissionFactor=1.0, roughnessFactor=roughness, metallicFactor=0.0
        )
    )
    return glass


def round_trip_checks(name, scene):
    """
    A scene which shows something renders the same after a glTF round
    trip with its viewer-only environment.
    """
    exported = g.trimesh.util.wrap_as_stream(scene.export(file_type="glb"))
    reloaded = g.trimesh.load_scene(exported, file_type="glb")
    if scene.has_lights:
        # glTF has no `EnvironmentLight`, which only the viewer draws,
        # and can't say a scene has no lights at all
        standard = []
        if reloaded.has_lights:
            standard = reloaded.lights
        reloaded.lights = standard + [
            light for light in scene.lights if isinstance(light, EnvironmentLight)
        ]
    before, after = (
        g.png(current.render(resolution=scene.camera.resolution)).astype(int)
        for current in (scene, reloaded)
    )
    return {
        f"{name} renders the same after a glTF round trip": (
            g.np.abs(before - after).max() <= 1
        ),
        f"{name} is not empty": (before[..., :3] < 250).any(axis=-1).mean() > 0.2,
    }


def test_conversion():
    # converting geometry to arrays needs no OpenGL so runs everywhere
    from trimesh.viewer.conversion import geometry_to_arrays
    from trimesh.viewer.opengl.shaders import ALPHA_MODES, MATERIAL, SEGMENT, SKIN, VERTEX

    box = colored_box([255, 0, 0, 255])
    sphere = g.trimesh.creation.icosphere()
    sphere.visual.vertex_colors = [0, 255, 0, 128]
    fuze = g.get_mesh("fuze.obj")
    blend = g.get_mesh("AlphaBlendModeTest.glb")
    wrench = g.get_mesh("2D/wrench.dxf")
    wrench.colors = g.np.tile([10, 20, 30, 255], (len(wrench.entities), 1))
    points = g.trimesh.PointCloud(g.random((10, 3)))
    # a PBR material with per-vertex colors, i.e. glTF `COLOR_0`
    painted = g.trimesh.creation.box()
    painted.visual = g.trimesh.visual.TextureVisuals(
        material=g.trimesh.visual.material.PBRMaterial(metallicFactor=0.5)
    )
    painted.visual.vertex_attributes["color"] = g.np.tile(
        [200, 150, 40, 255], (len(painted.vertices), 1)
    ).astype(g.np.uint8)
    # every material value the shader reads, which lands in the record exactly
    glass = g.trimesh.creation.box()
    glass.visual = g.trimesh.visual.TextureVisuals(
        material=g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[255, 0, 0, 255],
            emissiveFactor=[0.25, 0.5, 0.75],
            metallicFactor=0.25,
            roughnessFactor=0.75,
            alphaMode="MASK",
            alphaCutoff=0.375,
            doubleSided=True,
            transmissionFactor=0.5,
            diffuseTransmissionFactor=0.125,
            diffuseTransmissionColorFactor=[0.5, 0.25, 1.0],
        )
    )
    expected = g.np.zeros((), dtype=MATERIAL)
    expected["base_color"] = [1.0, 0.0, 0.0, 0.5]
    expected["emissive"] = [0.25, 0.5, 0.75, 0.0]
    expected["factors"] = [0.25, 0.75, 0.375, 0.125]
    expected["transmission_color"] = [0.5, 0.25, 1.0, 0.0]
    expected["flags"] = [ALPHA_MODES["BLEND"], 0, 0, 1]
    # a skinned mesh keeps which joints move each corner
    skinned = g.trimesh.creation.box()
    joints = g.np.tile(g.np.arange(4.0), (len(skinned.vertices), 1))
    weights = g.random((len(skinned.vertices), 4))
    skinned.vertex_attributes.update({"joints": joints, "weights": weights})
    voxels = g.trimesh.voxel.VoxelGrid(g.np.ones((2, 2, 2), dtype=bool))
    box_arrays, sphere_arrays, fuze_arrays = (
        geometry_to_arrays(geometry) for geometry in [box, sphere, fuze]
    )
    wrench_arrays, point_arrays, painted_arrays = (
        geometry_to_arrays(geometry) for geometry in [wrench, points, painted]
    )
    glass_arrays, skinned_arrays = (
        geometry_to_arrays(geometry) for geometry in [glass, skinned]
    )
    blend_modes = {
        int(geometry_to_arrays(geometry).material["flags"][0])
        for geometry in blend.geometry.values()
    }
    # the expected number of segments from discretizing every entity
    segment_count = sum(
        len(entity.discrete(wrench.vertices, scale=wrench.scale)) - 1
        for entity in wrench.entities
    )
    # converted once and cached on the mesh until its colors change
    cached = geometry_to_arrays(box) is box_arrays
    box.visual.face_colors = [0, 0, 255, 255]
    recolored = geometry_to_arrays(box)

    rows = box_arrays.rows
    checks = {
        "mesh rows": all(
            item.rows.dtype == VERTEX
            for item in [box_arrays, sphere_arrays, fuze_arrays, painted_arrays]
        ),
        "segment rows": all(
            item.rows.dtype == SEGMENT for item in [wrench_arrays, point_arrays]
        ),
        "box is indexed triangles": box_arrays.faces.shape == (12, 3),
        "box face colors in order": (box_arrays.face_colors == [255, 0, 0, 255]).all(),
        "box has face colors": box_arrays.material["has_textures"][1] == 1,
        "sphere has no face colors": sphere_arrays.material["has_textures"][1] == 0,
        "box triangles unchanged": g.np.allclose(
            rows["position"][box_arrays.faces], box.vertices[box.faces]
        ),
        "box normals are its face normals": g.np.allclose(
            rows["normal"][box_arrays.faces],
            box.face_normals[:, None, :].repeat(3, axis=1),
        ),
        "sphere vertex colors carried": (
            sphere_arrays.rows["color"][sphere_arrays.faces]
            == sphere.visual.vertex_colors[sphere.faces]
        ).all(),
        "translucent vertex colors blend": sphere_arrays.material["flags"][0]
        == ALPHA_MODES["BLEND"],
        "opaque box": box_arrays.material["flags"][0] == ALPHA_MODES["OPAQUE"],
        "fuze uv carried": g.np.allclose(
            fuze_arrays.rows["uv"][fuze_arrays.faces], fuze.visual.uv[fuze.faces]
        ),
        "fuze has a texture": "base_color_texture" in fuze_arrays.textures,
        "obj materials are not metal": fuze_arrays.material["factors"][0] == 0.0,
        "material keeps vertex colors": (
            painted_arrays.rows["color"] == [200, 150, 40, 255]
        ).all()
        and painted_arrays.material["factors"][0] == 0.5,
        "every material value": glass_arrays.material.tobytes() == expected.tobytes(),
        "every glTF alpha mode": blend_modes == set(ALPHA_MODES.values()),
        "unskinned has no skin": box_arrays.skin is None,
        "skin rows": skinned_arrays.skin.dtype == SKIN,
        "skin follows corners": g.np.allclose(
            skinned_arrays.skin["weights"][skinned_arrays.faces],
            weights[skinned.faces],
        )
        and g.np.allclose(
            skinned_arrays.skin["joints"][skinned_arrays.faces], joints[skinned.faces]
        ),
        "wrench segment count": len(wrench_arrays.rows) == segment_count,
        "wrench colors per segment": (
            wrench_arrays.rows["color"] == [10, 20, 30, 255]
        ).all(),
        "points are zero length segments": g.np.allclose(
            point_arrays.rows["segment_start"], point_arrays.rows["segment_end"]
        ),
        "points kind": point_arrays.kind == "points",
        "voxels become boxes": geometry_to_arrays(voxels).kind == "mesh",
        "converted once": cached,
        "recolored": (recolored.face_colors == [0, 0, 255, 255]).all(),
    }
    for name, ok in checks.items():
        assert ok, name


def block_offsets(prefix, dtype, start=0):
    """The byte offset numpy gives every uniform block member, named like OpenGL."""
    offsets = {}
    for name in dtype.names:
        field, offset = dtype.fields[name][:2]
        member = f"{prefix}.{name}"
        if field.base.names is not None:
            offsets.update(block_offsets(f"{member}[0]", field.base, start + offset))
            continue
        # arrays of vectors and matrices are named by their first element
        if len(field.shape) == 3 or (
            len(field.shape) == 2 and field.shape[0] != field.shape[1]
        ):
            member = f"{member}[0]"
        offsets[member] = start + offset
    return offsets


def test_layout():
    """GLSL generated from numpy dtypes is laid out exactly where numpy puts it."""
    import ctypes
    import shutil
    import subprocess

    from trimesh.viewer.opengl import shaders

    # the generated declarations are exactly what a person would write
    inputs = """
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 color;
layout(location = 4) in vec4 joints;
layout(location = 5) in vec4 weights;
layout(location = 6) in mat4 model_view;
layout(location = 10) in mat3 normal_matrix;
layout(location = 13) in float node_index;
layout(location = 14) in float joint_offset;
"""
    block = """
struct Light {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 falloff;
    vec4 shadow;
};
layout(std140, row_major) uniform Frame {
    mat4 projection;
    mat4 inverse_projection;
    vec4 viewport_size;
    ivec4 output_mode;
    ivec4 light_count;
    Light lights[8];
    mat4 shadow_matrices[4];
    mat4 environment_rotation;
    vec4 environment_color;
    vec4 irradiance[9];
    vec4 ground_light;
    vec4 occlusion_radius;
    vec4 exposure;
    vec4 bloom;
    vec4 line_widths;
    ivec4 smooth_shading;
} frame;
"""
    material_block = """
layout(std140, row_major) uniform Material {
    vec4 base_color;
    vec4 emissive;
    vec4 factors;
    vec4 transmission_color;
    ivec4 flags;
    ivec4 has_textures;
} material;
"""
    checks = {
        "mesh inputs": inputs.strip() in shaders.shader_source("mesh.vert"),
        "frame block": block.strip() in shaders.shader_source("mesh.frag"),
        "material block": material_block.strip() in shaders.shader_source("mesh.frag"),
    }
    # a block field std140 would pad differently than numpy is refused
    for block_name, original in shaders.BLOCKS.items():
        for name, dtype in {
            "vec3": [("padded", "<f4", 3)],
            "offset": [("scalar", "<f4"), ("vector", "<f4", 4)],
            "mat3": [("matrix", "<f4", (3, 3))],
        }.items():
            shaders.BLOCKS[block_name] = g.np.dtype(dtype)
            shaders._SHADER_SOURCE.clear()
            checks[f"std140 {block_name} refuses {name}"] = False
            try:
                shaders.shader_source("mesh.frag")
            except ValueError:
                checks[f"std140 {block_name} refuses {name}"] = True
            finally:
                shaders.BLOCKS[block_name] = original
                shaders._SHADER_SOURCE.clear()

    # the filled shaders pass the reference compiler, which is stricter
    # than drivers, as the programs `create_context` compiles validated
    # as SPIR-V: stage interfaces are checked by the driver linking them
    validator = shutil.which("glslangValidator")
    checks["glslangValidator installed"] = validator is not None or not g.all_dependencies
    with g.TemporaryDirectory() as directory:
        for file_name in g.os.listdir(shaders.DIRECTORY):
            if not file_name.endswith((".vert.glsl", ".frag.glsl")):
                continue
            name = file_name.removesuffix(".glsl")
            source = shaders.shader_source(name)
            checks[f"{name} fully filled"] = "$" not in source
            with open(g.os.path.join(directory, name), "w") as f:
                f.write(source)
        for program in shaders.PROGRAMS:
            if validator is None:
                continue
            command = [validator, "-G", "--auto-map-bindings", "--auto-map-locations"]
            command += ["--spirv-val", "-l", "-o", g.os.path.join(directory, "spv")]
            command += [f"{program}.vert", f"{program}.frag"]
            result = subprocess.run(command, cwd=directory, capture_output=True)
            checks[f"{program} validates"] = result.returncode == 0

    g.require_rendering()
    from trimesh.viewer.opengl.viewer import gl

    viewer = g.offscreen_viewer(g.trimesh.creation.box().scene())
    viewer.to_array()
    for name, item in viewer.context.programs.items():
        program, dtypes = item.program, shaders.INPUTS.get(name, ())
        # the passes with no inputs read the frame, and only meshes a material
        blocks = ("Frame", "Material")[: 1 + (name == "mesh")]
        # every input is where `shaders.attribute_locations` put it
        for name, location in shaders.attribute_locations(*dtypes).items():
            found = gl.glGetAttribLocation(
                program, ctypes.create_string_buffer(name.encode())
            )
            checks[f"{program} {name} location"] = found == location

        # each block is the size of its dtype and every member is where numpy put it
        offsets = {}
        for block_name in blocks:
            block = gl.glGetUniformBlockIndex(
                program, ctypes.create_string_buffer(block_name.encode())
            )
            size = gl.GLint()
            gl.glGetActiveUniformBlockiv(
                program, block, gl.GL_UNIFORM_BLOCK_DATA_SIZE, ctypes.byref(size)
            )
            dtype = shaders.BLOCKS[block_name]
            checks[f"{program} {block_name} size"] = size.value == dtype.itemsize
            offsets.update(block_offsets(block_name, dtype))
        for name, offset in offsets.items():
            names = (ctypes.c_char_p * 1)(name.encode())
            index, found = gl.GLuint(), gl.GLint()
            gl.glGetUniformIndices(
                program,
                1,
                ctypes.cast(names, ctypes.POINTER(ctypes.POINTER(gl.GLchar))),
                ctypes.byref(index),
            )
            gl.glGetActiveUniformsiv(
                program, 1, ctypes.byref(index), gl.GL_UNIFORM_OFFSET, ctypes.byref(found)
            )
            # an unknown name would leave the offset at zero and pass by accident
            checks[f"{program} {name} found"] = index.value != gl.GL_INVALID_INDEX
            checks[f"{program} {name} offset"] = found.value == offset
    viewer.close()
    checks["no OpenGL errors"] = gl.glGetError() == 0

    for name, ok in checks.items():
        assert ok, name


def rgb(scene, **kwargs):
    """A render's color as (h, w, 3) float64."""
    return g.render(scene, **kwargs).color[..., :3].astype(g.np.float64)


def shade_reference(scene, normal=None, lights=None):
    """
    The sRGB 0-255 image the viewer should draw for opaque `PBRMaterial`
    meshes, NaN where no ray hits, evaluated independently from camera
    rays and the glTF and `KHR_lights_punctual` specs, optionally with a
    constant tangent space `normal` map and `(lights, transforms)` from
    `render_lights` ending with the environment.
    """
    np = g.np
    if lights is None:
        lights = render_lights(scene, scene.camera_transform)
    lights, transforms = lights
    width, height = scene.camera.resolution
    meshes = [mesh.smooth_shaded for mesh in scene.dump()]
    hits = g.camera_hits(scene, meshes)
    points, corners = hits.points, hits.combined.faces[hits.face]
    barycentric = g.trimesh.triangles.points_to_barycentric(
        hits.combined.triangles[hits.face], points
    )
    surface = g.trimesh.util.unitize(
        (hits.combined.vertex_normals[corners] * barycentric[:, :, None]).sum(axis=1)
    )
    view = -g.trimesh.util.unitize(hits.vectors[hits.ray])
    # light the back of a triangle as if it were the front
    surface[(surface * view).sum(axis=1) < 0.0] *= -1.0
    if normal is not None:
        # the tangent and bitangent are how position changes with uv
        uv = np.vstack([m.visual.uv for m in meshes])[corners]
        position = hits.combined.vertices[corners]
        inverse = np.linalg.inv(uv[:, 1:] - uv[:, :1])
        edges = position[:, 1:] - position[:, :1]
        tangent = g.trimesh.util.unitize(np.matmul(inverse[:, :1], edges)[:, 0])
        bitangent = g.trimesh.util.unitize(np.matmul(inverse[:, 1:2], edges)[:, 0])
        surface = g.trimesh.util.unitize(
            tangent * normal[0] + bitangent * normal[1] + surface * normal[2]
        )

    materials = [m.visual.material for m in meshes]
    base = (np.array([m.baseColorFactor for m in materials]) / 255.0)[hits.mesh, :3]
    metallic = np.array([[m.metallicFactor] for m in materials])[hits.mesh]
    roughness = np.clip([[m.roughnessFactor] for m in materials], 0.04, 1.0)[hits.mesh]
    reflectance = 0.04 * (1.0 - metallic) + base * metallic
    diffuse_color = base * (1.0 - metallic)
    # `KHR_materials_diffuse_transmission` mixes the dielectric Lambert
    # reflection with Lambert transmission of light from the other side
    transmission = np.array([[m.diffuseTransmissionFactor or 0.0] for m in materials])
    transmission_color = np.ones((len(materials), 3))
    for index, material in enumerate(materials):
        if material.diffuseTransmissionColorFactor is not None:
            transmission_color[index] = material.diffuseTransmissionColorFactor
    transmission_color = (transmission_color * transmission)[hits.mesh] * (1.0 - metallic)
    diffuse_color = diffuse_color * (1.0 - transmission[hits.mesh])
    alpha_squared = roughness**4
    geometry_k = (roughness + 1.0) ** 2 / 8.0
    normal_dot_view = np.maximum((surface * view).sum(axis=1, keepdims=True), 1e-4)

    color = np.zeros_like(points)
    for light, pose in zip(lights[:-1], transforms[:-1]):
        # lights shine down their -Z, and only directional lights have
        # no position, falloff or range
        backwards = g.trimesh.util.unitize(pose[:3, 2])
        to_light = np.tile(backwards, (len(points), 1))
        distance_squared = np.ones((len(points), 1))
        window = np.ones_like(distance_squared)
        if not isinstance(light, DirectionalLight):
            to_light = pose[:3, 3] - points
            distance_squared = (to_light**2).sum(axis=1, keepdims=True)
            if light.radius is not None:
                window = np.clip(1.0 - (distance_squared / light.radius**2) ** 2, 0, 1)
        direction = g.trimesh.util.unitize(to_light)
        cone = np.ones_like(distance_squared)
        if isinstance(light, SpotLight):
            inner, outer = np.cos(light.innerConeAngle), np.cos(light.outerConeAngle)
            cosine = (direction * backwards).sum(axis=1, keepdims=True)
            cone = np.clip((cosine - outer) / max(inner - outer, 1e-3), 0.0, 1.0)
        radiance = light.color[:3] / 255.0 * light.intensity * (window * cone) ** 2
        radiance = radiance / distance_squared

        half = g.trimesh.util.unitize(direction + view)
        cosine_light = (surface * direction).sum(1, keepdims=True)
        normal_dot_light = np.maximum(cosine_light, 0.0)
        # light from behind is transmitted with the fresnel term of the
        # light mirrored through the surface
        behind = np.maximum(-cosine_light, 0.0)
        mirrored = g.trimesh.util.unitize(direction + 2.0 * behind * surface + view)
        mirrored_dot_half = np.clip((view * mirrored).sum(1, keepdims=True), 0.0, 1.0)
        transmitted = (
            (1.0 - reflectance)
            * (1.0 - (1.0 - mirrored_dot_half) ** 5)
            * transmission_color
            / np.pi
        )
        normal_dot_half = np.maximum((surface * half).sum(1, keepdims=True), 0.0)
        view_dot_half = np.maximum((view * half).sum(1, keepdims=True), 0.0)
        fresnel = reflectance + (1.0 - reflectance) * (1.0 - view_dot_half) ** 5
        distribution = alpha_squared / (
            np.pi * (normal_dot_half**2 * (alpha_squared - 1.0) + 1.0) ** 2
        )
        geometry = (
            normal_dot_view / (normal_dot_view * (1 - geometry_k) + geometry_k)
        ) * (normal_dot_light / (normal_dot_light * (1 - geometry_k) + geometry_k))
        specular = (
            fresnel
            * distribution
            * geometry
            / (4 * normal_dot_view * normal_dot_light + 1e-4)
        )
        diffuse = (1.0 - fresnel) * diffuse_color / np.pi
        color += (
            (diffuse + specular) * normal_dot_light + transmitted * behind
        ) * radiance

    # Karis's fit of the environment BRDF
    fit = roughness * [-1.0, -0.0275, -0.572, 0.022] + [1.0, 0.0425, 1.04, -0.04]
    bias = (
        np.minimum(fit[:, :1] ** 2, np.exp2(-9.28 * normal_dot_view)) * fit[:, :1]
        + fit[:, 1:2]
    )
    # diffuse sees the irradiance of the environment from its harmonics and
    # specular the reflection from the mip levels blurred like GGX lobes on
    # either side of the roughness, which `test_light` checks against
    # brute force integrals of the cube map, in the environment's frame
    environment, pose = lights[-1], transforms[-1]
    tint = environment.color[:3] / 255.0 * environment.intensity
    rotation = g.trimesh.util.unitize(pose[:3, :3].T)
    above, below = (
        spherical_harmonics(normals @ rotation.T) @ environment.irradiance * tint / np.pi
        for normals in (surface, -surface)
    )
    color += diffuse_color * above + transmission_color * below
    reflected = 2.0 * (surface * view).sum(axis=1, keepdims=True) * surface - view
    levels = environment_levels(environment.cubemap)
    samples = np.array(
        [cubemap_sample(level, reflected @ rotation.T) for level in levels]
    )
    level = roughness[:, 0] * (len(levels) - 1)
    low, rows = np.floor(level).astype(np.int64), np.arange(len(level))
    fraction = (level - low)[:, None]
    high = np.minimum(low + 1, len(levels) - 1)
    reflection = (
        samples[low, rows] * (1 - fraction) + samples[high, rows] * fraction
    ) * tint
    color += reflection * (
        reflectance * (-1.04 * bias + fit[:, 2:3]) + 1.04 * bias + fit[:, 3:4]
    )

    # the Khronos PBR Neutral tone mapper
    darkest = color.min(axis=1, keepdims=True)
    color -= np.where(darkest < 0.08, darkest - 6.25 * darkest**2, 0.04)
    peak = color.max(axis=1, keepdims=True)
    start, headroom = 0.76, 0.24
    new_peak = 1.0 - headroom**2 / (peak + headroom - start)
    fade = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0)
    compressed = color * new_peak / np.maximum(peak, 1e-12)
    compressed = compressed * (1 - fade) + new_peak * fade
    color = np.clip(np.where(peak < start, color, compressed), 0.0, 1.0)
    srgb = np.where(color < 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)

    image = np.full((height, width, 3), np.nan)
    image[hits.rows, hits.columns] = srgb * 255.0
    return image


def pbr_mesh(mesh, color, metallic, roughness, uv=None):
    """Give a mesh a plain `PBRMaterial`, with zero uv unless passed."""
    if uv is None:
        uv = g.np.zeros((len(mesh.vertices), 2))
    mesh.visual = g.trimesh.visual.TextureVisuals(
        uv=uv,
        material=g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=metallic, roughnessFactor=roughness
        ),
    )
    return mesh


def lit_scene(lights=(), poses=(), plane=False, environment=DARK, **material):
    """A sphere, or a plane seen from above, lit by `lights` and an `environment`."""
    options = {"color": [200, 120, 60, 255], "metallic": 0.0, "roughness": 0.5}
    options.update(material)
    mesh = g.trimesh.creation.icosphere(subdivisions=4)
    view = {"points": mesh.bounds, "distance": 4.0}
    if plane:
        mesh = g.trimesh.creation.box(extents=[10.0, 10.0, 0.1])
        mesh.apply_translation([0, 0, -0.05])
        rotation = tf.euler_matrix(0.5, 0, 0.2)
        view = {"points": [[-5, -5, 0], [5, 5, 0]], "rotation": rotation}
    scene = pbr_mesh(mesh, **options).scene()
    # square pixels, which is the only camera glTF can describe: a
    # wider image keeps the vertical field of view and sees more
    scene.camera = g.trimesh.scene.Camera(
        name="camera", resolution=[150, 150], fov=(40, 40)
    )
    scene.camera.fit_resolution((200, 150))
    scene.camera_transform = scene.camera.look_at(**view)
    scene.lights = list(lights)
    for light, pose in zip(lights, poses):
        scene.graph[light.name] = pose
    return g.set_environment(scene, environment)


def brighter(scene, factor, emissive=(0.1, 0.05, 0.0)):
    """
    A copy of a scene glowing `emissive` with its lights, including the
    default lights and environment, and its glow `factor` times as bright.
    """
    lights, transforms = render_lights(scene, scene.camera_transform)
    copy = scene.copy()
    copy.lights = g.copy.deepcopy(lights)
    for light, pose in zip(copy.lights, transforms):
        light.intensity *= factor
        copy.graph[light.name] = pose
    for geometry in copy.geometry.values():
        geometry.visual.material.emissiveFactor = g.np.multiply(emissive, factor)
    return copy


def lighting_cases():
    """Scenes lit by every kind of light and environment with a fixed camera."""
    down = tf.translation_matrix([0, 0, 2.0])
    under = tf.translation_matrix([0, 0, -2.0])
    # a shadow casting light shining up from below at an angle
    backlight = DirectionalLight(name="backlight", color=WHITE, intensity=1.0)
    upward = tf.euler_matrix(g.np.pi - 0.3, 0.2, 0.0)
    sun = DirectionalLight(name="sun", color=[255, 200, 150, 255], intensity=3.0)
    bulb = PointLight(name="bulb", color=WHITE, intensity=6.0)
    ranged = PointLight(name="bulb", color=WHITE, intensity=6.0, radius=3.0)
    spot = SpotLight(
        name="spot", color=WHITE, intensity=20.0, innerConeAngle=0.2, outerConeAngle=0.5
    )
    lamp = PointLight(name="lamp", color=WHITE, intensity=40.0)
    # a softbox which is bright, as plastic only reflects 4% of light head on
    window = studio([0.05] * 3, [0.05] * 3, [([0.6, 0.6, 0.5], 30.0, [40.0] * 3)])
    gradient = studio([0.9, 0.8, 0.7], [0.05, 0.05, 0.1])
    # a softbox twice a unit vector away, which is read as a direction
    reflected = studio([0.8, 0.7, 0.6], [0.05, 0.08, 0.1], [([1.4, 1, 1], 20.0, [4] * 3)])
    below = studio([0.02] * 3, [0.3, 0.2, 0.4])
    metal = {"metallic": 1.0, "roughness": 0.3}
    semi = {"metallic": 0.5, "roughness": 0.3}
    dim = ambient([0.1] * 3)
    # name : (lights, poses, `lit_scene` keywords)
    table = {
        # a colored directional light on a rotated node
        "directional": ([sun], [tf.euler_matrix(-0.6, 0.4, 0.0)], {}),
        # a point light over a plane falls off with the distance squared
        "point": ([bulb], [down], {"plane": True}),
        # with a range it is exactly dark past the range
        "point range": ([ranged], [down], {"plane": True, "environment": dim}),
        # a spot light lights a disk with a soft edge
        "spot": ([spot], [down], {"plane": True}),
        # a thin surface lit only from behind shows the light it lets
        # through, from lights and a bright ground under it, without
        # shadowing itself from the light it lets through
        "diffuse transmission": (
            [bulb, backlight],
            [under, upward],
            {"plane": True, "environment": below},
        ),
        # metals reflect the ambient light rather than going black
        "ambient metal": ([], [], {"environment": ambient([0.6] * 3), **metal}),
        # a light moved below to exactly the camera makes the half vector
        # the view vector, where rounding must not make any pixel NaN
        "headlamp": ([lamp], [g.np.eye(4)], {"plane": True}),
        # a gradient reflected in a metal, and a softbox reflected sharply
        # in a glossy surface and blurred in a rough one
        "gradient metal": ([], [], {"environment": gradient, **metal}),
        "softbox glossy": ([], [], {"environment": window, "roughness": 0.2}),
        "softbox rough": ([], [], {"environment": window, "roughness": 1.0}),
        # which pins the environment's axes and math in a semi-metal
        "custom environment": ([], [], {"environment": reflected, **semi}),
    }
    cases = {
        name: lit_scene(lights, poses, **keywords)
        for name, (lights, poses, keywords) in table.items()
    }
    cases["headlamp"].graph["lamp"] = cases["headlamp"].camera_transform
    for geometry in cases["diffuse transmission"].geometry.values():
        # only the top of the plane, so nothing under it blocks the light
        geometry.update_faces(geometry.face_normals[:, 2] > 0.5)
        geometry.remove_unreferenced_vertices()
        geometry.visual.material.diffuseTransmissionFactor = 0.5
        geometry.visual.material.diffuseTransmissionColorFactor = [0.3, 1.0, 0.2]
    # the default lights and studio around the camera
    for name, color in {
        "autolight": [180, 180, 180, 255],
        "autolight red": [200, 30, 30, 255],
        "autolight white": [255, 255, 255, 255],
    }.items():
        sphere = g.trimesh.creation.icosphere(subdivisions=4)
        cases[name] = pbr_mesh(sphere, color, 0.0, 0.5).scene()
        cases[name].camera.fit_resolution((200, 150))
    return cases


def floor_points(scene, height=0.0):
    """Where each pixel's camera ray meets the plane at `height`, as an image."""
    width, rows = scene.camera.resolution
    origins, vectors, pixels = scene.camera_rays()
    distance = (height - origins[:, 2]) / vectors[:, 2]
    image = g.np.zeros((rows, width, 3))
    image[rows - 1 - pixels[:, 1], pixels[:, 0]] = origins + vectors * distance[:, None]
    return image


def floor_shadow(scene, pose, axis, height=0.0):
    """Pixels seeing the floor at `height`, and those not seeing the light at `pose[:, axis]`."""
    points = floor_points(scene, height)
    hits = g.camera_hits(scene)
    # the height of each first hit, which is the floor where nothing is hit
    first = g.np.full(points.shape[:2], height)
    first[hits.rows, hits.columns] = hits.points[:, 2]
    floor = g.np.abs(first - height) < 1e-3
    # the light's homogeneous position: directional lights are at
    # infinity along their +Z, where `w` is zero
    to_light = pose[:3, axis] - points[floor] * pose[3, axis]
    blocked = floor.copy()
    blocked[floor] = hits.combined.ray.intersects_any(
        points[floor] + [0, 0, 1e-4], g.trimesh.util.unitize(to_light)
    )
    return floor, blocked


def test_lighting():
    """Every light and environment shades each pixel as the glTF spec says."""
    from scipy import ndimage

    # what makes some cases worth checking, from their interior pixels
    # and how far those are along the floor from under the light
    extra = {
        "ambient metal": {"is lit": lambda pixels, radius: pixels.mean() > 30},
        # a window-like highlight which a rough surface blurs away
        "softbox glossy": {"has a highlight": lambda pixels, radius: pixels.max() > 200},
        # a range of 3 from 2 above the floor leaves only the ambient light
        "point range": {
            "has pixels past the range": lambda pixels, radius: (
                (radius > 5**0.5 + 0.1).sum() > 100
            )
        },
        # highlights are white, but the body stays close to its 0.15
        "autolight red": {
            "stays red": lambda pixels, radius: (
                g.np.median(pixels[:, 1] / pixels[:, 0]) < 0.35
            )
        },
        # the default look keeps white surfaces bright without clipping them
        "autolight white": {
            "not clipped": lambda pixels, radius: pixels.max() < 253,
            "is bright": lambda pixels, radius: 110 < pixels.mean() < 220,
        },
    }
    cases = lighting_cases()
    checks = {}
    for name, scene in cases.items():
        image = rgb(scene, resolution=scene.camera.resolution)
        reference = shade_reference(scene)
        # stay away from the antialiased silhouette
        # a pixel past which a shaded edge is not antialiased into this one
        interior = ndimage.binary_erosion(g.ray_reference(scene).covered)
        radius = g.np.linalg.norm(floor_points(scene)[interior], axis=-1)
        checks[f"{name} matches reference"] = (
            g.np.abs(image - reference)[interior].max() <= 2.0
        )
        checks[f"{name} is not flat"] = g.np.ptp(reference[interior]) > 40
        checks.update(
            g.ray_checks(
                scene,
                g.render(
                    scene,
                    include_depth=True,
                    include_segmentation=True,
                    resolution=scene.camera.resolution,
                ),
                f"{name} ",
            )
        )
        for check, predicate in extra.get(name, {}).items():
            checks[f"{name} {check}"] = predicate(image[interior], radius)
        # exposure scales all light before tone mapping, which is the same
        # as every light, the environment and anything glowing twice as bright
        exposed, doubled = (
            rgb(scene_copy, resolution=scene.camera.resolution, **options)
            for scene_copy, options in (
                (brighter(scene, 1.0), {"exposure": 2.0}),
                (brighter(scene, 2.0), {}),
            )
        )
        checks[f"{name} exposure doubles light"] = g.np.abs(exposed - doubled).max() <= 1
        checks[f"{name} exposure brightens"] = (exposed - image)[interior].mean() > 10

    # everything a render uses but the environment survives glTF, and
    # the default lights are the same after loading a scene without lights
    for name, scene in cases.items():
        checks.update(round_trip_checks(name, scene))

    # the camera can see the environment behind the scene: a constant one
    # is one color, which the tone mapper lowers by 0.04 and is then sRGB
    # encoded, and a gradient from a dark floor to a bright sky gets
    # brighter upwards
    expected = (1.055 * (0.3 - 0.04) ** (1 / 2.4) - 0.055) * 255
    for name, (environment, predicate) in {
        "background is the environment": (
            ambient([0.3] * 3),
            lambda image: g.np.abs(image[:3, :3] - expected).max() <= 1,
        ),
        "background sky is up": (
            studio([0.8] * 3, [0.0] * 3),
            lambda image: image[0, 0].mean() > image[-1, 0].mean() + 40,
        ),
    }.items():
        scene = g.set_environment(
            g.trimesh.creation.box(extents=[0.1, 0.1, 0.1]).scene(), environment
        )
        checks[name] = predicate(rgb(scene, resolution=[80, 60], show_environment=True))

    # `KHR_materials_unlit` is the final linear color, its base color times
    # its texture and vertex colors, whatever the lights, environment or
    # exposure are, which stays unlit after a glTF round trip
    from PIL import Image

    color = g.trimesh.visual.color
    texture, vertex = [180, 220, 140], [230, 200, 250, 255]
    sun = DirectionalLight(name="sun", color=WHITE, intensity=20.0)
    dark = DirectionalLight(name="sun", color=WHITE, intensity=0.0)
    for name, (light, environment, exposure) in {
        "bright sun": (sun, ambient([3.0] * 3), 1.0),
        "dark": (dark, DARK, 1.0),
        "exposed": (sun, ambient([3.0] * 3), 2.0),
    }.items():
        scene = lit_scene([light], [SUN_POSE], plane=True, environment=environment)
        for geometry in scene.geometry.values():
            material = geometry.visual.material
            material.unlit = True
            material.baseColorFactor = [0.9, 0.5, 0.3, 1.0]
            material.baseColorTexture = Image.fromarray(
                g.np.full((2, 2, 3), texture, dtype=g.np.uint8)
            )
            geometry.visual.vertex_attributes["color"] = g.np.tile(
                vertex, (len(geometry.vertices), 1)
            )
        linear = (
            material.baseColorFactor[:3]
            / 255.0
            * color.srgb_to_linear(g.np.divide(texture, 255.0))
            * color.srgb_to_linear(g.np.divide(vertex[:3], 255.0))
        )
        image = rgb(scene, resolution=scene.camera.resolution, exposure=exposure)
        hits = g.camera_hits(scene)
        seen = g.np.zeros(image.shape[:2], dtype=bool)
        seen[hits.rows, hits.columns] = True
        interior = ndimage.binary_erosion(seen, iterations=2)
        checks[f"unlit {name} is its color"] = interior.mean() > 0.5 and (
            g.np.abs(image[interior] - color.linear_to_srgb(linear) * 255.0).max() <= 1
        )
        checks.update(round_trip_checks(f"unlit {name}", scene))

    # a camera node with a scale, which files exported from 3ds Max carry,
    # sees what glTF says: the same image with its planes in its own units,
    # so a near plane which would cut the sphere doesn't once scaled, and
    # the scale moves nothing else: geometry fixed to the camera, or a scroll
    gradient = studio(
        [0.8, 0.7, 0.6], [0.05, 0.05, 0.1], [([0.6, 0.6, 0.5], 30.0, [2.0] * 3)]
    )
    for projection, magnification in [
        ("perspective", None),
        ("orthographic", g.np.array([3.0, 2.25])),
    ]:
        glossy = lit_scene(environment=gradient, metallic=0.5, roughness=0.3)
        pose = glossy.camera_transform
        if magnification is not None:
            glossy.camera = g.trimesh.scene.Camera(
                name="camera", resolution=[80, 60], magnification=magnification
            )
            glossy.camera_transform = pose
        # a badge halfway to the camera which stays where the camera saw it
        glossy.add_geometry(
            g.trimesh.creation.box(extents=[0.2] * 3),
            geom_name="badge",
            transform=tf.translation_matrix(pose[:3, 3] / 2.0),
        )
        glossy.camera.z_near = 0.1
        viewers = {"plain": g.offscreen_viewer(glossy, (80, 60), fixed=["badge"])}
        # and an orthographic camera's magnification in its own units
        glossy.camera.z_near = 0.1 / 40.0
        glossy.camera.z_far = glossy.camera.z_far / 40.0
        if magnification is not None:
            glossy.camera.magnification = magnification / 40.0
        glossy.camera_transform = pose @ tf.scale_matrix(40.0)
        viewers["scaled"] = g.offscreen_viewer(glossy, (80, 60), fixed=["badge"])
        images, scrolled = {}, {}
        for name, viewer in viewers.items():
            images[name] = viewer.to_array().color.astype(g.np.int64)
            viewer.on_mouse_scroll(0, 0, 0, 1)
            scrolled[name] = viewer.to_array().color.astype(g.np.int64)
        checks[f"{projection} camera scale is glTF's"] = (
            g.np.abs(images["plain"] - images["scaled"]).max() <= 1
        )
        checks[f"{projection} scaled camera scrolls the same"] = g.np.allclose(
            viewers["plain"].camera_transform, viewers["scaled"].camera_transform
        ) and (g.np.abs(scrolled["plain"] - scrolled["scaled"]).max() <= 1)
        checks[f"{projection} scroll changes the image"] = (
            g.np.abs(scrolled["plain"] - images["plain"]).max() > 50
        )
        for viewer in viewers.values():
            viewer.close()

    for name, ok in checks.items():
        assert ok, name


def test_material_maps():
    """A normal map tilts shading along uv and an occlusion map darkens only ambient."""
    from PIL import Image
    from scipy import ndimage

    def constant(rgb):
        """A small image of one color."""
        return Image.fromarray(g.np.full((4, 4, 3), rgb, dtype=g.np.uint8))

    # a flat square seen from above with uv along x and y, lit only by a
    # low sun so a tilt is a big change in brightness
    vertices = g.np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    square = g.trimesh.Trimesh(vertices, [[0, 1, 2], [0, 2, 3]], process=False)
    scene = square.scene()
    scene.camera.fit_resolution((160, 120))
    scene.camera_transform = scene.camera.look_at(
        vertices, rotation=tf.euler_matrix(0.3, 0.2, 0.0)
    )
    scene.lights = [DirectionalLight(name="sun", color=WHITE, intensity=3.0)]
    scene.graph["sun"] = tf.euler_matrix(0.0, 1.0, 0.0)
    g.set_environment(scene, DARK)
    tilt = g.np.radians(30.0)
    normals = {
        "flat": [0, 0, 1],
        "u": [g.np.sin(tilt), 0, g.np.cos(tilt)],
        "v": [0, g.np.sin(tilt), g.np.cos(tilt)],
    }
    checks = {}
    for name, uv in {
        "uv along +x": (vertices[:, :2] + 1.0) / 2.0,
        # mirrored textures are common and flip the tangent
        "uv along -x": (vertices[:, :2] * [-1.0, 1.0] + 1.0) / 2.0,
    }.items():
        images = {}
        for direction, normal in normals.items():
            encoded = g.np.round((g.np.array(normal) * 0.5 + 0.5) * 255)
            pbr_mesh(square, [200, 200, 200, 255], 0.0, 0.6, uv=uv)
            square.visual.material.normalTexture = constant(encoded)
            image = rgb(scene, resolution=scene.camera.resolution)
            # the reference reads the decoded normal like the shader does
            reference = shade_reference(scene, normal=encoded / 255.0 * 2.0 - 1.0)
            interior = ndimage.binary_erosion(
                ~g.np.isnan(reference[..., 0]), iterations=2
            )
            checks[f"{name} {direction} matches"] = (
                g.np.abs(image - reference)[interior].max() <= 2.0
            )
            checks.update(round_trip_checks(f"{name} {direction}", scene))
            images[direction] = image[interior]
        checks[f"{name} tilt changes the image"] = (
            g.np.abs(images["u"] - images["flat"]).mean() > 20
            and g.np.abs(images["v"] - images["flat"]).mean() > 5
        )

    # occlusion scales the ambient light and nothing else
    renders = {}
    for name, (light, occlusion) in {
        "dark": (0.5, constant([0] * 3)),
        "open": (0.5, constant([255] * 3)),
        "half": (0.5, constant([128] * 3)),
        "no ambient": (0.0, None),
        "no map": (0.5, None),
        "half ambient": (0.5 * 128 / 255, None),
    }.items():
        sphere = pbr_mesh(
            g.trimesh.creation.icosphere(subdivisions=3), [200, 120, 60, 255], 0.0, 0.5
        )
        sphere.visual.material.occlusionTexture = occlusion
        scene = sphere.scene()
        scene.lights = [DirectionalLight(name="sun", color=WHITE, intensity=2.0)]
        scene.graph["sun"] = tf.euler_matrix(-0.5, 0.5, 0.0)
        g.set_environment(scene, ambient([light] * 3))
        renders[name] = g.render(scene).color.astype(g.np.int64)
        checks.update(round_trip_checks(f"occlusion {name}", scene))
    same = {"dark": "no ambient", "open": "no map", "half": "half ambient"}
    for first, second in same.items():
        checks[f"occlusion {first} is {second}"] = (
            g.np.abs(renders[first] - renders[second]).max() <= 1
        )
    checks["occlusion does something"] = (
        g.np.abs(renders["dark"] - renders["open"]).max() > 20
    )

    for name, ok in checks.items():
        assert ok, name


def box_shadow(light, pose):
    """A box over a plane lit by one light and a dim ambient."""
    scene = lit_scene([light], [pose], plane=True, environment=ambient([0.2] * 3))
    box = g.trimesh.creation.box(extents=[2, 2, 1])
    box = pbr_mesh(box.apply_translation([0, 0, 1.5]), [90, 160, 220, 255], 0, 0.5)
    scene.add_geometry(box)
    return scene


def test_shadows():
    """Lights shadow exactly where rays to them are blocked, leaving only ambient there."""
    from scipy import ndimage

    above = tf.translation_matrix([0.5, 0.0, 5.0])
    spot = SpotLight(name="spot", color=WHITE, intensity=60.0, outerConeAngle=0.7)
    checks = {}
    # name, light, pose, which pose column is its position, casts shadows
    for name, light, pose, axis, casts in [
        # from infinitely far away at an angle so the shadow is beside the box
        (
            "directional",
            DirectionalLight(name="sun", color=WHITE, intensity=3.0),
            SUN_POSE,
            2,
            True,
        ),
        # down from a point, so the shadow is bigger than the box
        ("spot", spot, above, 3, True),
        ("point", PointLight(name="bulb", color=WHITE, intensity=40.0), above, 3, False),
    ]:
        scene = box_shadow(light, pose)
        image = rgb(scene, resolution=scene.camera.resolution)
        unshadowed = shade_reference(scene)
        # only the environment, which `set_environment` puts last
        scene.lights = scene.lights[-1:]
        ambient_only = rgb(scene, resolution=scene.camera.resolution)
        floor, blocked = floor_shadow(scene, pose, axis)

        # only where the light would visibly light the plane can it be
        # shadowed, and there a pixel nearer to unlit than lit is shadow
        dark, bright = ambient_only.sum(axis=-1), unshadowed.sum(axis=-1)
        lights_plane = floor & (bright > dark + 30)
        rendered = lights_plane & (image.sum(axis=-1) < 0.5 * (bright + dark))
        blocked &= lights_plane & casts
        umbra = ndimage.binary_erosion(blocked, iterations=2)
        lit = ndimage.binary_erosion(lights_plane & ~blocked, iterations=2)
        checks[f"{name} shadow where rays are blocked"] = (
            not (rendered | blocked).any() or g.iou(rendered, blocked) > 0.98
        )
        checks[f"{name} umbra is only ambient"] = (
            not umbra.any() or g.np.abs(image - ambient_only)[umbra].max() <= 2
        )
        checks[f"{name} lit is unshadowed"] = g.np.abs(image - unshadowed)[lit].max() <= 2
        checks[f"{name} has lit pixels"] = lit.sum() > 500
        checks[f"{name} has a shadow"] = (umbra.sum() > 100) == casts

    # a light node's scale isn't part of the light, which shines down its
    # -Z from where it is, and is what files from 3ds Max carry
    sun = DirectionalLight(name="sun", color=WHITE, intensity=3.0)
    plain, scaled = (
        rgb(box_shadow(sun, pose), resolution=[160, 120])
        for pose in [SUN_POSE, SUN_POSE @ tf.scale_matrix(40.0)]
    )
    checks["light node scale changes nothing"] = g.np.abs(plain - scaled).max() <= 1

    # viewers drawing frame after frame redraw their shadows and skinned
    # meshes exactly when what casts them moves, though every hidden viewer
    # shares the shadow maps and the joint matrices
    from unittest import mock

    from trimesh.scene.animation import Skin

    scene = box_shadow(sun, SUN_POSE)
    box_node = scene.graph.nodes_geometry[-1]
    # a box moved by a bone rather than its own node
    skinned = colored_box([255, 0, 0, 255], extents=[1.0] * 3)
    skinned.vertex_attributes["joints"] = g.np.zeros((8, 4), dtype=g.np.int64)
    skinned.vertex_attributes["weights"] = g.np.tile([1.0, 0, 0, 0], (8, 1))
    scene.graph.update(frame_to="bone", matrix=tf.translation_matrix([-2.5, 0, 0.5]))
    scene.add_geometry(skinned, node_name="skinned")
    scene.skins["skinned"] = Skin(joints=["bone"])
    shift = tf.translation_matrix([1.5, 0.0, 0.0])
    moved = scene.copy()
    moved.graph.update(frame_to=box_node, matrix=shift)
    moved.graph.update(frame_to="bone", matrix=tf.translation_matrix([-2.5, 2.0, 0.5]))
    # without ambient occlusion, which draws depth every frame
    viewers = [g.offscreen_viewer(item, resolution=(160, 120)) for item in (scene, moved)]

    def frame(viewer):
        """Draw a frame of a scene whose geometry hasn't changed and read it."""
        viewer.render(changed=False)
        return viewer.read_pixels("resolved").astype(g.np.int64)

    still, shifted = frame(viewers[0]), frame(viewers[1])
    checks["moving the box moves pixels"] = g.np.abs(still - shifted).max() > 50
    # the same pixels as before the other viewer drew
    checks["shadow and skin kept"] = (frame(viewers[0]) == still).all()
    # nothing casting shadows moved, so moving the camera draws no shadows
    viewer = viewers[0]
    with mock.patch.object(viewer, "draw_rows", wraps=viewer.draw_rows) as drawn:
        viewer.camera_transform = (
            tf.translation_matrix([0, 0, 1]) @ viewer.camera_transform
        )
        viewer.to_array()
    modes = [call.args[2] for call in drawn.call_args_list]
    checks["moving the camera draws no shadows"] = modes == ["color"]
    viewer.camera_transform = viewer.initial_camera.copy()
    scene.graph.update(frame_to=box_node, matrix=shift)
    scene.graph.update(frame_to="bone", matrix=tf.translation_matrix([-2.5, 2.0, 0.5]))
    checks["shadow follows the box"] = (frame(viewers[0]) == shifted).all()
    for viewer in viewers:
        viewer.close()

    for name, ok in checks.items():
        assert ok, name


def test_depth_segmentation():
    """Depth and segmentation are what each camera ray hits, for any lens and scale."""

    def instanced():
        """One box drawn by three nodes, so instancing can't merge ids, and a sphere."""
        scene = g.trimesh.Scene()
        scene.geometry["small"] = g.trimesh.creation.box(extents=[0.6, 0.6, 0.6])
        scene.geometry["large"] = g.trimesh.creation.icosphere(radius=0.8)
        for node, geometry, offset in [
            ("small_0", "small", [-1.0, 0.0, 0.0]),
            ("small_1", "small", [0.0, 0.0, 0.0]),
            ("small_2", "small", [1.0, 0.0, 0.0]),
            ("large", "large", [0.0, 1.3, -1.0]),
        ]:
            matrix = tf.translation_matrix(offset)
            scene.graph.update(frame_to=node, geometry=geometry, matrix=matrix)
        return scene

    def nested():
        """Two boxes under a rotated and scaled parent node."""
        scene = g.trimesh.Scene()
        parent = tf.euler_matrix(0.3, 0.5, 0.1) @ tf.scale_matrix(1.5)
        scene.graph.update(frame_to="parent", matrix=parent)
        for index, offset in enumerate([-0.8, 0.8]):
            scene.add_geometry(
                g.trimesh.creation.box(extents=[0.7, 1.0, 0.5]),
                node_name=f"child_{index}",
                parent_node_name="parent",
                transform=tf.translation_matrix([offset, 0, 0]),
            )
        return scene

    def translucent():
        """The instanced scene behind a translucent sheet."""
        scene = instanced()
        scene.add_geometry(
            colored_box([0, 0, 255, 100], extents=[1.0, 1.0, 0.2]),
            node_name="glass",
            transform=tf.translation_matrix([0, 0, 1.0]),
        )
        return scene

    checks = {}
    # perspective, and orthographic seeing 0.6 of the scene's scale
    lenses = {
        "perspective": lambda size: None,
        "orthographic": lambda size: (size * 0.8, size * 0.6),
    }
    for (name, build), (lens, magnification), scale in g.itertools.product(
        {"instanced": instanced, "nested": nested, "translucent": translucent}.items(),
        lenses.items(),
        # the depth range has to fit tiny and huge scenes
        [1.0, 1e-3, 1e4],
    ):
        scene = build().scaled(scale)
        scene.camera.fit_resolution((160, 120))
        scene.camera_transform = scene.camera.look_at(
            scene.bounds, rotation=tf.euler_matrix(0.4, 0.3, 0.0)
        )
        scene.camera.magnification = magnification(scene.scale)
        arrays = g.render(
            scene, include_depth=True, include_segmentation=True, resolution=[160, 120]
        )
        depth, index = arrays.depth, arrays.segmentation
        # a camera's planes are lengths which scale with the scene
        rescaled = scene.scaled(7.0).camera
        label = f"{name} {lens} {scale}"
        checks[f"{label} planes scale"] = g.np.allclose(
            [rescaled.z_near, rescaled.z_far],
            [scene.camera.z_near * 7.0, scene.camera.z_far * 7.0],
        )
        checks[f"{label} shapes"] = depth.shape == index.shape == (120, 160)
        checks[f"{label} dtypes"] = (
            depth.dtype == g.np.float32 and index.dtype == g.np.int64
        )
        checks[f"{label} background depth zero"] = (depth[index < 0] == 0.0).all()
        checks[f"{label} foreground depth positive"] = (depth[index >= 0] > 0.0).all()
        checks.update(g.ray_checks(scene, arrays, f"{label} "))
        # including every node sharing one geometry
        checks[f"{label} every node has an id"] = set(
            range(len(scene.graph.nodes_geometry))
        ) <= set(g.np.unique(index))

    # each part is returned only when asked for
    viewer = g.offscreen_viewer(instanced(), resolution=[80, 60])
    color_only = viewer.to_array()
    everything = viewer.to_array(include_depth=True, include_segmentation=True)
    depth_only = viewer.to_array(include_depth=True)
    # hiding a node hides that instance but not others sharing its geometry
    viewer.hide_geometry("small_0")
    seen = set(g.np.unique(viewer.to_array(include_segmentation=True).segmentation))
    nodes = viewer.scene.graph.nodes_geometry
    viewer.close()
    checks["only color unless asked"] = (
        color_only.depth is None and color_only.segmentation is None
    )
    checks["only depth when asked"] = depth_only.segmentation is None
    checks["parts agree"] = (everything.color == color_only.color).all() and (
        everything.depth == depth_only.depth
    ).all()
    checks["hide one instance"] = (
        nodes.index("small_0") not in seen
        and {nodes.index("small_1"), nodes.index("small_2")} <= seen
    )

    # an orthographic camera sees the same image from any distance
    scene = nested()
    scene.camera.fit_resolution((160, 120))
    scene.camera.magnification = (2.4, 1.8)
    images = []
    for distance in [5.0, 50.0]:
        scene.camera_transform = tf.translation_matrix([0, 0, distance])
        images.append(g.render(scene, include_depth=True, resolution=[160, 120]))
    near, far = images
    mask = near.depth > 0
    checks["orthographic color is distance free"] = (
        g.np.abs(near.color.astype(int) - far.color.astype(int)).max() <= 1
    )
    checks["orthographic depth moves with camera"] = g.np.allclose(
        far.depth[mask] - near.depth[mask], 45.0, rtol=1e-4
    )
    # and the scroll wheel zooms the viewer's own camera rather than moving it
    viewer = g.offscreen_viewer(scene, resolution=[160, 120])
    before = viewer.camera_transform.copy()
    viewer.on_mouse_scroll(0, 0, 0, 1)
    checks["orthographic scroll zooms"] = g.np.allclose(
        viewer.camera.magnification, [2.4 * 0.9, 1.8 * 0.9]
    ) and g.np.allclose(viewer.camera_transform, before)
    checks["scene camera untouched"] = g.np.allclose(
        scene.camera.magnification, [2.4, 1.8]
    )
    viewer.close()

    # lines have depth and an id too
    scene = g.trimesh.Scene(g.trimesh.creation.box(extents=[0.2, 0.2, 0.2]))
    scene.add_geometry(g.trimesh.load_path(SEGMENT), node_name="line")
    scene.camera.fit_resolution((160, 120))
    scene.camera_transform = tf.translation_matrix([0, 0.5, 4.0])
    for lens, magnification in {"perspective": None, "orthographic": (2.0, 1.5)}.items():
        scene.camera.magnification = magnification
        _, depth, index = g.render(
            scene, include_depth=True, include_segmentation=True, resolution=[160, 120]
        )
        line = index == scene.graph.nodes_geometry.index("line")
        checks[f"{lens} line has pixels"] = line.sum() > 50
        checks[f"{lens} line depth"] = g.np.allclose(depth[line], 4.0, rtol=1e-4)

    for name, ok in checks.items():
        assert ok, name


def box_on_floor(sun=None, ground=False, light=1.0):
    """A white box on a white or the invisible `ground` floor, seen from above."""
    white = [235, 235, 235, 255]
    scene = g.trimesh.Scene()
    box = pbr_mesh(g.trimesh.creation.box(extents=[2.0, 2.0, 1.0]), white, 0.0, 0.8)
    scene.add_geometry(box, node_name="box", transform=tf.translation_matrix([0, 0, 0.5]))
    if not ground:
        floor = pbr_mesh(g.trimesh.creation.box(extents=[12, 12, 0.1]), white, 0, 0.8)
        scene.add_geometry(floor, transform=tf.translation_matrix([0, 0, -0.05]))
    scene.lights = []
    if sun is not None:
        scene.lights = [DirectionalLight(name="sun", color=WHITE, intensity=2.0)]
        scene.graph["sun"] = sun
    # square pixels: 40 degrees tall is about 52 wide at 4:3
    scene.camera = g.trimesh.scene.Camera(resolution=[240, 180], fov=(52, 40))
    # looking straight down at the floor, which is the case where the
    # studio can't use the camera's backwards direction for its frame
    scene.camera_transform = tf.rotation_matrix(1.4, [0, 0, 1]) @ tf.translation_matrix(
        [0, 0, 9.0]
    )
    environment = ambient([light] * 3)
    # with +Z up
    up = tf.rotation_matrix(g.np.pi / 2.0, [1, 0, 0])
    return g.set_environment(scene, environment, pose=up)


def occlusion_reference(scene, radius, count=64):
    """The cosine weighted fraction of rays from each pixel's surface hitting within `radius`."""
    width, height = scene.camera.resolution
    hits = g.camera_hits(scene, [scene.to_mesh()])
    normals = hits.combined.face_normals[hits.face]
    # face the camera like the viewer's two sided lighting
    normals[(normals * hits.vectors[hits.ray]).sum(axis=1) > 0] *= -1.0
    # cosine weighted directions on a spiral around +Z, turned onto each normal
    fraction = (g.np.arange(count) + 0.5) / count
    angle = g.np.arange(count) * 2.39996
    radial = g.np.sqrt(fraction)
    local = g.np.column_stack(
        (radial * g.np.cos(angle), radial * g.np.sin(angle), g.np.sqrt(1.0 - fraction))
    )
    rotations = g.np.array(
        [
            g.trimesh.geometry.align_vectors([0, 0, 1], normal)[:3, :3]
            for normal in normals
        ]
    )
    directions = g.np.matmul(local, g.np.swapaxes(rotations, 1, 2)).reshape((-1, 3))
    starts = g.np.repeat(hits.points + normals * 1e-4 * radius, count, axis=0)
    ends, ray, _ = hits.combined.ray.intersects_location(
        starts, directions, multiple_hits=False
    )
    distance = g.np.full(len(starts), g.np.inf)
    distance[ray] = g.np.linalg.norm(ends - starts[ray], axis=1)
    image = g.np.full((height, width), g.np.nan)
    image[hits.rows, hits.columns] = (distance < radius).reshape((-1, count)).mean(axis=1)
    return image


def test_studio_floor():
    """Occlusion darkens only where rays say the sky is hidden; the floor is only shadow."""
    from scipy import ndimage

    scene = box_on_floor()
    off, on = (
        rgb(scene, resolution=[240, 180], flags={"ambient_occlusion": enabled})
        for enabled in [False, True]
    )
    darkening = off.mean(axis=-1) - on.mean(axis=-1)
    reference = occlusion_reference(scene, OCCLUSION_RADIUS * scene.scale)
    covered = ndimage.binary_erosion(~g.np.isnan(reference), iterations=2)
    open_surface = covered & ndimage.binary_erosion(reference == 0.0, iterations=3)
    crease = covered & (reference >= 0.3)
    near = covered & (ndimage.distance_transform_edt(reference < 0.05) < 12)
    # with only a sun and a black environment there is nothing to occlude
    sunny = box_on_floor(sun=tf.euler_matrix(0.4, 0.3, 0.0), light=0.0)
    off, on = (
        g.render(
            sunny,
            include_depth=True,
            include_segmentation=True,
            flags={"ambient_occlusion": enabled},
            resolution=[240, 180],
        )
        for enabled in [False, True]
    )
    checks = {
        "has open surface": open_surface.sum() > 2000,
        "has creases": crease.sum() > 100,
        "open surface untouched": g.np.abs(darkening[open_surface]).max() <= 3.0,
        "crease darkened": darkening[crease].mean() >= 8.0,
        "crease mostly darker": (darkening[crease] >= 2.0).mean() >= 0.9,
        "tracks reference": g.np.corrcoef(darkening[near], reference[near])[0, 1] > 0.6,
        "direct light untouched": (
            g.np.abs(off.color.astype(int) - on.color.astype(int)).max() <= 1
        ),
    }
    checks.update(g.ray_checks(sunny, on, "ambient occlusion "))

    # the invisible floor under a sun
    scene = box_on_floor(sun=SUN_POSE, ground=True, light=0.3)
    options = {"studio": True, "flags": {"ambient_occlusion": False}}
    arrays = g.render(
        scene,
        include_depth=True,
        include_segmentation=True,
        resolution=[240, 180],
        **options,
    )
    color = arrays.color
    image = color[..., :3].astype(g.np.float64)
    # the floor is just under the box, far closer than a pixel
    floor, blocked = floor_shadow(scene, SUN_POSE, 2)
    shaded = floor & (image.mean(axis=-1) < 250)
    umbra = ndimage.binary_erosion(blocked, iterations=3)
    lit = ndimage.binary_erosion(floor & ~blocked, iterations=3)
    # the floor loses the sun and keeps the environment, where a white
    # floor facing up reflects the constant environment and `I / pi` of
    # the sun's intensity `I` times the cosine of its angle from up
    umbra_value = 255.0 * 0.3 / (0.3 + 2.0 * SUN_POSE[2, 2] / g.np.pi)
    checks.update(
        {
            "has a shadow": umbra.sum() > 300,
            "shadow where rays are blocked": g.iou(shaded, blocked) > 0.95,
            "umbra loses the sun": g.np.abs(
                image[umbra].mean(axis=-1) - umbra_value
            ).max()
            <= 3,
            "lit floor invisible": g.np.abs(image[lit] - 255.0).max() <= 1,
            "not in bounds": g.np.allclose(scene.bounds, [[-1, -1, 0], [1, 1, 1]]),
            "not dumped": len(scene.dump()) == 1,
        }
    )
    checks.update(g.ray_checks(scene, arrays, "floor "))
    # from below the floor can't be seen at all
    scene.camera_transform = (
        tf.rotation_matrix(g.np.pi, [1, 0, 0]) @ scene.camera_transform
    )
    with_ground, without_ground = (
        rgb(scene, resolution=[240, 180], ground_shadow=ground, **options)
        for ground in [True, False]
    )
    checks["invisible from below"] = g.np.abs(with_ground - without_ground).max() <= 1

    # bloom spills light the tone mapper compressed onto its neighbors: a
    # glowing square in the dark lights nothing around it until bloom is on
    from trimesh.viewer.opengl.viewer import BLOOM_SPREAD

    square = pbr_mesh(
        g.trimesh.creation.box(extents=[0.2, 0.2, 0.01]), [0, 0, 0, 255], 0.0, 1.0
    )
    square.visual.material.emissiveFactor = [8.0] * 3
    glowing = square.scene()
    glowing.lights = [DirectionalLight(name="off", intensity=0.0)]
    glowing.camera = g.trimesh.scene.Camera(resolution=[160, 120], fov=(52, 40))
    glowing.camera_transform = tf.translation_matrix([0.0, 0.0, 3.0])
    images = {}
    for name, bloom in [("off", 0.0), ("on", 1.0), ("off again", 0.0)]:
        g.set_environment(glowing, DARK)
        images[name] = g.render(
            glowing, resolution=[160, 120], background=[0, 0, 0, 255], bloom=bloom
        ).color.astype(int)
    hits = g.camera_hits(glowing)
    inside = g.np.zeros((120, 160), dtype=bool)
    inside[hits.rows, hits.columns] = True
    # within and past the antialiased edge, along the middle row both ways
    within = ndimage.binary_erosion(inside, iterations=1)
    outside = ~ndimage.binary_dilation(inside, iterations=1)
    row = int(hits.rows.mean())
    columns = g.np.flatnonzero(inside[row])
    bloom = images["on"][..., :3].mean(axis=-1) - images["off"][..., :3].mean(axis=-1)
    falloff = [
        bloom[row, columns.max() + 2 :],
        bloom[row, : columns.min() - 1][::-1],
    ]
    checks.update(
        {
            "glowing square is white": images["off"][within].min() >= 253,
            "no bloom is the background": not images["off"][outside][:, :3].any(),
            "bloom only adds": (images["on"] >= images["off"]).all(),
            "bloom keeps the square": (
                g.np.abs(images["on"] - images["off"])[within].max() <= 1
            ),
            # out past the edge for at least one standard deviation of the glow
            "bloom lights neighbors": all(
                (side > 0).sum() >= BLOOM_SPREAD * 120 for side in falloff
            ),
            "bloom fades with distance": all(
                (g.np.diff(side) <= 0).all() and side[-1] == 0 for side in falloff
            ),
            "bloom off again is never on": g.np.array_equal(
                images["off"], images["off again"]
            ),
        }
    )

    for name, ok in checks.items():
        assert ok, name


def test_export_unchanged():
    """Viewing and rendering never change a scene or the bytes it exports to."""
    from PIL import Image

    from trimesh.viewer.opengl.viewer import SceneViewer
    from trimesh.viewer.render import render_frames, stage

    g.require_rendering()
    scenes = {
        "box": g.trimesh.creation.box().scene(),
        "box moves": g.dolly_scene("box"),
        "camera rides": g.dolly_scene("dolly"),
    }
    for name in sorted(g.os.listdir(g.dir_models)):
        if name.endswith((".glb", ".gltf")):
            scenes[name] = g.trimesh.load_scene(g.os.path.join(g.dir_models, name))

    checks = {}
    for name, scene in scenes.items():
        if scene.is_empty:
            # empty scenes can't be exported at all
            continue
        before = scene.export(file_type="glb")
        checks[f"{name} export is deterministic"] = (
            scene.export(file_type="glb") == before
        )
        state = (
            scene.has_camera,
            scene.has_lights,
            g.copy.deepcopy(scene.metadata),
            [id(item) for item in [*scene.cameras, *scene.animations]],
        )
        # a render reads the scene's own geometry, so values cached on
        # it like `smooth_shaded` are computed once for every render
        staged = stage(scene)[0]
        checks[f"{name} render shares geometry"] = all(
            staged.geometry[key] is value for key, value in scene.geometry.items()
        )

        graph = hash(scene.graph)
        # render settings are keywords of each call, never scene content
        settings = {
            "exposure": 2.0,
            "bloom": 0.5,
            "ground_shadow": False,
            "show_environment": True,
            "line_width": 3.0,
            "point_size": 3.0,
        }
        scene.render(resolution=[64, 48], **settings)
        scene.save_image(resolution=[64, 48], **settings)
        scene.show(visible=False, start_loop=False, **settings).close()
        viewer = SceneViewer(scene, visible=False, start_loop=False, resolution=[64, 48])
        viewer.to_array(include_depth=True)
        # playing an animation draws it without posing the scene
        viewer.on_animate(0.37)
        viewer.to_array()
        checks[f"{name} graph unchanged while playing"] = hash(scene.graph) == graph
        viewer.close()
        # an animation has a frame for every tenth of its duration
        gif = scene.render(file_type="gif", resolution=[32, 24], duration=0.2, fps=10)
        frames = Image.open(g.trimesh.util.wrap_as_stream(gif)).n_frames
        checks[f"{name} gif has every frame"] = frames == 2

        checks[f"{name} export unchanged"] = scene.export(file_type="glb") == before
        checks[f"{name} graph unchanged"] = hash(scene.graph) == graph
        checks[f"{name} state unchanged"] = (
            scene.has_camera,
            scene.has_lights,
            scene.metadata,
            [id(item) for item in [*scene.cameras, *scene.animations]],
        ) == state

    # every frame of a render is exactly a still of the scene posed at its
    # time, with lights in the scene so both light it from the same place
    fps = 10
    for name in ["box moves", "camera rides"]:
        scene = g.set_environment(scenes[name], EnvironmentLight(name="sky"))
        frames = list(render_frames(scene, resolution=[64, 48], duration=0.5, fps=fps))
        for index, frame in enumerate(frames):
            posed = scene.copy()
            posed.animate(index / fps)
            (still,) = render_frames(posed, resolution=[64, 48], still=True)
            difference = g.np.abs(frame.astype(int) - still).max()
            checks[f"{name} frame {index} is the posed scene"] = difference <= 1
        checks[f"{name} moves"] = not (frames[0] == frames[-1]).all()

    for name, ok in checks.items():
        assert ok, name


def test_viewer():
    """Renders match camera rays every time, sharing a window and freeing the GPU."""
    g.require_rendering()
    import pyglet

    from trimesh.viewer.opengl.viewer import SceneViewer, gl

    # how to check each kind of OpenGL name a viewer records still exists
    exists = {
        "buffers": gl.glIsBuffer,
        "textures": gl.glIsTexture,
        "framebuffers": gl.glIsFramebuffer,
        "vertex_arrays": gl.glIsVertexArray,
        "programs": gl.glIsProgram,
    }

    mesh = g.trimesh.creation.box(extents=[3.0, 1.0, 2.0])
    mesh.apply_transform(tf.euler_matrix(0.3, 0.7, 0.2))
    scene = mesh.scene()

    checks = {}
    windows = set()
    # alternate resolutions including odd ones
    for index, resolution in enumerate([(320, 240), (97, 61), (1024, 768)] * 2):
        viewer = g.offscreen_viewer(scene, resolution=resolution)
        for render in range(3):
            arrays = viewer.to_array(include_depth=True, include_segmentation=True)
            image = arrays.color
            label = f"{resolution} pass {index} render {render}"
            checks[f"{label} shape"] = image.shape == (*resolution[::-1], 4)
            if render == 0:
                checks.update(g.ray_checks(scene, arrays, prefix=f"{label} "))
                first = image
            checks[f"{label} repeats"] = (image == first).all()
        windows.add(id(viewer.window))
        # everything the viewer and its geometry allocated
        owned = [viewer.names, *(item.names for item in viewer.geometry_buffers.values())]
        viewer.close()
    checks["no OpenGL errors"] = gl.glGetError() == 0
    checks["one hidden window"] = len(windows) == 1
    for kind, check in exists.items():
        names = [name for item in owned for name in item.get(kind, [])]
        checks[f"{kind} freed"] = not any(check(name) for name in names)
    checks["framebuffers were made"] = len(owned[0]["framebuffers"]) > 0

    # a window frees what its context allocated before it closes
    visible = SceneViewer(scene, start_loop=False, resolution=(64, 48))
    context = visible.context
    freed = {}
    close_window = visible.window.close

    def checked_close():
        """Check what the context still holds just before the window closes."""
        visible.window.switch_to()
        for kind, values in [*context.names.items(), *visible.names.items()]:
            freed[kind] = not any(exists[kind](name) for name in values)
        close_window()

    visible.window.close = checked_close
    visible.to_array()
    # resized to another shape, which a hidden window can't really be,
    # the image keeps square pixels and the camera's vertical view
    visible.window.get_framebuffer_size = lambda: (97, 61)
    visible.on_resize(97, 61)
    scene.camera.fit_resolution((97, 61))
    checks["resize keeps the vertical view"] = g.np.allclose(
        visible.camera.fov, scene.camera.fov
    )
    checks.update(g.ray_checks(scene, visible.to_array(True, True), prefix="resized "))
    visible.close()
    checks["visible window closed"] = len(freed) == len(exists)
    for name, ok in freed.items():
        checks[f"visible window {name} freed"] = ok

    # `render` draws exactly what its camera sees, of instances and
    # textures too, and leaves the camera as it was
    sphere = g.trimesh.creation.icosphere
    assembly = g.trimesh.Scene({"small": sphere(2, 0.6), "large": sphere(2, 1.0)})
    offsets = {"small": (-2.5, 0, 0), "small_1": (2.5, 0, 0), "small_2": (0, 0, 2.5)}
    for node, offset in offsets.items():
        assembly.graph.update(
            frame_to=node, matrix=tf.translation_matrix(offset), geometry="small"
        )
    assembly.graph.update(frame_to="large", matrix=tf.translation_matrix([0, 2.5, 0]))
    fuze = g.trimesh.load_scene(g.dir_models + "/fuze.obj")
    for name, current in {"box": scene, "fuze": fuze, "assembly": assembly}.items():
        # without the floor every drawn pixel is the scene's geometry
        before = [*current.camera.resolution, *current.camera.fov]
        image = g.png(current.render(resolution=[400, 300], ground_shadow=False))
        label = f"render {name}"
        checks[f"{label} size"] = image.shape[:2] == (300, 400)
        after = [*current.camera.resolution, *current.camera.fov]
        checks[f"{label} camera unchanged"] = g.np.allclose(after, before)
        # and an odd size, which catches off by one pixel
        arrays = g.render(
            current, include_depth=True, include_segmentation=True, resolution=[257, 193]
        )
        checks.update(g.ray_checks(current, arrays, f"{label} "))
    # the hidden window isn't one of the user's, which would keep
    # `pyglet.app.run` from ever returning
    checks["no windows left open"] = len(pyglet.app.windows) == 0

    if g.sys.platform.startswith("linux"):
        # a machine with only OpenGL 3.0, which Mesa pretends to be, gets
        # one error saying what is needed and is left with no windows
        script = [
            "import pyglet, trimesh",
            "try:",
            "    trimesh.creation.box().scene().render(resolution=[32, 32])",
            "except Exception as error:",
            "    print(error)",
            "print(len(pyglet.app.windows))",
        ]
        environment = {
            key: value for key, value in g.os.environ.items() if key != "DISPLAY"
        }
        environment["MESA_GL_VERSION_OVERRIDE"] = "3.0"
        environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        )
        result = g.subprocess.run(
            [g.sys.executable, "-c", "\n".join(script)],
            env=environment,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
        checks["old OpenGL says what it needs"] = "OpenGL 3.3 core" in result.stdout
        checks["old OpenGL leaves no windows"] = lines[-1:] == ["0"]

    # with culling off only the depth test hides the back face, so a
    # box with a red front and blue back must only show red
    front = colored_box([0, 0, 255, 255])
    front.visual.face_colors[front.face_normals[:, 2] > 0.5] = [255, 0, 0, 255]
    checks["depth test hides the back"] = (
        mean_color(g.render(front.scene(), flags={"cull": False}).color)[0] > 0.95
    )

    # a node scaled to nothing, which animations do, draws nothing
    # without breaking the rest of the scene
    flat = g.trimesh.Scene(mesh)
    flat.add_geometry(g.trimesh.creation.box(), transform=g.np.diag([0.0, 0.0, 0.0, 1.0]))
    arrays = g.render(flat, include_depth=True, include_segmentation=True)
    checks.update(g.ray_checks(flat, arrays, "zero scale node "))

    # a render must not depend on what was rendered before it
    translucent = colored_box([0, 0, 255, 100]).scene()
    images = [g.render(current).color for current in [scene, translucent, scene]]
    checks["renders are independent"] = (images[0] == images[2]).all()

    for name, ok in checks.items():
        assert ok, name


def test_methods():
    """Colors, transparency, lines, points and scene changes show up in the next render."""
    from scipy import ndimage

    g.require_rendering()
    import pyglet
    from pyglet.window import key

    def framed(*geometry):
        """A scene of `geometry` with a camera looking down -Z at the origin."""
        scene = g.trimesh.Scene(list(geometry))
        scene.camera_transform = tf.translation_matrix([0, 0, 5])
        return scene

    # an uncolored mesh renders lit, not black
    image = g.render(g.trimesh.creation.box().scene()).color
    checks = {"default mesh is lit": image[(image[..., :3] < 250).any(-1)].mean() > 60}

    # face colors, vertex colors and changing them in place
    for kind in ["face", "vertex"]:
        box = g.trimesh.creation.box()
        setattr(box.visual, f"{kind}_colors", [255, 0, 0, 255])
        viewer = g.offscreen_viewer(box.scene())
        checks[f"{kind} colors red"] = mean_color(viewer.to_array().color)[0] > 0.95
        setattr(box.visual, f"{kind}_colors", [0, 0, 255, 255])
        checks[f"{kind} colors changed"] = mean_color(viewer.to_array().color)[2] > 0.95
        viewer.close()

    # a red box behind a translucent blue sheet shows both colors
    # no matter which was added to the scene first
    red = colored_box([255, 0, 0, 255])
    blue = colored_box([0, 0, 255, 128], extents=[3.0, 3.0, 0.01])
    blue.apply_translation([0.0, 0.0, 1.0])
    for name, order in {"red first": [red, blue], "blue first": [blue, red]}.items():
        center = g.render(framed(*order)).color[120, 160].astype(g.np.int64)
        checks[f"transparency {name}"] = center[0] > 60 and center[2] > 60

    # clear glass shows what is behind it but still reflects, which on a
    # black background is all there is to see where nothing is behind it
    bare, glass = (
        g.render(framed(*geometry), background=[0, 0, 0, 255]).color[..., :3].astype(int)
        for geometry in [[red], [red, clear_glass(0.01, 1.0, 0.2)]]
    )
    behind, through = bare[120, 160], glass[120, 160]
    checks["glass shows what is behind it"] = (
        through[0] >= behind[0] - 10 and (through[1:] <= behind[1:] + 40).all()
    )
    # a corner of the sheet with nothing behind it
    checks["glass reflects"] = glass[40, 60].sum() > bare[40, 60].sum()
    # translucency blends what the display shows, so a half transparent
    # glowing sheet over black is half as bright as an opaque one
    glow = {}
    for name, alpha in {"opaque": 255, "translucent": 128}.items():
        sheet = g.trimesh.creation.box(extents=[3.0, 3.0, 0.01])
        sheet.visual = g.trimesh.visual.TextureVisuals(
            material=g.trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0, 0, 0, alpha],
                emissiveFactor=[0.9, 0.5, 0.2],
                alphaMode="BLEND",
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
        )
        lit = framed(sheet)
        lit.lights = []
        g.set_environment(lit, DARK)
        image = g.render(lit, background=[0, 0, 0, 255]).color
        glow[name] = image[120, 160, :3].astype(g.np.float64)
    checks["translucency blends after tone mapping"] = (
        glow["opaque"].min() > 50
        and g.np.abs(glow["translucent"] - glow["opaque"] * 128 / 255).max() <= 2
    )

    for width in [1, 4, 9]:
        # black on white so darkness is coverage: summed down every
        # column across the middle of the line it is exactly the width
        segment = framed(g.trimesh.load_path(SEGMENT))
        image = g.render(segment, line_width=width).color
        coverage = (255 - image[:, 110:210, 0].astype(g.np.float64)).sum(axis=0) / 255
        checks[f"line width {width}"] = (g.np.abs(coverage - width) < 0.25).all()

    points = g.trimesh.PointCloud(
        [[-1, -1, 0], [1, 0.5, 0], [0, 0, 0]], colors=[255, 0, 0, 255]
    )
    scene = framed(points)
    image = g.render(scene, point_size=8).color
    width, height = scene.camera.resolution
    # normalized device coordinates of every point through the camera
    homogeneous = g.np.column_stack((points.vertices, g.np.ones(len(points.vertices))))
    world_to_clip = scene.camera.projection @ g.np.linalg.inv(scene.camera_transform)
    clip = homogeneous @ world_to_clip.T
    ndc = clip[:, :3] / clip[:, 3:]
    columns = ((ndc[:, 0] + 1) / 2 * width).astype(int)
    rows = ((1 - ndc[:, 1]) / 2 * height).astype(int)
    checks["points drawn at projections"] = (image[rows, columns, 0] > 200).all() and (
        image[rows, columns, 1] < 60
    ).all()
    drawn = (image[..., :3] < 250).any(-1)
    checks["points are separate dots"] = ndimage.label(drawn)[1] == 3
    # and lit like little spheres from the upper left of the view: the
    # upper left quarter of every dot is brighter than the lower right,
    # inside the antialiased edge which is lightened by the background
    labels, count = ndimage.label(ndimage.binary_erosion(drawn))
    brightness = image[..., :3].astype(g.np.float64).sum(axis=-1)
    index = g.np.arange(1, count + 1)
    middle = g.np.zeros((count + 1, 2))
    middle[1:] = ndimage.center_of_mass(labels > 0, labels, index)
    # each pixel's (row, column) offset from the middle of its dot
    offset = g.np.indices(labels.shape) - g.np.moveaxis(middle[labels], -1, 0)
    lit = ndimage.mean(brightness, labels * (offset < 0).all(axis=0), index)
    shaded = ndimage.mean(brightness, labels * (offset > 0).all(axis=0), index)
    checks["points are lit spheres"] = (
        count == 3 and (g.np.asarray(lit) > g.np.asarray(shaded) + 30).all()
    )

    # path colors changed in place show up
    line = g.trimesh.load_path(SEGMENT)
    scene = framed(line)
    viewer = g.offscreen_viewer(scene, line_width=5)
    viewer.to_array()
    line.colors = [[0, 255, 0, 255]]
    checks["path color changed"] = (
        viewer.to_array().color[120, 160, :3] == [0, 255, 0]
    ).all()

    # adding, sharing, and deleting geometry in a live viewer
    scene.add_geometry(colored_box([255, 0, 0, 255], extents=[0.5] * 3), geom_name="b")
    scene.graph.update(
        frame_to="other", geometry="b", matrix=tf.translation_matrix([1.5, 0, 0])
    )
    image = viewer.to_array().color
    # count red blobs bigger than a few antialiased edge pixels
    labels, count = ndimage.label((image[..., 0] > 150) & (image[..., 1] < 80))
    sizes = ndimage.sum_labels(labels > 0, labels, range(1, count + 1))
    checks["added and shared geometry"] = (sizes > 20).sum() == 2
    scene.delete_geometry("b")
    image = viewer.to_array().color
    checks["deleted geometry"] = not ((image[..., 0] > 150) & (image[..., 1] < 80)).any()
    viewer.close()

    # removing faces in place is reflected in the next render: the
    # inside of the box is back faces so draw them to match the rays
    box = g.trimesh.creation.box()
    viewer = g.offscreen_viewer(box.scene(), flags={"cull": False})
    before = viewer.to_array().color
    box.update_faces(box.face_normals[:, 2] < 0.5)
    arrays = viewer.to_array(include_depth=True, include_segmentation=True)
    checks["update faces"] = (arrays.color != before).any()
    checks.update(g.ray_checks(viewer.scene, arrays, "after update "))
    viewer.close()

    # keys toggle state and change the image
    scene = g.get_mesh("cycloidal.3DXML")
    scene.add_geometry(g.get_mesh("fuze.obj"))
    scene.add_geometry(g.get_mesh("2D/spline.DXF"))
    viewer = g.offscreen_viewer(scene)
    base = viewer.to_array().color
    for symbol in [key.W, key.S, key.C, key.A, key.G]:
        viewer.on_key_press(symbol=symbol, modifiers=0)
        checks[f"key {symbol} redraws"] = viewer.state.dirty
        checks[f"key {symbol} changes the image"] = not (
            viewer.to_array().color == base
        ).all()
        viewer.on_key_press(symbol=symbol, modifiers=0)
    viewer.on_key_press(symbol=key.Q, modifiers=0)
    checks["quit"] = viewer.closed

    # the viewer holds a pyglet window rather than being one, as a
    # subclass would shadow the window's many internal attributes
    viewer = g.offscreen_viewer(g.trimesh.creation.box().scene())
    checks["holds a window"] = isinstance(viewer.window, pyglet.window.Window)
    checks["is not a window"] = not isinstance(viewer, pyglet.window.Window)
    viewer.close()

    # subclassing works
    class Viewer(g.trimesh.viewer.opengl.viewer.SceneViewer):
        pass

    subclass = Viewer(g.trimesh.creation.box().scene(), visible=False, start_loop=False)
    checks["subclass"] = subclass.to_array().color.ndim == 3
    subclass.close()

    # what `main`'s viewer took and had warns and works until April 2028
    with g.pytest.warns(DeprecationWarning, match="April 2028"):
        legacy = g.offscreen_viewer(
            g.trimesh.creation.box().scene(), line_settings={"point_size": 7.0}, record=1
        )
    with g.pytest.warns(DeprecationWarning, match="April 2028"):
        legacy.toggle_wireframe()
    with g.pytest.warns(DeprecationWarning, match="April 2028"):
        legacy.add_geometry("box", g.trimesh.creation.box())
    checks["legacy keywords"] = legacy.render_settings["point_size"] == 7.0
    checks["legacy toggles"] = legacy.state.wireframe and legacy.state.scene_changed
    checks["unknown attribute"] = not hasattr(legacy, "toggle_bogus")
    legacy.close()

    for name, ok in checks.items():
        assert ok, name


def test_video():
    """A video render has every frame of the animation, or a turntable, as rendered."""
    from trimesh.interfaces import ffmpeg

    if not ffmpeg.exists:
        if g.include_rendering:
            raise ValueError("rendering video requires ffmpeg!")
        g.pytest.skip("rendering video requires ffmpeg")
    g.require_rendering()
    resolution = (160, 120)
    fps = 10

    def animated_camera(scene, camera, index, count):
        """The scene's camera at frame `index`, which may ride an animated node."""
        scene.animate(index / fps)
        return scene.camera_transform

    def turntable_camera(scene, camera, index, count):
        """The camera orbiting the world axis nearest its up, keeping the horizon level."""
        up = g.np.abs(camera[:3, 1]).argmax()
        axis = g.np.eye(3)[up] * g.np.sign(camera[up, 1])
        return (
            tf.rotation_matrix(2 * g.np.pi * index / count, axis, scene.centroid) @ camera
        )

    # a tilted camera
    static = g.trimesh.creation.box(extents=[3, 1, 2]).scene()
    static.camera_transform = tf.euler_matrix(0.5, 0.2, 0.1) @ tf.translation_matrix(
        [0, 0, 6]
    )

    checks = {}
    for name, scene, count, camera_at in [
        ("animated", g.dolly_scene("box"), 10, animated_camera),
        ("turntable", static, 40, turntable_camera),
        # a still box filmed from a camera riding a moving node, like the
        # cameras on VirtualCity's cars
        ("riding", g.dolly_scene("dolly"), 10, animated_camera),
    ]:
        camera = scene.camera_transform.copy()
        video = scene.render(file_type="mkv", resolution=resolution, fps=fps)
        with g.TemporaryDirectory() as directory:
            path = g.os.path.join(directory, "video.mkv")
            with open(path, "wb") as video_file:
                video_file.write(video)
            frames = g.decode_video(path, resolution)
        checks[f"{name} frame count"] = len(frames) == count
        checks[f"{name} camera restored"] = g.np.allclose(scene.camera_transform, camera)
        checks[f"{name} moves"] = not g.np.allclose(frames[0], frames[-1])

        # render the same frames directly to compare against the video,
        # with the floor and ambient occlusion every render has
        viewer = g.offscreen_viewer(scene, resolution=resolution, studio=True)
        for index in [0, count // 2, count - 1]:
            viewer.camera_transform = camera_at(scene, camera, index, count)
            expected = viewer.to_array().color[..., :3].astype(g.np.float64)
            error = g.np.abs(frames[index] - expected).mean()
            checks[f"{name} frame {index} matches"] = error < 4.0
        viewer.close()

    # the advertised one-liner: a model on the web with its textures
    # beside it, written as a turntable video
    with g.serve_meshes() as address, g.TemporaryDirectory() as directory:
        path = g.os.path.join(directory, "film.mp4")
        returned = g.trimesh.load_remote(f"{address}/fuze.obj").render(
            path, resolution=resolution, fps=fps
        )
        film = g.decode_video(path, resolution)
        with open(path, "rb") as written:
            checks["one-liner writes the path and returns nothing"] = (
                returned is None and len(written.read()) > 0
            )
    checks["one-liner turntable"] = len(film) == 40 and film.std(axis=(1, 2, 3)).min() > 1

    for name, ok in checks.items():
        assert ok, name


def test_interaction():
    """Speed, idle, settings, fixed default lights, and dragging then `z` resetting."""
    from scipy import ndimage

    sphere = g.trimesh.creation.icosphere(subdivisions=4)
    scene = pbr_mesh(sphere, [180, 180, 180, 255], 0.0, 0.4).scene()
    initial = scene.camera_transform.copy()
    viewer = g.offscreen_viewer(scene, resolution=[200, 150])
    # pyglet's window module needs a context to exist before importing
    from pyglet.window import key, mouse

    full = viewer.to_array().color
    viewer.started -= 10.0
    # a few fast draws decide nothing
    viewer.draw_seconds = [0.01] * 200
    viewer.state.dirty = True
    viewer.on_draw()
    checks = {"fast machine stays full quality": not viewer.state.fast}
    # a second of draws at 5 frames a second
    viewer.draw_seconds = [0.2] * 10
    viewer.state.dirty = True
    viewer.on_draw()
    checks["slow machine draws fast"] = viewer.state.fast
    # fast drawing skips the floor's shadow and occlusion
    studio = g.offscreen_viewer(scene, resolution=[200, 150], studio=True)
    studio_full = studio.to_array().color
    studio.toggle("fast")
    checks["fast looks different"] = (studio_full != studio.to_array().color).any()
    studio.close()
    # the user turns quality back on and it stays on
    viewer.toggle("fast")
    viewer.draw_seconds = [0.2] * 10
    viewer.state.dirty = True
    viewer.on_draw()
    checks["user choice sticks"] = not viewer.state.fast
    # a smooth sphere can round one pixel differently between draws
    checks["full quality again"] = (
        g.np.abs(viewer.to_array().color.astype(int) - full).max() <= 1
    )

    # nothing changed so nothing is drawn
    count = viewer.frame_count
    for _ in range(10):
        viewer.on_draw()
    checks["idle draws nothing"] = viewer.frame_count == count
    lights = viewer.light_transforms.copy()
    viewer.on_mouse_press(100, 75, mouse.LEFT, 0)
    viewer.on_mouse_drag(160, 95, 60, 20, mouse.LEFT, 0)
    checks["drag moves camera"] = not g.np.allclose(viewer.camera_transform, initial)
    dragged = viewer.camera_transform.copy()
    image = viewer.to_array().color[..., :3].astype(g.np.float64)
    started = (
        [*viewer.lights, viewer.environment],
        g.np.concatenate((viewer.light_transforms, [viewer.environment_transform])),
    )
    viewer.on_key_press(symbol=key.Z, modifiers=0)
    count = viewer.frame_count
    viewer.on_draw()
    checks["reset camera"] = g.np.allclose(viewer.camera_transform, initial)
    # dragging moved the viewer's camera and not the scene's
    checks["scene camera untouched"] = g.np.allclose(scene.camera_transform, initial)
    checks["reset redraws"] = viewer.frame_count == count + 1
    checks["reset image"] = (
        g.np.abs(viewer.to_array().color.astype(int) - full).max() <= 1
    )
    viewer.close()

    # the default lights are fixed in the world where the viewer started,
    # so moving the camera shows the same lighting from somewhere else
    scene.camera_transform = dragged
    fixed = shade_reference(scene, lights=started)
    moved = shade_reference(scene)
    interior = ndimage.binary_erosion(~g.np.isnan(fixed[..., 0]), iterations=2)
    checks["lights stay put"] = g.np.allclose(started[1][:-1], lights)
    checks["lit from where it started"] = g.np.abs(image - fixed)[interior].max() <= 2.0
    checks["not lit from the new camera"] = g.np.abs(image - moved)[interior].max() > 10.0

    # settings are `ViewerState` fields passed as keywords or flags, and a
    # misspelled one raises rather than being quietly ignored
    viewer = g.offscreen_viewer(scene, smooth=False, flags={"wireframe": True})
    checks["settings"] = not viewer.state.smooth and viewer.state.wireframe
    viewer.close()
    checks["misspelled setting raises"] = False
    try:
        g.offscreen_viewer(scene, wirefram=True)
    except TypeError:
        checks["misspelled setting raises"] = True

    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_conversion()
    test_viewer()
    test_methods()
    test_interaction()
