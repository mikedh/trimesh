try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.lighting import DirectionalLight, PointLight, SpotLight
from trimesh.scene.metadata import Environment, RenderSettings, Softbox
from trimesh.viewer.render import render_frames, turntable

tf = g.trimesh.transformations
# a straight line segment along X through the origin
SEGMENT = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
# an environment with no light at all
DARK = Environment()
# a sun at an angle, so a box's shadow falls beside it
SUN_POSE = tf.euler_matrix(0.5, -0.4, 0.0)


def test_all_exists():
    # every name in `__all__` should be an attribute — a stale
    # entry breaks `from trimesh.viewer import *`
    from trimesh import viewer

    assert all(hasattr(viewer, name) for name in viewer.__all__)


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
    return Environment(sky=color, ground=color)


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
    """A scene which shows something renders the same after a glTF round trip."""
    exported = g.trimesh.util.wrap_as_stream(scene.export(file_type="glb"))
    reloaded = g.trimesh.load_scene(exported, file_type="glb")
    before, after = (
        g.png(current.save_image(resolution=scene.camera.resolution)).astype(int)
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
    from trimesh.viewer.conversion import ALPHA_MODES, geometry_to_arrays

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
    voxels = g.trimesh.voxel.VoxelGrid(g.np.ones((2, 2, 2), dtype=bool))
    box_arrays, sphere_arrays, fuze_arrays = (
        geometry_to_arrays(geometry) for geometry in [box, sphere, fuze]
    )
    wrench_arrays, point_arrays, painted_arrays = (
        geometry_to_arrays(geometry) for geometry in [wrench, points, painted]
    )
    blend_modes = {
        geometry_to_arrays(geometry).material.alpha_mode
        for geometry in blend.geometry.values()
    }
    # the expected number of segments from discretizing every entity
    segment_count = sum(
        len(entity.discrete(wrench.vertices, scale=wrench.scale)) - 1
        for entity in wrench.entities
    )

    checks = {
        "box is indexed triangles": box_arrays.faces.shape == (12, 3),
        "box face colors in order": (
            box_arrays.face_colors == box.visual.face_colors
        ).all(),
        "box triangles unchanged": g.np.allclose(
            box_arrays.vertices[box_arrays.faces], box.vertices[box.faces]
        ),
        "sphere vertex colors carried": (
            sphere_arrays.vertex_colors[sphere_arrays.faces]
            == sphere.visual.vertex_colors[sphere.faces]
        ).all(),
        "translucent vertex colors blend": sphere_arrays.material.alpha_mode
        == ALPHA_MODES["BLEND"],
        "opaque box": box_arrays.material.alpha_mode == ALPHA_MODES["OPAQUE"],
        "fuze uv carried": g.np.allclose(
            fuze_arrays.uv[fuze_arrays.faces], fuze.visual.uv[fuze.faces]
        ),
        "fuze has a texture": fuze_arrays.material.base_color_texture is not None,
        "obj materials are not metal": fuze_arrays.material.metallic == 0.0,
        "material keeps vertex colors": (
            painted_arrays.vertex_colors == [200, 150, 40, 255]
        ).all()
        and painted_arrays.material.metallic == 0.5,
        "every glTF alpha mode": blend_modes == set(ALPHA_MODES.values()),
        "wrench segment count": wrench_arrays.vertices.shape == (segment_count, 2, 3),
        "wrench colors per segment": (
            wrench_arrays.vertex_colors == [10, 20, 30, 255]
        ).all(),
        "points are zero length segments": g.np.allclose(
            point_arrays.vertices[:, 0], point_arrays.vertices[:, 1]
        ),
        "points kind": point_arrays.kind == "points",
        "voxels become boxes": geometry_to_arrays(voxels).kind == "mesh",
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
struct Softbox {
    vec4 direction;
    vec4 color;
};
layout(std140, row_major) uniform Frame {
    mat4 projection;
    mat4 inverse_projection;
    vec4 viewport_size;
    ivec4 write_depth_and_index;
    ivec4 light_count;
    Light lights[8];
    mat4 shadow_matrices[4];
    vec4 sky_color;
    vec4 ground_color;
    vec4 up_direction;
    ivec4 softbox_count;
    Softbox softboxes[4];
    vec4 ground_light;
    vec4 occlusion_radius;
} frame;
"""
    checks = {
        "mesh inputs": inputs.strip() in shaders.shader_source("mesh.vert"),
        "frame block": block.strip() in shaders.shader_source("mesh.frag"),
    }
    # a frame field std140 would pad differently than numpy is refused
    frame = shaders.FRAME
    for name, dtype in {
        "vec3": [("padded", "<f4", 3)],
        "offset": [("scalar", "<f4"), ("vector", "<f4", 4)],
        "mat3": [("matrix", "<f4", (3, 3))],
    }.items():
        shaders.FRAME = g.np.dtype(dtype)
        shaders._SHADER_SOURCE.clear()
        checks[f"std140 refuses {name}"] = False
        try:
            shaders.shader_source("mesh.frag")
        except ValueError:
            checks[f"std140 refuses {name}"] = True
        finally:
            shaders.FRAME = frame
            shaders._SHADER_SOURCE.clear()

    # the filled shaders pass the reference compiler when it exists
    validator = shutil.which("glslangValidator")
    for file_name in g.os.listdir(shaders.DIRECTORY):
        if not file_name.endswith((".vert.glsl", ".frag.glsl")):
            continue
        name = file_name.removesuffix(".glsl")
        source = shaders.shader_source(name)
        checks[f"{name} fully filled"] = "$" not in source
        if validator is not None:
            command = [validator, "--stdin", "-S", name.split(".")[-1]]
            result = subprocess.run(command, input=source.encode(), capture_output=True)
            checks[f"{name} validates"] = result.returncode == 0

    g.require_rendering()
    from trimesh.viewer.opengl import gl

    viewer = g.offscreen_viewer(g.trimesh.creation.box().scene())
    viewer.to_array()
    context = viewer.context
    for program, dtypes in [
        (context.mesh_program.program, (shaders.VERTEX, shaders.SKIN, shaders.INSTANCE)),
        (context.line_program.program, (shaders.SEGMENT,)),
        # full screen passes have no inputs but read the frame
        (context.occlusion_program.program, ()),
        (context.blur_program.program, ()),
    ]:
        # every input is where `shaders.attribute_locations` put it
        for name, location in shaders.attribute_locations(*dtypes).items():
            found = gl.glGetAttribLocation(
                program, ctypes.create_string_buffer(name.encode())
            )
            checks[f"{program} {name} location"] = found == location

        # the block is the size of the dtype and every member is where numpy put it
        block = gl.glGetUniformBlockIndex(program, ctypes.create_string_buffer(b"Frame"))
        size = gl.GLint()
        gl.glGetActiveUniformBlockiv(
            program, block, gl.GL_UNIFORM_BLOCK_DATA_SIZE, ctypes.byref(size)
        )
        checks[f"{program} frame size"] = size.value == shaders.FRAME.itemsize
        for name, offset in block_offsets("Frame", shaders.FRAME).items():
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
    scene = g.pbr_mesh(mesh, **options).scene()
    # square pixels, which is the only camera glTF can describe: a
    # wider image keeps the vertical field of view and sees more
    scene.camera = g.trimesh.scene.Camera(
        name="camera", resolution=[150, 150], fov=(40, 40)
    )
    scene.camera.resolution = (200, 150)
    scene.camera_transform = scene.camera.look_at(**view)
    scene.lights = list(lights)
    for light, pose in zip(lights, poses):
        scene.graph[light.name] = pose
    return g.set_render(scene, environment=environment, ground_shadow=False)


def lighting_cases():
    """Scenes lit by every kind of light and environment with a fixed camera."""
    down = tf.translation_matrix([0, 0, 2.0])
    under = tf.translation_matrix([0, 0, -2.0])
    # a shadow casting light shining up from below at an angle
    backlight = DirectionalLight(name="backlight", intensity=1.0)
    upward = tf.euler_matrix(g.np.pi - 0.3, 0.2, 0.0)
    sun = DirectionalLight(name="sun", color=[255, 200, 150, 255], intensity=3.0)
    bulb = PointLight(name="bulb", intensity=6.0)
    ranged = PointLight(name="bulb", intensity=6.0, radius=3.0)
    spot = SpotLight(name="spot", intensity=20.0, innerConeAngle=0.2, outerConeAngle=0.5)
    lamp = PointLight(name="lamp", intensity=40.0)
    # bright, as plastic only reflects 4% of light head on
    softbox = Softbox(direction=[0.6, 0.6, 0.5], sharpness=30.0, color=[40.0] * 3)
    window = Environment(sky=[0.05] * 3, ground=[0.05] * 3, softboxes=[softbox])
    gradient = Environment(sky=[0.9, 0.8, 0.7], ground=[0.05, 0.05, 0.1])
    # twice a unit vector, which every engine reads as a direction
    near = Softbox(direction=[1.4, 1.0, 1.0], sharpness=20.0, color=[4] * 3)
    reflected = Environment(
        sky=[0.8, 0.7, 0.6], ground=[0.05, 0.08, 0.1], softboxes=[near]
    )
    below = Environment(sky=[0.02] * 3, ground=[0.3, 0.2, 0.4])
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
        cases[name] = g.pbr_mesh(sphere, color, 0.0, 0.5).scene()
        cases[name].camera.resolution = (200, 150)
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
    checks = {"default lights": len(cases["autolight"].lights) == 2}
    for name, scene in cases.items():
        image = g.rgb(scene, resolution=scene.camera.resolution)
        reference = g.shade_reference(scene)
        # stay away from the antialiased silhouette
        interior = ndimage.binary_erosion(~g.np.isnan(reference[..., 0]), iterations=2)
        radius = g.np.linalg.norm(floor_points(scene)[interior], axis=-1)
        checks[f"{name} matches reference"] = (
            g.np.abs(image - reference)[interior].max() <= 2.0
        )
        checks[f"{name} is not flat"] = g.np.ptp(reference[interior]) > 40
        checks[f"{name} silhouette"] = all(g.silhouette_checks(scene, image).values())
        for check, predicate in extra.get(name, {}).items():
            checks[f"{name} {check}"] = predicate(image[interior], radius)

    # everything a render uses is scene content which survives glTF, but
    # no lights at all can't be stored and loads as the default lights
    lit = {name: scene for name, scene in cases.items() if len(scene.lights) > 0}
    for name, scene in lit.items():
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
            Environment(up=[0, 1, 0], sky=[0.8] * 3, ground=[0.0] * 3),
            lambda image: image[0, 0].mean() > image[-1, 0].mean() + 40,
        ),
    }.items():
        scene = g.set_render(
            g.trimesh.creation.box(extents=[0.1, 0.1, 0.1]).scene(),
            environment=environment,
            ground_shadow=False,
            show_environment=True,
        )
        checks[name] = predicate(g.rgb(scene, resolution=[80, 60]))

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
    scene.camera.resolution = (160, 120)
    scene.camera_transform = scene.camera.look_at(
        vertices, rotation=tf.euler_matrix(0.3, 0.2, 0.0)
    )
    scene.lights = [DirectionalLight(name="sun", intensity=3.0)]
    scene.graph["sun"] = tf.euler_matrix(0.0, 1.0, 0.0)
    g.set_render(scene, environment=DARK, ground_shadow=False)
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
            g.pbr_mesh(square, [200, 200, 200, 255], 0.0, 0.6, uv=uv)
            square.visual.material.normalTexture = constant(encoded)
            image = g.rgb(scene, resolution=scene.camera.resolution)
            # the reference reads the decoded normal like the shader does
            reference = g.shade_reference(scene, normal=encoded / 255.0 * 2.0 - 1.0)
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
        sphere = g.pbr_mesh(
            g.trimesh.creation.icosphere(subdivisions=3), [200, 120, 60, 255], 0.0, 0.5
        )
        sphere.visual.material.occlusionTexture = occlusion
        scene = sphere.scene()
        scene.lights = [DirectionalLight(name="sun", intensity=2.0)]
        scene.graph["sun"] = tf.euler_matrix(-0.5, 0.5, 0.0)
        g.set_render(scene, environment=ambient([light] * 3), ground_shadow=False)
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
    box = g.pbr_mesh(box.apply_translation([0, 0, 1.5]), [90, 160, 220, 255], 0, 0.5)
    scene.add_geometry(box)
    return scene


def test_shadows():
    """Lights shadow exactly where rays to them are blocked, leaving only ambient there."""
    from scipy import ndimage

    above = tf.translation_matrix([0.5, 0.0, 5.0])
    spot = SpotLight(name="spot", intensity=60.0, outerConeAngle=0.7)
    checks = {}
    # name, light, pose, which pose column is its position, casts shadows
    for name, light, pose, axis, casts in [
        # from infinitely far away at an angle so the shadow is beside the box
        ("directional", DirectionalLight(name="sun", intensity=3.0), SUN_POSE, 2, True),
        # down from a point, so the shadow is bigger than the box
        ("spot", spot, above, 3, True),
        ("point", PointLight(name="bulb", intensity=40.0), above, 3, False),
    ]:
        scene = box_shadow(light, pose)
        image = g.rgb(scene, resolution=scene.camera.resolution)
        unshadowed = g.shade_reference(scene)
        scene.lights = []
        ambient_only = g.rgb(scene, resolution=scene.camera.resolution)
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

    for name, ok in checks.items():
        assert ok, name


def test_depth_segmentation():
    """Depth and segmentation are what each camera ray hits, for any lens and scale."""
    from scipy import ndimage

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
        scene.camera.resolution = (160, 120)
        scene.camera_transform = scene.camera.look_at(
            scene.bounds, rotation=tf.euler_matrix(0.4, 0.3, 0.0)
        )
        scene.camera.magnification = magnification(scene.scale)
        color, depth, index = g.render(
            scene, include_depth=True, include_segmentation=True, resolution=[160, 120]
        )
        reference_depth, reference = g.camera_ray_reference(scene)
        # a camera's planes are lengths which scale with the scene
        rescaled = scene.scaled(7.0).camera
        checks[f"{name} {lens} {scale} planes scale"] = g.np.allclose(
            [rescaled.z_near, rescaled.z_far],
            [scene.camera.z_near * 7.0, scene.camera.z_far * 7.0],
        )
        # pixels whose 3x3 neighborhood all hit the same thing
        interior = ndimage.maximum_filter(reference, 3) == ndimage.minimum_filter(
            reference, 3
        )
        covered = interior & (reference >= 0)
        label = f"{name} {lens} {scale}"
        checks[f"{label} shapes"] = depth.shape == index.shape == (120, 160)
        checks[f"{label} dtypes"] = (
            depth.dtype == g.np.float32 and index.dtype == g.np.int64
        )
        checks[f"{label} segmentation is rays"] = (index == reference)[interior].all()
        checks[f"{label} depth is rays"] = g.np.allclose(
            depth[covered], reference_depth[covered], rtol=1e-4
        )
        checks[f"{label} background depth zero"] = (depth[index < 0] == 0.0).all()
        checks[f"{label} foreground depth positive"] = (depth[index >= 0] > 0.0).all()
        checks[f"{label} not vacuous"] = (
            covered.sum() > 0.5 * (reference >= 0).sum() > 500
        )
        checks[f"{label} color silhouette"] = all(
            g.silhouette_checks(scene, color).values()
        )
        # including every node sharing one geometry
        checks[f"{label} every node has an id"] = set(
            range(len(scene.graph.nodes_geometry))
        ) <= set(g.np.unique(index))

    # pixels are read one way: each part only when asked for, and the
    # depth and segmentation both come from one more draw
    viewer = g.offscreen_viewer(instanced(), resolution=[80, 60])
    draws = []
    draw = viewer.draw
    viewer.draw = lambda *args: draws.append(args) or draw(*args)
    color_only = viewer.to_array()
    single = len(draws)
    everything = viewer.to_array(include_depth=True, include_segmentation=True)
    both = len(draws) - single
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
    checks["one more draw for both"] = single == 1 and both == 2
    checks["hide one instance"] = (
        nodes.index("small_0") not in seen
        and {nodes.index("small_1"), nodes.index("small_2")} <= seen
    )

    # an orthographic camera sees the same image from any distance
    scene = nested()
    scene.camera.resolution = (160, 120)
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
    scene.camera.resolution = (160, 120)
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


def studio_of(scene):
    """The default lights and environment a viewer of `scene` starts with."""
    settings = RenderSettings.from_metadata(scene.metadata)
    return settings.resolve(scene, scene.camera_transform)


def box_on_floor(sun=None, ground=False, light=1.0):
    """A white box on a white or the invisible `ground` floor, seen from above."""
    white = [235, 235, 235, 255]
    scene = g.trimesh.Scene()
    box = g.pbr_mesh(g.trimesh.creation.box(extents=[2.0, 2.0, 1.0]), white, 0.0, 0.8)
    scene.add_geometry(box, node_name="box", transform=tf.translation_matrix([0, 0, 0.5]))
    if not ground:
        floor = g.pbr_mesh(g.trimesh.creation.box(extents=[12, 12, 0.1]), white, 0, 0.8)
        scene.add_geometry(floor, transform=tf.translation_matrix([0, 0, -0.05]))
    scene.lights = []
    if sun is not None:
        scene.lights = [DirectionalLight(name="sun", intensity=2.0)]
        scene.graph["sun"] = sun
    scene.camera = g.trimesh.scene.Camera(resolution=[240, 180], fov=(50, 50))
    # looking straight down at the floor, which is the case where the
    # studio can't use the camera's backwards direction for its frame
    scene.camera_transform = tf.rotation_matrix(1.4, [0, 0, 1]) @ tf.translation_matrix(
        [0, 0, 9.0]
    )
    environment = Environment(up=[0, 0, 1], sky=[light] * 3, ground=[light] * 3)
    return g.set_render(scene, environment=environment, ground_shadow=ground)


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
        g.rgb(scene, resolution=[240, 180], flags={"ambient_occlusion": enabled})
        for enabled in [False, True]
    )
    darkening = off.mean(axis=-1) - on.mean(axis=-1)
    reference = occlusion_reference(scene, studio_of(scene).occlusion_radius)
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
        "depth untouched": g.np.allclose(
            on.depth, g.camera_ray_reference(sunny)[0], rtol=1e-3
        ),
    }

    # the invisible floor under a sun
    scene = box_on_floor(sun=SUN_POSE, ground=True, light=0.3)
    options = {"studio": True, "flags": {"ambient_occlusion": False}}
    color, depth, index = g.render(
        scene,
        include_depth=True,
        include_segmentation=True,
        resolution=[240, 180],
        **options,
    )
    studio = studio_of(scene)
    image = color[..., :3].astype(g.np.float64)
    box_depth, box_index = g.camera_ray_reference(scene)
    floor, blocked = floor_shadow(scene, SUN_POSE, 2, height=studio.ground[:, 2].mean())
    shaded = floor & (image.mean(axis=-1) < 250)
    umbra = ndimage.binary_erosion(blocked, iterations=3)
    lit = ndimage.binary_erosion(floor & ~blocked, iterations=3)
    # the floor loses the sun and keeps the environment
    umbra_value = 255.0 * (0.3 / studio.ground_light)
    checks.update(
        {
            "has a shadow": umbra.sum() > 300,
            "shadow where rays are blocked": g.iou(shaded, blocked) > 0.95,
            "umbra loses the sun": g.np.abs(
                image[umbra].mean(axis=-1) - umbra_value
            ).max()
            <= 3,
            "lit floor invisible": g.np.abs(image[lit] - 255.0).max() <= 1,
            "not in depth": g.np.allclose(depth, box_depth, rtol=1e-3),
            "not in segmentation": (index == box_index).mean() > 0.99,
            "not in bounds": g.np.allclose(scene.bounds, [[-1, -1, 0], [1, 1, 1]]),
            "not dumped": len(scene.dump()) == 1,
        }
    )
    # from below the floor can't be seen at all
    scene.camera_transform = (
        tf.rotation_matrix(g.np.pi, [1, 0, 0]) @ scene.camera_transform
    )
    with_ground, without_ground = (
        g.rgb(g.set_render(scene, ground_shadow=ground), resolution=[240, 180], **options)
        for ground in [True, False]
    )
    checks["invisible from below"] = g.np.abs(with_ground - without_ground).max() <= 1

    for name, ok in checks.items():
        assert ok, name


def test_export_unchanged():
    """Viewing and rendering never change a scene or the bytes it exports to."""
    from trimesh.interfaces.ffmpeg import VideoSettings
    from trimesh.viewer.windowed import SceneViewer

    g.require_rendering()
    scenes = {"box": g.trimesh.creation.box().scene()}
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
        state = (scene.has_camera, scene.has_lights, g.copy.deepcopy(scene.metadata))

        scene.save_image(resolution=[64, 48])
        viewer = SceneViewer(scene, visible=False, start_loop=False, resolution=[64, 48])
        viewer.to_array(include_depth=True)
        viewer.close()
        video = VideoSettings(fps=10)
        scene.export(file_type="gif", resolution=[32, 24], duration=0.2, settings=video)

        checks[f"{name} export unchanged"] = scene.export(file_type="glb") == before
        checks[f"{name} state unchanged"] = (
            scene.has_camera,
            scene.has_lights,
            scene.metadata,
        ) == state

    for name, ok in checks.items():
        assert ok, name


def test_viewer():
    """Renders match camera rays every time, sharing a window and freeing the GPU."""
    g.require_rendering()
    from trimesh.viewer.opengl import gl

    mesh = g.trimesh.creation.box(extents=[3.0, 1.0, 2.0])
    mesh.apply_transform(tf.euler_matrix(0.3, 0.7, 0.2))
    scene = mesh.scene()

    checks = {}
    windows = set()
    # alternate resolutions including odd ones
    for index, resolution in enumerate([(320, 240), (97, 61), (1024, 768)] * 2):
        viewer = g.offscreen_viewer(scene, resolution=resolution)
        for render in range(3):
            image = viewer.to_array().color
            label = f"{resolution} pass {index} render {render}"
            checks[f"{label} shape"] = image.shape == (*resolution[::-1], 4)
            checks.update(g.silhouette_checks(scene, image, prefix=f"{label} "))
        windows.add(id(viewer.window))
        uploaded = list(viewer.geometry_buffers.values())
        framebuffers = [viewer.framebuffer, viewer.resolve_framebuffer]
        viewer.close()
    checks["no OpenGL errors"] = gl.glGetError() == 0
    checks["one hidden window"] = len(windows) == 1
    checks["framebuffers freed"] = not any(
        gl.glIsFramebuffer(framebuffer.framebuffer) for framebuffer in framebuffers
    )
    checks["vertex arrays freed"] = not any(
        gl.glIsVertexArray(buffers.vertex_array) for buffers in uploaded
    )
    checks["buffers freed"] = not any(
        gl.glIsBuffer(buffer) for buffers in uploaded for buffer in buffers.buffers
    )
    checks["textures freed"] = not any(
        gl.glIsTexture(texture)
        for buffers in uploaded
        for texture in buffers.textures.values()
    )

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
    checks.update(
        g.silhouette_checks(flat, g.render(flat).color, prefix="zero scale node ")
    )

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

    for width in [1, 4, 9]:
        # set for one render, or stored in the scene for every engine
        segment = framed(g.trimesh.load_path(SEGMENT))
        for source, image in {
            "keyword": g.render(segment, line_width=width).color,
            "scene": g.render(g.set_render(segment, line_width=width)).color,
        }.items():
            # black on white so darkness is coverage: summed down every
            # column across the middle of the line it is exactly the width
            coverage = (255 - image[:, 110:210, 0].astype(g.np.float64)).sum(axis=0) / 255
            checks[f"line width {width} by {source}"] = (
                g.np.abs(coverage - width) < 0.25
            ).all()

    points = g.trimesh.PointCloud(
        [[-1, -1, 0], [1, 0.5, 0], [0, 0, 0]], colors=[255, 0, 0, 255]
    )
    scene = framed(points)
    image = g.render(scene, point_size=8).color
    width, height = scene.camera.resolution
    ndc = scene.camera_project(points.vertices)
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
    after = viewer.to_array().color
    checks["update faces"] = (after != before).any()
    checks.update(g.silhouette_checks(viewer.scene, after, prefix="after update "))
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
    class Viewer(g.trimesh.viewer.windowed.SceneViewer):
        pass

    subclass = Viewer(g.trimesh.creation.box().scene(), visible=False, start_loop=False)
    checks["subclass"] = subclass.to_array().color.ndim == 3
    subclass.close()

    for name, ok in checks.items():
        assert ok, name


def test_video():
    """A video export has every frame of the animation, or a turntable, as rendered."""
    from trimesh.interfaces import ffmpeg
    from trimesh.scene.animation import Animation

    if not ffmpeg.exists:
        if g.include_rendering:
            raise ValueError("rendering video requires ffmpeg!")
        g.log.warning("skipping video: no ffmpeg")
        return
    g.require_rendering()
    resolution = (160, 120)
    settings = ffmpeg.VideoSettings(fps=10)

    def animated_camera(scene, camera, index, count):
        """The scene's camera at frame `index`, which may ride an animated node."""
        scene.animate(index / settings.fps)
        return scene.camera_transform

    def turntable_camera(scene, camera, index, count):
        """The camera orbiting the world axis nearest its up, keeping the horizon level."""
        up = g.np.abs(camera[:3, 1]).argmax()
        axis = g.np.eye(3)[up] * g.np.sign(camera[up, 1])
        return (
            tf.rotation_matrix(2 * g.np.pi * index / count, axis, scene.centroid) @ camera
        )

    def sliding(node):
        """A red box seen by a camera on a dolly, with `node` sliding along X for a second."""
        scene = g.trimesh.Scene()
        scene.add_geometry(colored_box([255, 0, 0, 255]), node_name="box")
        scene.graph.update(frame_to="dolly", matrix=g.np.eye(4))
        scene.camera = g.trimesh.scene.Camera(name="lens", fov=(60, 45))
        lens = tf.translation_matrix([0, 0, 6])
        scene.graph.update(frame_from="dolly", frame_to="lens", matrix=lens)
        times = g.np.linspace(0.0, 1.0, 5)
        slide = g.np.tile(g.np.eye(4), (len(times), 1, 1))
        slide[:, 0, 3] = times
        scene.animations.append(
            Animation(edges=[(None, node)], times=times, matrices=slide)
        )
        return scene

    # a tilted camera
    static = g.trimesh.creation.box(extents=[3, 1, 2]).scene()
    static.camera_transform = tf.euler_matrix(0.5, 0.2, 0.1) @ tf.translation_matrix(
        [0, 0, 6]
    )

    checks = {}
    for name, scene, count, camera_at in [
        ("animated", sliding("box"), 10, animated_camera),
        ("turntable", static, 40, turntable_camera),
        # a still box filmed from a camera riding a moving node, like the
        # cameras on VirtualCity's cars
        ("riding", sliding("dolly"), 10, animated_camera),
    ]:
        camera = scene.camera_transform.copy()
        video = scene.export(file_type="mkv", resolution=resolution, settings=settings)
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
        g.trimesh.load_remote(f"{address}/fuze.obj").export(
            path, resolution=resolution, settings=settings
        )
        film = g.decode_video(path, resolution)
    checks["one-liner turntable"] = len(film) == 40 and film.std(axis=(1, 2, 3)).min() > 1

    for name, ok in checks.items():
        assert ok, name


def test_engines():
    """Blender and OpenGL render the same exported scene alike, and animations have every frame."""
    from PIL import Image, ImageSequence
    from scipy import ndimage

    from trimesh.interfaces import blender

    g.require_rendering()
    scene = g.trimesh.creation.box(extents=[3, 1, 2]).scene()
    scene.camera_transform = tf.euler_matrix(0.4, 0.3, 0.1) @ tf.translation_matrix(
        [0, 0, 6]
    )
    settings = g.trimesh.interfaces.ffmpeg.VideoSettings(fps=5)

    checks = {}
    for file_type in ["gif", "webp"]:
        data = scene.export(
            file_type=file_type, resolution=[64, 48], duration=2.0, settings=settings
        )
        checks[f"{file_type} frames"] = (
            Image.open(g.trimesh.util.wrap_as_stream(data)).n_frames == 10
        )

    if blender.exists:
        engines = ["opengl", "blender"]
        # every frame of an animation agrees, which a turntable turning 120
        # degrees a frame makes very different frame to frame, without the
        # floor whose soft shadows the engines blur differently, and the
        # turn loops in a video twice as long
        floorless = g.set_render(scene.copy(), ground_shadow=False)
        floorless.animations.append(turntable(floorless, 0.6, 5))
        # and the box turns too in a second animation, as every animation
        # plays at once
        box = floorless.graph.nodes_geometry[0]
        times = g.np.arange(4) * 0.2
        spin = tf.rotation_matrix(times * 2.0 * g.np.pi / 0.6, [1, 0, 0])
        floorless.animations.append(
            g.trimesh.scene.animation.Animation(
                [(None, box)], times, spin[:, None], name="spin"
            )
        )
        animation = {"resolution": [80, 60], "duration": 1.2, "settings": settings}
        frames = []
        for engine in engines:
            gif = floorless.export(file_type="gif", engine=engine, **animation)
            sequence = ImageSequence.Iterator(
                Image.open(g.trimesh.util.wrap_as_stream(gif))
            )
            frames.append(
                [(g.np.array(frame.convert("RGB")) < 250).any(-1) for frame in sequence]
            )
        checks["frames agree"] = len(frames[0]) == len(frames[1]) == 6 and all(
            g.iou(first, second) > 0.9 for first, second in zip(*frames)
        )
        for engine, drawn in zip(engines, frames):
            # a GIF merges repeated frames, so a held pose is fewer of them
            checks[f"{engine} loops"] = (
                len(drawn) == 6
                and all(
                    g.iou(drawn[index], drawn[index + 3]) > 0.99 for index in range(3)
                )
                and all(g.iou(drawn[index], drawn[index + 1]) < 0.9 for index in range(2))
            )

        # a saved `.blend` renders the frames Blender renders, here the
        # second time around the loop
        with g.tempfile.TemporaryDirectory() as directory:
            path = g.os.path.join(directory, "scene.blend")
            floorless.export(path, resolution=[80, 60], duration=1.2, fps=5)
            g.subprocess.run(
                [blender._blender_executable, "--background", path]
                + ["--render-output", g.os.path.join(directory, "frame_")]
                + ["--render-frame", "4"],
                check=True,
                capture_output=True,
            )
            reopened = Image.open(g.os.path.join(directory, "frame_0004.png"))
            white = Image.new("RGBA", reopened.size, (255, 255, 255, 255))
            reopened = g.np.asarray(
                Image.alpha_composite(white, reopened.convert("RGBA"))
            ).astype(int)
        direct = list(
            render_frames(
                floorless, resolution=[80, 60], engine="blender", duration=1.2, fps=5
            )
        )[4].astype(int)
        checks["blend renders the same"] = g.np.abs(reopened - direct).max() <= 2

        # an environment seen behind the scene, which both engines draw alike
        wide = Softbox(direction=[0.3, 0.8, 0.5], sharpness=6.0, color=[2] * 3)
        shown = Environment(
            sky=[0.6, 0.5, 0.4], ground=[0.05, 0.08, 0.1], softboxes=[wide]
        )
        # saturated sRGB colors, which glTF stores as linear
        colored = g.trimesh.creation.icosphere(subdivisions=4)
        colored.visual.vertex_colors = g.trimesh.visual.color.interpolate(
            colored.vertices[:, 0], color_map="viridis"
        )
        faced = g.trimesh.creation.box(extents=[2.0, 1.5, 1.0])
        faced.visual.face_colors = g.np.random.default_rng(4).integers(
            0, 255, (len(faced.faces), 3)
        )
        cases = lighting_cases()
        # a camera whose fields of view don't match the image's shape
        # has pixels which aren't square, and both engines keep them
        stretched = scene.copy()
        stretched.camera.fov = [70, 40]
        # and whose far plane cuts the box in half, which both engines clip
        clipped = scene.copy()
        clipped.camera.z_far = 6.0
        # the lights and environment reach Blender through the glTF in the
        # same units, so the shading agrees too and not just shape
        for name, lit in {
            "silhouette": scene,
            "stretched pixels": stretched,
            "far plane": clipped,
            # both engines cast the same shadow from the same light
            "shadowed": box_shadow(DirectionalLight(name="sun", intensity=3.0), SUN_POSE),
            "environment shown": g.set_render(
                lit_scene(environment=shown, metallic=0.5, roughness=0.3),
                show_environment=True,
            ),
            # no material or colors at all, which isn't in the file
            "default color": g.trimesh.creation.icosphere(subdivisions=4).scene(),
            "vertex colors": colored.scene(),
            # a different color on every triangle, which glTF only has
            # on vertices
            "face colors": faced.scene(),
            # clear glass over a colored sphere, which both engines see through
            "glass": g.trimesh.Scene([colored, clear_glass(0.02, 1.5, 0.3)]),
            # the shared cases, where Blender draws a spot light's soft
            # edge with a different curve than glTF, fit as close as it can,
            # and a thin sheet lit through from behind
            **{
                name: cases[name]
                for name in [
                    "directional",
                    "spot",
                    "autolight",
                    "custom environment",
                    "diffuse transmission",
                ]
            },
        }.items():
            images = [
                g.png(lit.export(file_type="png", resolution=[160, 120], engine=engine))[
                    ..., :3
                ].astype(g.np.float64)
                for engine in engines
            ]
            drawn = [(image < 250).any(axis=-1) for image in images]
            inside = ndimage.binary_erosion(drawn[0] & drawn[1], iterations=2)
            difference = g.np.abs(images[0] - images[1])
            # mid tones are where falloffs like a spot cone's soft edge are,
            # which a mean over everything would hide
            gray = g.np.array([image.mean(axis=-1) for image in images])
            middle = ((gray > 30) & (gray < 200)).any(axis=0)
            checks[f"{name} engines agree"] = g.iou(*drawn) > 0.95
            checks[f"{name} engines shade alike"] = difference[inside].mean() < 6.0
            checks[f"{name} engines compared"] = inside.sum() > 2000
            checks[f"{name} engines shade mid tones alike"] = (
                difference[middle].mean() < 25.0
            )

        # points and lines, which Blender draws as spheres and tubes as
        # many pixels across as the viewer, under nodes which scale them
        rng = g.np.random.default_rng(3)
        hues = g.np.column_stack([rng.random(200), g.np.ones((200, 2))])
        cloud = g.trimesh.PointCloud(
            rng.uniform(-1, 1, (200, 3)) * [0.5, 0.33, 0.33],
            colors=g.trimesh.visual.color.hsv_to_rgba(hues),
        )
        angle = g.np.linspace(0, 12, 200)
        helix = g.np.column_stack([g.np.cos(angle), g.np.sin(angle), angle / 6 - 1])
        segments = g.trimesh.Scene()
        segments.add_geometry(cloud, transform=tf.scale_matrix(3))
        segments.add_geometry(
            g.trimesh.load_path(helix / 2), transform=tf.scale_matrix(2)
        )
        segments.camera_transform = scene.camera_transform
        images = [
            g.png(segments.export(file_type="png", resolution=[160, 120], engine=engine))[
                ..., :3
            ].astype(g.np.float64)
            for engine in engines
        ]
        drawn = [(image < 250).any(axis=-1) for image in images]
        checks["segments engines agree"] = g.iou(*drawn) > 0.9
        # Blender renders glass over its floor the same every time
        glassy = g.trimesh.Scene([colored, clear_glass(0.02, 1.5, 0.3)])
        checks["blender repeats exactly"] = g.np.array_equal(
            *[
                g.png(
                    glassy.export(
                        file_type="png", resolution=[160, 120], engine="blender"
                    )
                )
                for _ in range(2)
            ]
        )
        # the spheres are lit the way the viewer lights its points
        means = [image[mask].mean(axis=0) for image, mask in zip(images, drawn)]
        checks["segments engines color alike"] = g.np.abs(means[0] - means[1]).max() < 16
    elif g.all_dependencies:
        raise ValueError("blender is required!")

    for name, ok in checks.items():
        assert ok, name


def test_interaction():
    """Speed, idle, settings, fixed default lights, and dragging then `z` resetting."""
    from scipy import ndimage

    sphere = g.trimesh.creation.icosphere(subdivisions=4)
    scene = g.pbr_mesh(sphere, [180, 180, 180, 255], 0.0, 0.4).scene()
    initial = scene.camera_transform.copy()
    viewer = g.offscreen_viewer(scene, resolution=[200, 150])
    # pyglet's window module needs a context to exist before importing
    from pyglet.window import key, mouse

    full = viewer.to_array().color
    viewer.started -= 10.0
    # a few fast draws decide nothing
    viewer.draw_seconds = [0.01] * 200
    viewer.check_speed()
    checks = {"fast machine stays full quality": not viewer.state.fast}
    # a second of draws at 5 frames a second
    viewer.draw_seconds = [0.2] * 10
    viewer.check_speed()
    checks["slow machine draws fast"] = viewer.state.fast
    checks["fast looks different"] = (full != viewer.to_array().color).any()
    # the user turns quality back on and it stays on
    viewer.toggle("fast")
    viewer.draw_seconds = [0.2] * 10
    viewer.check_speed()
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
    lights = viewer.studio.transforms.copy()
    viewer.on_mouse_press(100, 75, mouse.LEFT, 0)
    viewer.on_mouse_drag(160, 95, 60, 20, mouse.LEFT, 0)
    checks["drag moves camera"] = not g.np.allclose(viewer.camera_transform, initial)
    dragged = viewer.camera_transform.copy()
    image = viewer.to_array().color[..., :3].astype(g.np.float64)
    started = viewer.studio
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
    fixed = g.shade_reference(scene, studio=started)
    moved = g.shade_reference(scene, studio=studio_of(scene))
    interior = ndimage.binary_erosion(~g.np.isnan(fixed[..., 0]), iterations=2)
    checks["lights stay put"] = g.np.allclose(started.transforms, lights)
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
