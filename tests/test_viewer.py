try:
    from . import generic as g
except BaseException:
    import generic as g

import PIL.Image

from trimesh.exchange.gltf.lights import rgbd_pack, rgbd_unpack
from trimesh.scene.lighting import (
    CUBE_FACES,
    HARMONICS,
    DirectionalLight,
    EnvironmentLight,
    PointLight,
    SpotLight,
    prefilter,
    spherical_harmonics,
)
from trimesh.viewer.conversion import (
    OCCLUSION_RADIUS,
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

    assert all(hasattr(viewer, name) for name in viewer.__all__), "every name exists"


def colored_box(color, extents=None):
    box = g.trimesh.creation.box(extents=extents)
    box.visual.face_colors = color
    return box


def ambient(color):
    """A constant environment, which is an ambient light."""
    cubemap = g.np.tile(color, (6, 1, 1, 1))
    return EnvironmentLight(name="environment", cubemap=cubemap)


def cubemap_sample(cubemap, directions):
    """
    The value of a cube map towards directions as OpenGL's linear filtering
    with `GL_TEXTURE_CUBE_MAP_SEAMLESS` blends the four texels around each.
    """
    np = g.np
    size = cubemap.shape[1]
    # the texels of every face with a border one texel past its edges
    middles = (2.0 * np.arange(-1, size + 1) + 1.0 - size) / size
    faces = CUBE_FACES[:, None, None]
    border = (
        faces[..., 0, :]
        + middles[:, None] * faces[..., 1, :]
        + middles[:, None, None] * faces[..., 2, :]
    )
    # the face each of those and the directions are on, then each in the
    # face's frame as the row and column on it
    flat = np.vstack((border.reshape((-1, 3)), directions.reshape((-1, 3))))
    axis = np.abs(flat).argmax(axis=1)
    face = 2 * axis + (flat[np.arange(len(flat)), axis] < 0.0)
    local = np.matmul(CUBE_FACES[face], flat[:, :, None])[:, :, 0]
    coordinates = (local[:, [2, 1]] / local[:, :1] + 1.0) * size / 2.0 - 0.5
    # the border is the nearest texel on the neighboring face
    count = len(border.reshape((-1, 3)))
    nearest = np.clip(np.round(coordinates[:count]), 0, size - 1).astype(np.int64)
    padded = cubemap[face[:count], nearest[:, 0], nearest[:, 1]]
    padded = padded.reshape((6, size + 2, size + 2, -1))
    coordinates, face = coordinates[count:] + 1.0, face[count:]
    low = np.floor(coordinates).astype(np.int64)
    fraction = (coordinates - low)[:, None]
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    index = low[:, None] + corners
    weights = np.prod(np.where(corners, fraction, 1.0 - fraction), axis=2)
    texels = padded[face[:, None], index[..., 0], index[..., 1]]
    samples = (texels * weights[..., None]).sum(axis=1)
    return samples.reshape((*directions.shape[:-1], cubemap.shape[-1]))


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


def assert_round_trip(name, scene):
    """
    Assert a scene renders the same after a glTF round trip, with its
    environment stored as RGBD like the file.

    Parameters
    ------------
    name : str
      Label of the assert.
    scene : trimesh.Scene
      Scene which shows something.
    """
    exported = g.trimesh.util.wrap_as_stream(scene.export(file_type="glb"))
    reloaded = g.trimesh.load_scene(exported, file_type="glb")
    stored = scene.copy()
    if stored.has_lights:
        for light in stored.lights:
            if isinstance(light, EnvironmentLight):
                light.cubemap = rgbd_unpack(rgbd_pack(prefilter(light.cubemap)[0]))
    before, after = (
        g.png(current.render(resolution=scene.camera.resolution)).astype(int)
        for current in (stored, reloaded)
    )
    assert (before[..., :3] < 250).any(), name
    g.assert_images(before, after, name, 2)


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
    tint = environment.intensity
    rotation = g.trimesh.util.unitize(pose[:3, :3].T)
    irradiance = spherical_harmonics(environment.cubemap) * HARMONICS[:, None]
    above, below = (
        np.column_stack(
            (np.ones(len(x)), y, z, x, x * y, y * z, 3 * z**2 - 1, x * z, x**2 - y**2)
        )
        @ irradiance
        * tint
        / np.pi
        for x, y, z in ((normals @ rotation.T).T for normals in (surface, -surface))
    )
    color += diffuse_color * above + transmission_color * below
    reflected = 2.0 * (surface * view).sum(axis=1, keepdims=True) * surface - view
    levels = prefilter(environment.cubemap)
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


def constant_image(color):
    """
    A small image of one color.

    Parameters
    ------------
    color : (3,) int
      RGB bytes.

    Returns
    ------------
    image : PIL.Image.Image
      4 by 4 pixels.
    """
    return PIL.Image.fromarray(g.np.full((4, 4, 3), color, dtype=g.np.uint8))


def case_scene(case):
    """
    The scene of a row of `tests/data/render_cases.json`.

    Parameters
    ------------
    case : dict
      Geometry, materials, lights, environment and camera.

    Returns
    ------------
    scene : trimesh.Scene
      The scene the row describes.
    """
    np = g.np
    scene = g.trimesh.Scene()
    items = case.get("geometry", [{"kind": "sphere"}])
    for item in items:
        uv = None
        if item["kind"] == "sphere":
            mesh = g.trimesh.creation.icosphere(subdivisions=item.get("subdivisions", 4))
        elif item["kind"] == "plane":
            mesh = g.trimesh.creation.box(extents=[10.0, 10.0, 0.1])
            mesh.apply_translation([0, 0, -0.05])
            if item.get("top_only", False):
                # nothing under the top to block light from below
                mesh.update_faces(mesh.face_normals[:, 2] > 0.5)
                mesh.remove_unreferenced_vertices()
        elif item["kind"] == "square":
            vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
            mesh = g.trimesh.Trimesh(vertices, [[0, 1, 2], [0, 2, 3]], process=False)
            uv = (vertices[:, :2] + 1.0) / 2.0
            if item.get("mirror", False):
                # mirrored textures flip the tangent
                uv[:, 0] = 1.0 - uv[:, 0]
        else:
            mesh = g.trimesh.creation.box(extents=item.get("extents", [1.0] * 3))
        mesh.apply_translation(item.get("translation", [0.0] * 3))
        options = {"color": [200, 120, 60, 255], "metallic": 0.0, "roughness": 0.5}
        options.update(item.get("material", {}))
        pbr_mesh(mesh, options["color"], options["metallic"], options["roughness"], uv)
        material = mesh.visual.material
        material.diffuseTransmissionFactor = options.get("diffuse_transmission")
        material.diffuseTransmissionColorFactor = options.get(
            "diffuse_transmission_color"
        )
        material.transmissionFactor = options.get("transmission")
        material.emissiveFactor = options.get("emissive")
        material.alphaMode = options.get("alpha_mode")
        if "transmission_texture" in options:
            material.transmissionTexture = constant_image(options["transmission_texture"])
        if "normal" in options:
            material.normalTexture = constant_image(
                np.round((np.array(options["normal"]) * 0.5 + 0.5) * 255)
            )
        if "occlusion" in options:
            material.occlusionTexture = constant_image([options["occlusion"]] * 3)
        if options.get("unlit", False):
            material.unlit = True
            material.baseColorTexture = constant_image(options["texture"])
            mesh.visual.vertex_attributes["color"] = np.tile(
                options["vertex_color"], (len(mesh.vertices), 1)
            )
        if "visual" in item:
            # trimesh's own colors rather than a material
            mesh.visual = g.trimesh.visual.ColorVisuals(mesh)
            if item["visual"] != "default":
                setattr(mesh.visual, f"{item['visual']}_colors", options["color"])
        scene.add_geometry(mesh)

    view = case.get("camera", "fixed")
    if view == "default":
        scene.camera.fit_resolution((200, 150))
    elif view == "square":
        scene.camera.fit_resolution((160, 120))
        scene.camera_transform = scene.camera.look_at(
            mesh.vertices, rotation=tf.euler_matrix(0.3, 0.2, 0.0)
        )
    else:
        # square pixels, keeping the vertical view when wider
        scene.camera = g.trimesh.scene.Camera(
            name="camera", resolution=[150, 150], fov=(40, 40)
        )
        scene.camera.fit_resolution((200, 150))
        look = {"points": mesh.bounds, "distance": 4.0}
        if items[0]["kind"] == "plane":
            look = {
                "points": [[-5, -5, 0], [5, 5, 0]],
                "rotation": tf.euler_matrix(0.5, 0, 0.2),
            }
        scene.camera_transform = scene.camera.look_at(**look)
    if case.get("studio", False):
        return scene

    classes = {"directional": DirectionalLight, "point": PointLight, "spot": SpotLight}
    lights = []
    for index, item in enumerate(case.get("lights", [])):
        options = {"name": f"light_{index}", "color": item.get("color", WHITE)}
        options.update(intensity=item["intensity"], radius=item.get("radius"))
        if item["type"] == "spot":
            options.update(
                innerConeAngle=item.get("inner", 0.0), outerConeAngle=item["outer"]
            )
        lights.append(classes[item["type"]](**options))
        pose = tf.translation_matrix(item.get("translation", [0, 0, 0]))
        pose = pose @ tf.euler_matrix(*item.get("euler", [0, 0, 0]))
        if item.get("at_camera", False):
            pose = scene.camera_transform
        scene.graph[lights[-1].name] = pose
    scene.lights = lights
    environment = case.get("environment", {"constant": [0.0, 0.0, 0.0]})
    if "constant" in environment:
        light = ambient(environment["constant"])
    else:
        light = g.gradient_environment(
            environment["sky"], environment["ground"], environment.get("softboxes", ())
        )
    return g.set_environment(scene, light)


CASES = {case["name"]: case for case in g.data["render_cases"]}


def assert_shadow(name, scene, image, casts):
    """
    Assert the first light shadows exactly where rays to it are blocked,
    leaving only the environment there.

    Parameters
    ------------
    name : str
      Label of the assert.
    scene : trimesh.Scene
      A plane under something, lit by a light and an environment.
    image : (h, w, 3) float
      Render of `scene`.
    casts : bool
      If the light casts shadows.
    """
    from scipy import ndimage

    unshadowed = shade_reference(scene)
    light, pose = scene.lights[0], scene.graph[scene.lights[0].name][0]
    # directional lights are at infinity along +Z, the rest at their origin
    axis = 3
    if isinstance(light, DirectionalLight):
        axis = 2
    ambient_scene = scene.copy()
    ambient_scene.lights = ambient_scene.lights[-1:]
    ambient_only = rgb(ambient_scene, resolution=scene.camera.resolution)
    floor, blocked = floor_shadow(scene, pose, axis)
    # away from the antialiased edges where the floor meets something
    floor = ndimage.binary_erosion(floor)
    # only where the light visibly lights the plane can it be shadowed
    dark, bright = ambient_only.sum(axis=-1), unshadowed.sum(axis=-1)
    lights_plane = floor & (bright > dark + 30)
    rendered = lights_plane & (image.sum(axis=-1) < 0.5 * (bright + dark))
    blocked &= lights_plane & casts
    umbra = ndimage.binary_erosion(blocked, iterations=2)
    lit = ndimage.binary_erosion(lights_plane & ~blocked, iterations=2)
    assert lit.any() and umbra.any() == casts, name
    assert not (rendered | blocked).any() or g.iou(rendered, blocked) > 0.98, name
    assert not umbra.any() or g.np.abs(image - ambient_only)[umbra].max() <= 2, name
    assert g.np.abs(image - unshadowed)[lit].max() <= 2, name


def test_render_cases():
    """
    Every row of `render_cases.json` renders as its computed truth, fast
    and with quality: the reference has neither shadows nor occlusion so
    is checked fast, shadows with quality, and the rest in both.
    """
    from scipy import ndimage

    np = g.np
    color = g.trimesh.visual.color
    renders = {}
    for (name, case), quality in g.itertools.product(CASES.items(), [False, True]):
        label, checks = f"{name} quality {quality}", case["checks"]
        scene = case_scene(case)
        resolution = scene.camera.resolution
        options = {"resolution": resolution, "flags": {"quality": quality}}
        exposure, background = case.get("exposure", 1.0), case.get("background")
        image = rgb(scene, exposure=exposure, background=background, **options)
        renders[label] = image
        if "reference" in checks and not quality:
            normal = case.get("geometry", [{}])[0].get("material", {}).get("normal")
            if normal is not None:
                # as the shader decodes the bytes of its texture
                encoded = np.round((np.array(normal) * 0.5 + 0.5) * 255)
                normal = encoded / 255.0 * 2.0 - 1.0
            reference = shade_reference(scene, normal=normal)
            interior = ndimage.binary_erosion(~np.isnan(reference[..., 0]), iterations=2)
            assert interior.any(), label
            assert np.abs(image - reference)[interior].max() <= 2.0, label
        if "exposure" in checks:
            # exposure scales all light: every light and glow twice as bright
            exposed = rgb(brighter(scene, 1.0), exposure=2.0, **options)
            g.assert_images(exposed, rgb(brighter(scene, 2.0), **options), label)
        if "shadow" in checks and quality:
            assert_shadow(label, scene, image, case["casts"])
        if quality and scene.has_lights:
            # glTF light nodes' scale isn't part of the light, which 3ds Max sets
            scaled = scene.copy()
            for light in scaled.lights:
                pose = scaled.graph[light.name][0]
                scaled.graph[light.name] = pose @ tf.scale_matrix(40.0)
            scaled = rgb(scaled, exposure=exposure, background=background, **options)
            g.assert_images(scaled, image, label)
        if "unlit" in checks:
            # the final linear color: base color times texture and vertex colors
            material = case["geometry"][0]["material"]
            stored = next(iter(scene.geometry.values())).visual.material
            linear = (
                stored.baseColorFactor[:3]
                / 255.0
                * color.srgb_to_linear(np.divide(material["texture"], 255.0))
                * color.srgb_to_linear(np.divide(material["vertex_color"][:3], 255.0))
            )
            hits = g.camera_hits(scene)
            seen = np.zeros(image.shape[:2], dtype=bool)
            seen[hits.rows, hits.columns] = True
            interior = ndimage.binary_erosion(seen, iterations=2)
            assert interior.any(), label
            expected = color.linear_to_srgb(linear) * 255.0
            assert np.abs(image[interior] - expected).max() <= 1, label
        if "round_trip" in checks and quality:
            assert_round_trip(name, scene)
    # cases which are other cases drawn another way, or scaled by a factor
    # which rounds both images, in both modes
    for (name, case), quality in g.itertools.product(CASES.items(), [False, True]):
        if "same_as" in case:
            label, factor = f"{name} quality {quality}", case.get("factor", 1.0)
            other = renders[f"{case['same_as']} quality {quality}"] * factor
            g.assert_images(renders[label], other, label, 1 + (factor != 1.0))


def test_lighting():
    """The background is the environment and a scaled camera node is glTF's."""
    np = g.np
    # a constant environment behind the scene is one color, lowered by
    # 0.04 by the tone mapper and then sRGB encoded
    expected = (1.055 * (0.3 - 0.04) ** (1 / 2.4) - 0.055) * 255
    box = g.trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    scene = g.set_environment(box.scene(), ambient([0.3] * 3))
    image = rgb(scene, resolution=[80, 60], show_environment=True)
    assert np.abs(image[:3, :3] - expected).max() <= 1
    # and a camera rolled upside down sees the environment upside down
    scene = g.set_environment(box.scene(), g.gradient_environment([0.8] * 3, [0.0] * 3))
    upright = g.render(scene, include_segmentation=True, show_environment=True)
    scene.camera_transform = scene.camera_transform @ tf.rotation_matrix(np.pi, [0, 0, 1])
    rolled = g.render(scene, include_segmentation=True, show_environment=True)
    background = (upright.segmentation < 0) & (rolled.segmentation[::-1, ::-1] < 0)
    flipped = rolled.color[::-1, ::-1].astype(int)
    assert np.abs(upright.color.astype(int) - flipped)[background].max() <= 1

    # a camera node with a scale, which files from 3ds Max carry, sees the
    # same image with its planes in its own units, and the scale moves
    # nothing else: geometry fixed to the camera, or a scroll
    for magnification in [None, np.array([3.0, 2.25])]:
        glossy = case_scene(CASES["custom environment"])
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
        plain = g.offscreen_viewer(glossy, (80, 60), fixed=["badge"])
        glossy.camera.z_near = 0.1 / 40.0
        glossy.camera.z_far = glossy.camera.z_far / 40.0
        if magnification is not None:
            glossy.camera.magnification = magnification / 40.0
        glossy.camera_transform = pose @ tf.scale_matrix(40.0)
        scaled = g.offscreen_viewer(glossy, (80, 60), fixed=["badge"])
        images, scrolled = [], []
        for viewer in (plain, scaled):
            images.append(viewer.to_array().color.astype(np.int64))
            viewer.on_mouse_scroll(0, 0, 0, 1)
            scrolled.append(viewer.to_array().color.astype(np.int64))
        g.assert_images(images[0], images[1], "camera scale")
        assert np.allclose(plain.camera_transform, scaled.camera_transform)
        g.assert_images(scrolled[0], scrolled[1], "scaled scroll")
        assert np.abs(scrolled[0] - images[0]).max() > 1, "scroll changes the image"
        plain.close()
        scaled.close()


def test_shadows():
    """Shadows redraw when casters move."""
    np = g.np
    # viewers redraw shadows and skinned meshes exactly when what casts
    # them moves, though every hidden viewer shares the shadow maps and
    # the joint matrices
    from trimesh.scene.animation import Skin

    scene = case_scene(CASES["shadow directional"])
    box_node = scene.graph.nodes_geometry[-1]
    # a box moved by a bone rather than its own node
    skinned = colored_box([255, 0, 0, 255], extents=[1.0] * 3)
    skinned.vertex_attributes["joints"] = np.zeros((8, 4), dtype=np.int64)
    skinned.vertex_attributes["weights"] = np.tile([1.0, 0, 0, 0], (8, 1))
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
        """
        Draw and read a frame of a scene whose geometry hasn't changed.

        Parameters
        ------------
        viewer : SceneViewer
          Viewer to draw.

        Returns
        ------------
        pixels : (h, w, 4) int
          The resolved frame.
        """
        viewer.draw()
        return viewer.read_pixels("resolved").astype(np.int64)

    still, shifted = frame(viewers[0]), frame(viewers[1])
    assert np.abs(still - shifted).max() > 1, "moving the box moves pixels"
    # the same pixels as before the other viewer drew
    g.assert_images(frame(viewers[0]), still, "shadow and skin kept")
    scene.graph.update(frame_to=box_node, matrix=shift)
    scene.graph.update(frame_to="bone", matrix=tf.translation_matrix([-2.5, 2.0, 0.5]))
    g.assert_images(frame(viewers[0]), shifted, "shadow follows the box")
    for viewer in viewers:
        viewer.close()


def test_rays():
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

    def rotated():
        """A box turned off every axis."""
        mesh = g.trimesh.creation.box(extents=[3.0, 1.0, 2.0])
        return mesh.apply_transform(tf.euler_matrix(0.3, 0.7, 0.2)).scene()

    def flattened():
        """A node scaled to nothing, which animations do, beside a box."""
        scene = rotated()
        scene.add_geometry(g.trimesh.creation.box(), transform=g.np.diag([0, 0, 0, 1.0]))
        return scene

    builders = {
        "instanced": instanced,
        "nested": nested,
        "translucent": translucent,
        "rotated": rotated,
        "flattened": flattened,
        "fuze": lambda: g.trimesh.load_scene(g.os.path.join(g.dir_models, "fuze.obj")),
    }
    # perspective, and orthographic seeing 0.6 of the scene's scale, at an
    # odd size which catches off by one pixel, with depth fit to any scale
    for (name, build), orthographic, scale in g.itertools.product(
        builders.items(), [False, True], [1.0, 1e-3, 1e4]
    ):
        scene = build().scaled(scale)
        scene.camera.fit_resolution((97, 61))
        scene.camera_transform = scene.camera.look_at(
            scene.bounds, rotation=tf.euler_matrix(0.4, 0.3, 0.0)
        )
        if orthographic:
            scene.camera.magnification = (scene.scale * 0.8, scene.scale * 0.6)
        label = f"{name} orthographic {orthographic} at {scale}"
        arrays = g.render(
            scene, include_depth=True, include_segmentation=True, resolution=[97, 61]
        )
        g.assert_rays(scene, arrays, f"{label} ")
        # including every node sharing one geometry
        drawn = [n for n in scene.graph.nodes_geometry if scene.graph[n][0][:3, :3].any()]
        assert set(range(len(drawn))) <= set(g.np.unique(arrays.segmentation)), (
            f"{label} every node has an id"
        )

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
    assert color_only.depth is None and color_only.segmentation is None, (
        "only color unless asked"
    )
    assert depth_only.segmentation is None, "only depth when asked"
    g.assert_images(everything.color, color_only.color, "parts agree")
    assert (everything.depth == depth_only.depth).all(), "depth agrees"
    assert (
        nodes.index("small_0") not in seen
        and {nodes.index("small_1"), nodes.index("small_2")} <= seen
    ), "hide one instance"

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
    g.assert_images(near.color, far.color, "orthographic color is distance free")
    assert g.np.allclose(far.depth[mask] - near.depth[mask], 45.0, rtol=1e-4), (
        "orthographic depth moves with camera"
    )
    # and the scroll wheel zooms the viewer's own camera rather than moving it
    viewer = g.offscreen_viewer(scene, resolution=[160, 120])
    before = viewer.camera_transform.copy()
    viewer.on_mouse_scroll(0, 0, 0, 1)
    assert g.np.allclose(
        viewer.camera.magnification, [2.4 * 0.9, 1.8 * 0.9]
    ) and g.np.allclose(viewer.camera_transform, before), "orthographic scroll zooms"
    assert g.np.allclose(scene.camera.magnification, [2.4, 1.8]), "scene camera untouched"
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
        assert line.sum() > 50, f"{lens} line has pixels"
        assert g.np.allclose(depth[line], 4.0, rtol=1e-4), f"{lens} line depth"


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
    # any frame around each normal will do, as the spiral is around +Z
    helper = g.np.where(g.np.abs(normals[:, :1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])
    tangent = g.trimesh.util.unitize(g.np.cross(normals, helper))
    frame = g.np.stack((tangent, g.np.cross(normals, tangent), normals), axis=1)
    directions = g.np.matmul(local, frame).reshape((-1, 3))
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
    """Quality occlusion darkens only where rays say the sky is hidden; the floor is only shadow."""
    from scipy import ndimage

    # only an environment, so quality adds nothing but occlusion
    scene = box_on_floor()
    off, on = (
        rgb(scene, resolution=[240, 180], flags={"quality": enabled})
        for enabled in [False, True]
    )
    darkening = off.mean(axis=-1) - on.mean(axis=-1)
    reference = occlusion_reference(scene, OCCLUSION_RADIUS * scene.scale)
    covered = ndimage.binary_erosion(~g.np.isnan(reference), iterations=2)
    open_surface = covered & ndimage.binary_erosion(reference == 0.0, iterations=3)
    crease = covered & (reference >= 0.3)
    near = covered & (ndimage.distance_transform_edt(reference < 0.05) < 12)
    assert open_surface.sum() > 2000, "has open surface"
    assert crease.sum() > 100, "has creases"
    assert g.np.abs(darkening[open_surface]).max() <= 3.0, "open surface untouched"
    assert darkening[crease].mean() >= 8.0, "crease darkened"
    assert (darkening[crease] >= 2.0).mean() >= 0.9, "crease mostly darker"
    assert g.np.corrcoef(darkening[near], reference[near])[0, 1] > 0.6, "tracks reference"

    # the invisible floor under a sun
    scene = box_on_floor(sun=SUN_POSE, ground=True, light=0.3)
    options = {"studio": True}
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
    assert umbra.sum() > 300, "has a shadow"
    assert g.iou(shaded, blocked) > 0.95, "shadow where rays are blocked"
    assert g.np.abs(image[umbra].mean(axis=-1) - umbra_value).max() <= 3, (
        "umbra loses the sun"
    )
    assert g.np.abs(image[lit] - 255.0).max() <= 1, "lit floor invisible"
    assert g.np.allclose(scene.bounds, [[-1, -1, 0], [1, 1, 1]]), "not in bounds"
    assert len(scene.dump()) == 1, "not dumped"
    g.assert_rays(scene, arrays, "floor ")
    # from below the floor can't be seen at all
    scene.camera_transform = (
        tf.rotation_matrix(g.np.pi, [1, 0, 0]) @ scene.camera_transform
    )
    with_ground, without_ground = (
        rgb(scene, resolution=[240, 180], ground_shadow=ground, **options)
        for ground in [True, False]
    )
    g.assert_images(with_ground, without_ground, "invisible from below")

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
    assert images["off"][within].min() >= 253, "glowing square is white"
    assert not images["off"][outside][:, :3].any(), "no bloom is the background"
    assert (images["on"] >= images["off"]).all(), "bloom only adds"
    assert g.np.abs(images["on"] - images["off"])[within].max() <= 1, (
        "bloom keeps the square"
    )
    assert all((side > 0).sum() >= BLOOM_SPREAD * 120 for side in falloff), (
        "bloom lights neighbors"
    )
    assert all((g.np.diff(side) <= 0).all() and side[-1] == 0 for side in falloff), (
        "bloom fades with distance"
    )
    g.assert_images(images["off"], images["off again"], "bloom off again is never on")


def test_frames():
    """
    Every frame of a render is exactly a still of the scene posed at its
    time, with lights in the scene so both light it from the same place.
    """
    g.require_rendering()
    fps = 10
    for name in ["box", "dolly"]:
        scene = g.set_environment(g.dolly_scene(name), EnvironmentLight(name="sky"))
        frames = g.render_frames(scene, [64, 48], fps, duration=0.5)
        for index, frame in enumerate(frames):
            posed = scene.copy()
            posed.animate(index / fps)
            still = g.png(posed.render(resolution=[64, 48]))
            g.assert_images(frame, still, f"{name} frame {index} is the posed scene", 2)
        assert g.np.abs(frames[0].astype(int) - frames[-1]).max() > 1, f"{name} moves"


def test_viewer():
    """Hidden viewers share a window, windows start fast, and resizes keep the view."""
    g.require_rendering()
    import pyglet

    from trimesh.viewer.opengl.viewer import SceneViewer

    scene = g.trimesh.creation.box(extents=[3.0, 1.0, 2.0]).scene()
    # every hidden viewer shares one window, which isn't one of the user's
    # as that would keep `pyglet.app.run` from ever returning
    windows = set()
    for resolution in [(320, 240), (97, 61)]:
        viewer = g.offscreen_viewer(scene, resolution=resolution)
        windows.add(id(viewer.window))
        viewer.close()
    assert len(windows) == 1 and len(pyglet.app.windows) == 0, "one hidden window"
    window = SceneViewer(scene, start_loop=False)
    assert not window.state.quality, "a window starts fast"
    window.close()

    # resized to another shape, which a hidden window can't really be,
    # the image keeps square pixels and the camera's vertical view
    visible = SceneViewer(scene, start_loop=False, resolution=(64, 48))
    visible.to_array()
    visible.window.get_framebuffer_size = lambda: (97, 61)
    visible.on_resize(97, 61)
    scene.camera.fit_resolution((97, 61))
    assert g.np.allclose(visible.camera.fov, scene.camera.fov), (
        "resize keeps the vertical view"
    )
    g.assert_rays(scene, visible.to_array(True, True), prefix="resized ")
    visible.close()

    # with culling off only the depth test hides the back face, so a
    # box with a red front and blue back must only show red
    front = colored_box([0, 0, 255, 255])
    front.visual.face_colors[front.face_normals[:, 2] > 0.5] = [255, 0, 0, 255]
    unculled = g.render(front.scene(), flags={"cull": False}).color
    red = g.render(colored_box([255, 0, 0, 255]).scene()).color
    g.assert_images(unculled, red, "depth test hides the back")

    # a render must not depend on what was rendered before it
    translucent = colored_box([0, 0, 255, 100]).scene()
    images = [g.render(current).color for current in [scene, translucent, scene]]
    g.assert_images(images[0], images[2], "renders are independent")


def test_methods():
    """Lines, points and every live edit of a scene show up in the next render."""
    from scipy import ndimage

    g.require_rendering()

    def framed(*geometry):
        """A scene of `geometry` with a camera looking down -Z at the origin."""
        scene = g.trimesh.Scene(list(geometry))
        scene.camera_transform = tf.translation_matrix([0, 0, 5])
        return scene

    for width, scale in g.itertools.product([1, 4, 9], [1.0, 1e-6, 1e4]):
        # black on white so darkness is coverage: summed down every
        # column across the middle of the line it is exactly the width,
        # whatever the size of the scene
        segment = framed(g.trimesh.load_path(SEGMENT)).scaled(scale)
        image = g.render(segment, line_width=width).color
        coverage = (255 - image[:, 110:210, 0].astype(g.np.float64)).sum(axis=0) / 255
        assert (g.np.abs(coverage - width) < 0.25).all(), f"line width {width} at {scale}"

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
    assert (image[rows, columns, 0] > 200).all() and (
        image[rows, columns, 1] < 60
    ).all(), "points drawn at projections"
    drawn = (image[..., :3] < 250).any(-1)
    assert ndimage.label(drawn)[1] == 3, "points are separate dots"
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
    assert count == 3 and (g.np.asarray(lit) > g.np.asarray(shaded) + 30).all(), (
        "points are lit spheres"
    )

    # every edit of a live scene draws what a new viewer of it does
    line, box = g.trimesh.load_path(SEGMENT), colored_box(WHITE, extents=[0.5] * 3)
    scene = framed(line)
    away, back = tf.translation_matrix([1.5, 0, 0]), tf.translation_matrix([-1.5, 0, 0])
    edits = {
        "path color": lambda: setattr(line, "colors", [[0, 255, 0, 255]]),
        "added": lambda: scene.add_geometry(box, geom_name="b", transform=away),
        "shared": lambda: scene.graph.update(frame_to="c", geometry="b", matrix=back),
        "face colors": lambda: setattr(box.visual, "face_colors", [255, 0, 0, 255]),
        "vertex colors": lambda: setattr(box.visual, "vertex_colors", [0, 0, 255, 255]),
        "faces removed": lambda: box.update_faces(box.face_normals[:, 2] < 0.5),
        "moved": lambda: scene.graph.update(frame_to="c", matrix=back @ back),
        "deleted": lambda: scene.delete_geometry("b"),
    }
    options = {"line_width": 5, "flags": {"cull": False}}
    viewer = g.offscreen_viewer(scene, **options)
    previous = viewer.to_array().color
    for name, edit in edits.items():
        edit()
        live = viewer.to_array().color
        g.assert_images(live, g.render(scene, **options).color, f"{name} drawn live")
        assert g.np.abs(live.astype(int) - previous).max() > 1, f"{name} is visible"
        previous = live
    viewer.close()

    # subclassing, which users of `main`'s viewer did, draws the same
    base = g.trimesh.viewer.opengl.viewer.SceneViewer

    class Viewer(base):
        pass

    renders = []
    for kind in [base, Viewer]:
        current = kind(g.trimesh.creation.box().scene(), visible=False, start_loop=False)
        renders.append(current.to_array().color)
        current.close()
    g.assert_images(renders[0], renders[1], "subclass draws the same")

    # what `main`'s viewer took and had warns and works until April 2028
    with g.pytest.warns(DeprecationWarning, match="April 2028"):
        legacy = g.offscreen_viewer(
            g.trimesh.creation.box().scene(), line_settings={"point_size": 7.0}, record=1
        )
    with g.pytest.warns(DeprecationWarning, match="April 2028"):
        legacy.toggle_wireframe()
    with g.pytest.warns(DeprecationWarning, match="April 2028"):
        legacy.add_geometry("box", g.trimesh.creation.box())
    assert legacy.render_settings["point_size"] == 7.0, "legacy keywords"
    assert legacy.state.wireframe and legacy.state.scene_changed, "legacy toggles"
    assert not hasattr(legacy, "toggle_bogus"), "unknown attribute"
    legacy.close()


def test_every_scene():
    """
    Every scene trimesh loads and every render case, at odd sizes, fast
    and with quality: something is drawn, a redraw is the same, a window
    draws what is drawn offscreen, every switch a key toggles pressed
    twice is where it started, and the scene is unchanged by all of it.
    """
    from dataclasses import fields

    from PIL import Image

    from trimesh.scene.cameras import current_view
    from trimesh.viewer.opengl.viewer import KEY_BINDINGS, SceneViewer, ViewerState
    from trimesh.viewer.render import stage

    formats = set(g.trimesh.available_formats())
    paths = [
        path
        for path in sorted(g.data["model_paths"])
        if path.rsplit(".", 1)[-1].lower() in formats
    ]
    scenes = {g.os.path.basename(path): g.trimesh.load_scene(path) for path in paths}
    scenes.update({name: case_scene(case) for name, case in CASES.items()})
    scenes.update(
        {"box moves": g.dolly_scene("box"), "camera rides": g.dolly_scene("dolly")}
    )
    # render settings are keywords of each call, never scene content
    settings = {
        "exposure": 2.0,
        "bloom": 0.5,
        "ground_shadow": False,
        "show_environment": True,
        "line_width": 3.0,
        "point_size": 3.0,
    }
    # switches drawn into the image, not the window or the animation clock
    switches = {
        item.name for item in fields(ViewerState) if isinstance(item.default, bool)
    }
    switches -= {"fullscreen", "playing"}
    letters = [letter for letter, _, name in KEY_BINDINGS if name in switches]
    for name, scene in scenes.items():
        if scene.is_empty:
            continue
        before = scene.export(file_type="glb")
        assert scene.export(file_type="glb") == before, f"{name} export is deterministic"
        camera = (scene.has_camera, scene.has_camera and [*scene.camera.fov])
        state = (
            hash(scene.graph),
            camera,
            scene.has_lights,
            g.copy.deepcopy(scene.metadata),
            [id(item) for item in [*scene.cameras, *scene.animations]],
        )
        # a render reads the scene's own geometry, so values cached on
        # it like `smooth_shaded` are computed once for every render
        staged = stage(scene)[0]
        assert all(
            staged.geometry[key] is value for key, value in scene.geometry.items()
        ), f"{name} render shares geometry"
        scene.render(resolution=[64, 48], **settings)
        scene.save_image(resolution=[64, 48], **settings)
        scene.show(visible=False, start_loop=False, **settings).close()
        # an animation plays for its duration, where Pillow merges repeated frames
        gif = scene.render(file_type="gif", resolution=[32, 24], duration=0.2, fps=10)
        with Image.open(g.trimesh.util.wrap_as_stream(gif)) as image:
            played = 0
            for index in range(image.n_frames):
                image.seek(index)
                played += image.info["duration"]
        assert played == 200, f"{name} gif plays for its duration"
        # size is only units, from millimeters to meters: powers of two scale
        # exactly, so they draw the same
        drawn = g.render(scene.copy(), resolution=(64, 48), studio=True).color
        for factor in (1024.0, 1.0 / 16.0):
            scaled = g.render(scene.scaled(factor), resolution=(64, 48), studio=True)
            g.assert_images(scaled.color, drawn, f"{name} scaled {factor}")
        # and the default view never depends on rounding
        view = current_view(scene)[1][:3, :3]
        for factor in (1000.0, 3.0, 1.0 / 3.0):
            other = current_view(scene.scaled(factor))[1][:3, :3]
            assert g.np.allclose(other, view), f"{name} view scaled {factor}"
        # vertices stored far from their node, which moves them back, draw the same
        shifted, parents = scene.copy(), scene.graph.transforms.parents
        away = tf.translation_matrix([2.0**22] * 3)
        for key in shifted.geometry:
            shifted.geometry[key] = shifted.geometry[key].copy()
            shifted.geometry[key].apply_transform(away)
        moved = set(shifted.graph.nodes_geometry)
        for node in [n for n in scene.graph.nodes if n in parents]:
            # a geometry node's frame moves back, which its children undo
            local = shifted.graph.get(node, parents[node])[0]
            if node in moved:
                local = local @ g.np.linalg.inv(away)
            if parents[node] in moved:
                local = away @ local
            shifted.graph.update(frame_to=node, frame_from=parents[node], matrix=local)
        if len(scene.skins) == 0:
            far = g.render(shifted, resolution=(64, 48), studio=True).color
            g.assert_images(far, drawn, f"{name} stored far away")
        for size, quality in g.itertools.product([(97, 61), (64, 48)], [False, True]):
            label = f"{name} {size} quality {quality}"
            # bloom is part of quality
            atol = 1 + quality
            options = {
                "start_loop": False,
                "resolution": size,
                "flags": {"quality": quality},
            }
            hidden = SceneViewer(scene, visible=False, **options)
            from pyglet.window import key

            first = hidden.to_array(include_segmentation=True)
            assert (first.segmentation >= 0).any(), f"{label} draws something"
            g.assert_images(hidden.to_array().color, first.color, f"{label} redraw", atol)
            # playing an animation draws it without posing the scene
            hidden.on_animate(0.37)
            hidden.to_array()
            assert hash(scene.graph) == state[0], f"{label} graph unchanged playing"
            hidden.animation_time = None
            for letter in letters:
                symbol = getattr(key, letter.upper())
                hidden.on_key_press(symbol=symbol, modifiers=0)
                hidden.to_array()
                hidden.on_key_press(symbol=symbol, modifiers=0)
                twice = hidden.to_array().color
                g.assert_images(twice, first.color, f"{label} `{letter}` twice", atol)
            hidden.close()
            window = SceneViewer(scene, **options)
            window.window.dispatch_events()
            g.assert_images(window.to_array().color, first.color, f"{label} window", atol)
            window.close()
        # the defaults `Scene.lights` and `Scene.camera` generate are the ones drawn
        read = scene.copy()
        images = []
        for _ in range(2):
            viewer = SceneViewer(read, visible=False, **options)
            images.append(viewer.to_array().color)
            viewer.close()
            assert read.lights and read.camera, f"{label} defaults generated"
        g.assert_images(*images, f"{label} defaults read", atol)
        assert scene.export(file_type="glb") == before, f"{name} export unchanged"
        assert (
            hash(scene.graph),
            (scene.has_camera, scene.has_camera and [*scene.camera.fov]),
            scene.has_lights,
            scene.metadata,
            [id(item) for item in [*scene.cameras, *scene.animations]],
        ) == state, f"{name} state unchanged"


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
    # a tilted camera orbiting the world axis nearest its up, which
    # keeps the horizon level
    scene = g.trimesh.creation.box(extents=[3, 1, 2]).scene()
    camera = tf.euler_matrix(0.5, 0.2, 0.1) @ tf.translation_matrix([0, 0, 6])
    scene.camera_transform = camera
    video = scene.render(file_type="mkv", resolution=resolution, fps=fps)
    with g.TemporaryDirectory() as directory:
        path = g.os.path.join(directory, "video.mkv")
        with open(path, "wb") as video_file:
            video_file.write(video)
        frames = g.decode_video(path, resolution)
    count = 40
    assert len(frames) == count, "frame count"
    assert g.np.allclose(scene.camera_transform, camera), "camera restored"
    # the same frames rendered directly, with the floor and ambient
    # occlusion every render has, match the lossy video
    up = g.np.abs(camera[:3, 1]).argmax()
    axis = g.np.eye(3)[up] * g.np.sign(camera[up, 1])
    viewer = g.offscreen_viewer(scene, resolution=resolution, studio=True)
    for index in [0, count // 2, count - 1]:
        viewer.camera_transform = (
            tf.rotation_matrix(2 * g.np.pi * index / count, axis, scene.centroid) @ camera
        )
        expected = viewer.to_array().color[..., :3].astype(g.np.float64)
        assert g.np.abs(frames[index] - expected).mean() < 4.0, f"frame {index} matches"
    viewer.close()

    # the advertised one-liner: a model on the web with its textures
    # beside it, written as a turntable video
    with g.serve_meshes() as address, g.TemporaryDirectory() as directory:
        path = g.os.path.join(directory, "film.mp4")
        returned = g.trimesh.load_remote(f"{address}/fuze.obj").render(
            path, resolution=resolution, fps=fps
        )
        film = g.decode_video(path, resolution)
    assert returned is None, "one-liner returns nothing"
    assert len(film) == 40 and film.std(axis=(1, 2, 3)).min() > 1, "one-liner turntable"


def test_interaction():
    """Idle, settings, fixed default lights, and dragging then `z` resetting."""
    from scipy import ndimage

    sphere = g.trimesh.creation.icosphere(subdivisions=4)
    scene = pbr_mesh(sphere, [180, 180, 180, 255], 0.0, 0.4).scene()
    initial = scene.camera_transform.copy()
    # drawn fast like a window, which the reference without occlusion is
    viewer = g.offscreen_viewer(scene, resolution=[200, 150], flags={"quality": False})
    # pyglet's window module needs a context to exist before importing
    from pyglet.window import key, mouse

    full = viewer.to_array().color

    # nothing changed so nothing is drawn
    assert not any(viewer.on_draw() for _ in range(10)), "idle draws nothing"
    lights = viewer.light_transforms.copy()
    viewer.on_mouse_press(100, 75, mouse.LEFT, 0)
    viewer.on_mouse_drag(160, 95, 60, 20, mouse.LEFT, 0)
    assert not g.np.allclose(viewer.camera_transform, initial), "drag moves camera"
    dragged = viewer.camera_transform.copy()
    image = viewer.to_array().color[..., :3].astype(g.np.float64)
    started = (
        [*viewer.lights, viewer.environment],
        g.np.concatenate((viewer.light_transforms, [viewer.environment_transform])),
    )
    viewer.on_key_press(symbol=key.Z, modifiers=0)
    assert viewer.on_draw(), "reset redraws"
    assert g.np.allclose(viewer.camera_transform, initial), "reset camera"
    # dragging moved the viewer's camera and not the scene's
    assert g.np.allclose(scene.camera_transform, initial), "scene camera untouched"
    g.assert_images(viewer.to_array().color, full, "reset image")
    viewer.close()

    # the default lights are fixed in the world where the viewer started,
    # so moving the camera shows the same lighting from somewhere else
    scene.camera_transform = dragged
    fixed = shade_reference(scene, lights=started)
    moved = shade_reference(scene)
    interior = ndimage.binary_erosion(~g.np.isnan(fixed[..., 0]), iterations=2)
    assert g.np.allclose(started[1][:-1], lights), "lights stay put"
    assert g.np.abs(image - fixed)[interior].max() <= 2.0, "lit from where it started"
    assert g.np.abs(image - moved)[interior].max() > 10.0, "not lit from the new camera"

    # settings are `ViewerState` fields passed as keywords or flags, and a
    # misspelled one raises rather than being quietly ignored
    viewer = g.offscreen_viewer(scene, smooth=False, flags={"wireframe": True})
    assert not viewer.state.smooth and viewer.state.wireframe, "settings"
    viewer.close()
    with g.pytest.raises(TypeError):
        g.offscreen_viewer(scene, wirefram=True)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_viewer()
    test_methods()
    test_interaction()
