try:
    from . import generic as g
except BaseException:
    import generic as g

from types import SimpleNamespace

from scipy import ndimage

from trimesh.exchange.gltf.lights import rgbd_pack, rgbd_unpack
from trimesh.scene.lighting import DirectionalLight, EnvironmentLight, SpotLight
from trimesh.viewer.conversion import OCCLUSION_RADIUS, render_lights

np = g.np
tf = g.trimesh.transformations
# a straight line segment along X through the origin
SEGMENT = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
CASES = {case["name"]: case for case in g.data["render_cases"]}


def test_all_exists():
    # every name in `__all__` should be an attribute: a stale
    # entry breaks `from trimesh.viewer import *`
    from trimesh import viewer

    assert all(hasattr(viewer, name) for name in viewer.__all__), "every name exists"


def rgb(scene, **kwargs):
    """A render's color as (h, w, 3) float64."""
    return g.render(scene, **kwargs).color[..., :3].astype(np.float64)


def hit_mask(scene, hits=None):
    """Pixels whose camera ray hits the scene."""
    if hits is None:
        hits = g.camera_hits(scene)
    mask = np.zeros(scene.camera.resolution[::-1], dtype=bool)
    mask[hits.rows, hits.columns] = True
    return mask


def shade_reference(scene, normal=None):
    """
    The sRGB 0-255 image of opaque `PBRMaterial` meshes lit by punctual
    lights and a constant environment, NaN where no ray hits, from camera
    rays and the glTF specs, with an optional constant tangent space `normal`.
    """
    unitize = g.trimesh.util.unitize
    lights, transforms = render_lights(scene, scene.camera_transform)
    width, height = scene.camera.resolution
    meshes = [mesh.smooth_shaded for mesh in scene.dump()]
    hits = g.camera_hits(scene, meshes)
    points, corners = hits.points, hits.combined.faces[hits.face]
    barycentric = g.trimesh.triangles.points_to_barycentric(
        hits.combined.triangles[hits.face], points
    )
    surface = unitize(
        (hits.combined.vertex_normals[corners] * barycentric[:, :, None]).sum(axis=1)
    )
    view = -unitize(hits.vectors[hits.ray])
    # light the back of a triangle as if it were the front
    surface[(surface * view).sum(axis=1) < 0.0] *= -1.0
    if normal is not None:
        # the tangent and bitangent are how position changes with uv
        uv = np.vstack([m.visual.uv for m in meshes])[corners]
        position = hits.combined.vertices[corners]
        inverse = np.linalg.inv(uv[:, 1:] - uv[:, :1])
        edges = position[:, 1:] - position[:, :1]
        tangent = unitize(np.matmul(inverse[:, :1], edges)[:, 0])
        bitangent = unitize(np.matmul(inverse[:, 1:2], edges)[:, 0])
        surface = unitize(
            tangent * normal[0] + bitangent * normal[1] + surface * normal[2]
        )

    materials = [m.visual.material for m in meshes]
    base = (np.array([m.baseColorFactor for m in materials]) / 255.0)[hits.mesh, :3]
    metallic = np.array([[m.metallicFactor] for m in materials])[hits.mesh]
    roughness = np.clip([[m.roughnessFactor] for m in materials], 0.04, 1.0)[hits.mesh]
    reflectance = 0.04 * (1.0 - metallic) + base * metallic
    # `KHR_materials_diffuse_transmission` mixes the dielectric Lambert
    # reflection with Lambert transmission of light from the other side
    transmission = np.array([[m.diffuseTransmissionFactor or 0.0] for m in materials])
    transmission_color = [
        m.diffuseTransmissionColorFactor or [1.0] * 3 for m in materials
    ]
    transmission_color = np.multiply(transmission_color, transmission)[hits.mesh]
    transmission_color = transmission_color * (1.0 - metallic)
    diffuse_color = base * (1.0 - metallic) * (1.0 - transmission[hits.mesh])
    alpha_squared = roughness**4
    geometry_k = (roughness + 1.0) ** 2 / 8.0
    normal_dot_view = np.maximum((surface * view).sum(axis=1, keepdims=True), 1e-4)

    color = np.zeros_like(points)
    for light, pose in zip(lights[:-1], transforms[:-1]):
        # lights shine down their -Z, and only directional lights have
        # no position, falloff or range
        backwards = unitize(pose[:3, 2])
        to_light = np.tile(backwards, (len(points), 1))
        distance_squared = np.ones((len(points), 1))
        window = np.ones_like(distance_squared)
        if not isinstance(light, DirectionalLight):
            to_light = pose[:3, 3] - points
            distance_squared = (to_light**2).sum(axis=1, keepdims=True)
            if light.radius is not None:
                window = np.clip(1.0 - (distance_squared / light.radius**2) ** 2, 0, 1)
        direction = unitize(to_light)
        cone = np.ones_like(distance_squared)
        if isinstance(light, SpotLight):
            inner, outer = np.cos(light.innerConeAngle), np.cos(light.outerConeAngle)
            cosine = (direction * backwards).sum(axis=1, keepdims=True)
            cone = np.clip((cosine - outer) / max(inner - outer, 1e-3), 0.0, 1.0)
        radiance = light.color[:3] / 255.0 * light.intensity * (window * cone) ** 2
        radiance = radiance / distance_squared

        half = unitize(direction + view)
        cosine_light = (surface * direction).sum(1, keepdims=True)
        normal_dot_light = np.maximum(cosine_light, 0.0)
        # light from behind is transmitted with the fresnel term of the
        # light mirrored through the surface
        behind = np.maximum(-cosine_light, 0.0)
        mirrored = unitize(direction + 2.0 * behind * surface + view)
        mirrored_dot_half = np.clip((view * mirrored).sum(1, keepdims=True), 0.0, 1.0)
        transmitted = (1.0 - reflectance) * (1.0 - (1.0 - mirrored_dot_half) ** 5)
        transmitted = transmitted * transmission_color / np.pi
        normal_dot_half = np.maximum((surface * half).sum(1, keepdims=True), 0.0)
        view_dot_half = np.maximum((view * half).sum(1, keepdims=True), 0.0)
        fresnel = reflectance + (1.0 - reflectance) * (1.0 - view_dot_half) ** 5
        distribution = alpha_squared / (
            np.pi * (normal_dot_half**2 * (alpha_squared - 1.0) + 1.0) ** 2
        )
        geometry = (
            normal_dot_view / (normal_dot_view * (1 - geometry_k) + geometry_k)
        ) * (normal_dot_light / (normal_dot_light * (1 - geometry_k) + geometry_k))
        specular = fresnel * distribution * geometry
        specular = specular / (4 * normal_dot_view * normal_dot_light + 1e-4)
        diffuse = (1.0 - fresnel) * diffuse_color / np.pi
        color += (
            (diffuse + specular) * normal_dot_light + transmitted * behind
        ) * radiance

    # Karis's fit of the environment BRDF
    fit = roughness * [-1.0, -0.0275, -0.572, 0.022] + [1.0, 0.0425, 1.04, -0.04]
    bias = np.minimum(fit[:, :1] ** 2, np.exp2(-9.28 * normal_dot_view)) * fit[:, :1]
    bias = bias + fit[:, 1:2]
    # a constant environment's irradiance over pi and every prefiltered
    # level are its radiance, for diffuse, transmission and specular
    radiance = lights[-1].cubemap.reshape((-1, 3))
    assert np.ptp(radiance, axis=0).max() == 0.0, "constant environment"
    radiance = radiance[0] * lights[-1].intensity
    color += (diffuse_color + transmission_color) * radiance
    color += radiance * (
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


def floor_shadow(scene, pose, axis, height=0.0):
    """Pixels seeing the floor at `height`, and those not seeing the light at `pose[:, axis]`."""
    width, rows = scene.camera.resolution
    origins, vectors, pixels = scene.camera_rays()
    distance = (height - origins[:, 2]) / vectors[:, 2]
    points = np.zeros((rows, width, 3))
    points[rows - 1 - pixels[:, 1], pixels[:, 0]] = origins + vectors * distance[:, None]
    hits = g.camera_hits(scene)
    # the height of each first hit, which is the floor where nothing is hit
    first = np.full(points.shape[:2], height)
    first[hits.rows, hits.columns] = hits.points[:, 2]
    floor = np.abs(first - height) < 1e-3
    # the light's homogeneous position: directional lights are at
    # infinity along their +Z, where `w` is zero
    to_light = pose[:3, axis] - points[floor] * pose[3, axis]
    blocked = floor.copy()
    blocked[floor] = hits.combined.ray.intersects_any(
        points[floor] + [0, 0, 1e-4], g.trimesh.util.unitize(to_light)
    )
    return floor, blocked


def assert_reference(row):
    """Opaque PBR meshes are what the glTF specs and camera rays compute."""
    if row.quality:
        return
    texture = row.case["geometry"][0].get("material", {}).get("normalTexture")
    normal = None
    if texture is not None:
        # as the shader decodes the bytes of its texture
        normal = np.array(texture) / 255.0 * 2.0 - 1.0
    reference = shade_reference(row.scene, normal=normal)
    interior = ndimage.binary_erosion(~np.isnan(reference[..., 0]), iterations=2)
    assert interior.any(), row.label
    assert np.abs(row.image - reference)[interior].max() <= 2.0, row.label


def assert_exposure(row):
    """Exposure scales all light: every light and glow twice as bright."""

    def brighter(factor):
        lights, transforms = render_lights(row.scene, row.scene.camera_transform)
        copy = row.scene.copy()
        copy.lights = g.copy.deepcopy(lights)
        for light, pose in zip(copy.lights, transforms):
            light.intensity *= factor
            copy.graph[light.name] = pose
        for geometry in copy.geometry.values():
            geometry.visual.material.emissiveFactor = np.multiply(
                [0.1, 0.05, 0.0], factor
            )
        return copy

    exposed = rgb(brighter(1.0), exposure=2.0, **row.options)
    g.assert_images(exposed, rgb(brighter(2.0), **row.options), row.label)


def assert_shadow(row):
    """
    The first light of a plane under something shadows exactly where rays
    to it are blocked, leaving only the environment there.
    """
    if not row.quality:
        return
    scene, image, casts = row.scene, row.image, row.case["casts"]
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
    assert lit.any() and umbra.any() == casts, row.label
    assert not (rendered | blocked).any() or g.iou(rendered, blocked) > 0.98, row.label
    assert not umbra.any() or np.abs(image - ambient_only)[umbra].max() <= 2, row.label
    assert np.abs(image - unshadowed)[lit].max() <= 2, row.label


def assert_unlit(row):
    """The final linear color is base color times texture and vertex colors."""
    color = g.trimesh.visual.color
    item = row.case["geometry"][0]
    stored = next(iter(row.scene.geometry.values())).visual.material
    linear = (
        stored.baseColorFactor[:3]
        / 255.0
        * color.srgb_to_linear(np.divide(item["material"]["baseColorTexture"], 255.0))
        * color.srgb_to_linear(np.divide(item["vertex_color"][:3], 255.0))
    )
    interior = ndimage.binary_erosion(hit_mask(row.scene), iterations=2)
    assert interior.any(), row.label
    expected = color.linear_to_srgb(linear) * 255.0
    assert np.abs(row.image[interior] - expected).max() <= 1, row.label


def assert_round_trip(row):
    """A scene renders the same after a GLB round trip, its environment as RGBD."""
    if not row.quality:
        return
    scene = row.scene
    reloaded = g.roundtrip(scene.export(file_type="glb"), "glb")
    stored = scene.copy()
    for light in stored.lights:
        if isinstance(light, EnvironmentLight):
            light.cubemap = rgbd_unpack(rgbd_pack(light.cubemap))
    before, after = (
        g.png(current.render(resolution=scene.camera.resolution)).astype(int)
        for current in (stored, reloaded)
    )
    assert (before[..., :3] < 250).any(), row.label
    g.assert_images(before, after, row.label, 2)


def assert_rays(row):
    """
    Depth and segmentation are what each camera ray hits: perspective, and
    orthographic seeing 0.6 of the scene's scale, at an odd size which
    catches off by one pixel, with depth fit to any scale.
    """
    if row.quality:
        return
    for orthographic, scale in g.itertools.product([False, True], [1.0, 1e-3, 1e4]):
        scene = row.scene.scaled(scale)
        scene.camera.fit_resolution((97, 61))
        scene.camera_transform = scene.camera.look_at(
            scene.bounds, rotation=tf.euler_matrix(0.4, 0.3, 0.0)
        )
        if orthographic:
            scene.camera.magnification = (scene.scale * 0.8, scene.scale * 0.6)
        label = f"{row.label} orthographic {orthographic} at {scale} "
        arrays = g.render(scene, True, True, resolution=[97, 61])
        g.assert_rays(scene, arrays, label)
        # including every node sharing one geometry
        drawn = [n for n in scene.graph.nodes_geometry if scene.graph[n][0][:3, :3].any()]
        assert set(range(len(drawn))) <= set(np.unique(arrays.segmentation)), label


def assert_occlusion(row):
    """Quality occlusion darkens only where rays say the sky is hidden."""
    if not row.quality:
        return
    scene = row.scene
    # only an environment, so quality adds nothing but occlusion
    fast = rgb(scene, resolution=scene.camera.resolution, flags={"quality": False})
    darkening = fast.mean(axis=-1) - row.image.mean(axis=-1)
    # the box covers |x|, |y| < 1 and the floor is at zero: its top and the
    # floor past the radius see open sky, the floor beside it a crease
    hits = g.camera_hits(scene)
    radius = OCCLUSION_RADIUS * scene.scale
    away = np.linalg.norm(np.maximum(np.abs(hits.points[:, :2]) - 1.0, 0.0), axis=1)
    on_floor = hits.points[:, 2] < 1e-3
    regions = [np.zeros(darkening.shape, dtype=bool) for _ in range(2)]
    for region, selected in zip(
        regions, [~on_floor | (away > 1.2 * radius), on_floor & (away < 0.15)]
    ):
        region[hits.rows[selected], hits.columns[selected]] = True
    open_surface, crease = ndimage.binary_erosion(regions[0], iterations=3), regions[1]
    assert open_surface.sum() > 2000 and crease.sum() > 100, "has open and crease"
    assert np.abs(darkening[open_surface]).max() <= 3.0, "open surface untouched"
    assert darkening[crease].mean() >= 8.0, "crease darkened"
    assert (darkening[crease] >= 2.0).mean() >= 0.9, "crease mostly darker"


def assert_ground(row):
    """The invisible ground under a sun is only its shadow, and outside the scene."""
    if not row.quality:
        return
    scene, image, sun = row.scene, row.image, row.scene.graph["light_0"][0]
    arrays = g.render(scene, True, True, **row.options)
    floor, blocked = floor_shadow(scene, sun, 2)
    shaded = floor & (image.mean(axis=-1) < 250)
    umbra = ndimage.binary_erosion(blocked, iterations=3)
    lit = ndimage.binary_erosion(floor & ~blocked, iterations=3)
    # the floor loses the sun and keeps the environment, where a white
    # floor facing up reflects the constant environment and `I / pi` of
    # the sun's intensity `I` times the cosine of its angle from up
    umbra_value = 255.0 * 0.3 / (0.3 + 2.0 * sun[2, 2] / np.pi)
    assert umbra.sum() > 300, "has a shadow"
    assert g.iou(shaded, blocked) > 0.95, "shadow where rays are blocked"
    assert np.abs(image[umbra].mean(axis=-1) - umbra_value).max() <= 3, "umbra"
    assert np.abs(image[lit] - 255.0).max() <= 1, "lit floor invisible"
    assert np.allclose(scene.bounds, [[-1, -1, 0], [1, 1, 1]]), "not in bounds"
    assert len(scene.dump()) == 1, "not dumped"
    g.assert_rays(scene, arrays, "floor ")
    # from below the floor can't be seen at all
    scene = scene.copy()
    scene.camera_transform = tf.rotation_matrix(np.pi, [1, 0, 0]) @ scene.camera_transform
    options = {**row.options, "ground_shadow": False}
    g.assert_images(rgb(scene, **row.options), rgb(scene, **options), "from below")


def assert_bloom(row):
    """
    Bloom spills light the tone mapper compressed onto its neighbors: a
    glowing square in the dark lights nothing around it until bloom is on.
    """
    from trimesh.viewer.opengl.viewer import BLOOM_SPREAD

    if not row.quality:
        return
    images = {
        name: g.render(row.scene, bloom=bloom, **row.look, **row.options)
        .color[..., :3]
        .astype(int)
        for name, bloom in [("off", 0.0), ("on", 1.0), ("off again", 0.0)]
    }
    inside = hit_mask(row.scene)
    # within and past the antialiased edge, along the middle row both ways
    within = ndimage.binary_erosion(inside, iterations=1)
    outside = ~ndimage.binary_dilation(inside, iterations=1)
    middle = int(np.flatnonzero(inside.any(axis=1)).mean())
    columns = np.flatnonzero(inside[middle])
    bloom = (images["on"] - images["off"]).mean(axis=-1)[middle]
    falloff = [bloom[columns.max() + 2 :], bloom[: columns.min() - 1][::-1]]
    assert images["off"][within].min() >= 253, "glowing square is white"
    assert not images["off"][outside].any(), "no bloom is the background"
    assert (images["on"] >= images["off"]).all(), "bloom only adds"
    assert np.abs(images["on"] - images["off"])[within].max() <= 1, "keeps square"
    # every neighbor within the spread is lit, fading to nothing
    for side in falloff:
        assert (side > 0).sum() >= BLOOM_SPREAD * 120, "bloom lights neighbors"
        assert (np.diff(side) <= 0).all() and side[-1] == 0, "bloom fades"
    g.assert_images(images["off"], images["off again"], "bloom off again is never on")


def assert_follows(row):
    """
    Viewers redraw shadows and skinned meshes exactly when what casts them
    moves, though every hidden viewer shares the shadow maps and joints.
    """
    from trimesh.scene.animation import Skin

    if not row.quality:
        return
    scene = row.scene.copy()
    box_node = scene.graph.nodes_geometry[-1]
    # a box moved by a bone rather than its own node
    skinned = g.trimesh.creation.box()
    skinned.vertex_attributes["joints"] = np.zeros((8, 4), dtype=np.int64)
    skinned.vertex_attributes["weights"] = np.tile([1.0, 0, 0, 0], (8, 1))
    scene.graph.update(frame_to="bone", matrix=tf.translation_matrix([-2.5, 0, 0.5]))
    scene.add_geometry(skinned, node_name="skinned")
    scene.skins["skinned"] = Skin(joints=["bone"])
    shift, lift = (
        tf.translation_matrix([1.5, 0, 0]),
        tf.translation_matrix([-2.5, 2, 0.5]),
    )
    moved = scene.copy()
    moved.graph.update(frame_to=box_node, matrix=shift)
    moved.graph.update(frame_to="bone", matrix=lift)
    viewers = [g.offscreen_viewer(item, resolution=(160, 120)) for item in (scene, moved)]

    def frame(viewer):
        """The resolved frame of a scene whose geometry hasn't changed."""
        viewer.draw()
        return viewer.read_pixels("resolved").astype(np.int64)

    still, shifted = frame(viewers[0]), frame(viewers[1])
    assert np.abs(still - shifted).max() > 1, "moving the box moves pixels"
    # the same pixels as before the other viewer drew
    g.assert_images(frame(viewers[0]), still, "shadow and skin kept")
    scene.graph.update(frame_to=box_node, matrix=shift)
    scene.graph.update(frame_to="bone", matrix=lift)
    g.assert_images(frame(viewers[0]), shifted, "shadow follows the box")
    for viewer in viewers:
        viewer.close()


def assert_background(row):
    """A constant environment behind the scene is one color, tone mapped."""
    # lowered by 0.04 by the tone mapper and then sRGB encoded
    expected = (1.055 * (0.3 - 0.04) ** (1 / 2.4) - 0.055) * 255
    assert np.abs(row.image[:3, :3] - expected).max() <= 1, row.label


CHECKS = {
    "reference": assert_reference,
    "exposure": assert_exposure,
    "shadow": assert_shadow,
    "unlit": assert_unlit,
    "round_trip": assert_round_trip,
    "rays": assert_rays,
    "occlusion": assert_occlusion,
    "ground": assert_ground,
    "bloom": assert_bloom,
    "follows": assert_follows,
    "background": assert_background,
    # checked by `test_blender`
    "blender": lambda row: None,
}


def test_scenes():
    """
    Every row of `render_cases.json`, fast and with quality, passes its
    `checks`, and the scale of light and camera nodes changes nothing.
    """
    renders = {}
    for (name, case), quality in g.itertools.product(CASES.items(), [False, True]):
        scene = g.case_scene(case)
        options = {"resolution": scene.camera.resolution, "flags": {"quality": quality}}
        options.update(case.get("settings", {}))
        look = {
            "exposure": case.get("exposure", 1.0),
            "background": case.get("background"),
        }
        label = f"{name} quality {quality}"
        image = renders[label] = rgb(scene, **look, **options)
        row = SimpleNamespace(label=label, case=case, scene=scene, image=image)
        row.options, row.quality, row.look = options, quality, look
        for check in case["checks"]:
            CHECKS[check](row)
        if quality and not case.get("studio", False):
            # glTF light and camera nodes' scale, which 3ds Max sets, isn't
            # part of the light, and the camera's planes are in its units
            scaled = scene.copy()
            for light in scaled.lights:
                scaled.graph[light.name] = scaled.graph[light.name][0] @ tf.scale_matrix(
                    40
                )
            scaled.camera_transform = scene.camera_transform @ tf.scale_matrix(40.0)
            scaled.camera.z_near /= 40.0
            scaled.camera.z_far /= 40.0
            g.assert_images(rgb(scaled, **look, **options), image, f"{label} scaled")
    # cases which are other cases drawn another way, or scaled by a factor
    # which rounds both images, in both modes
    for (name, case), quality in g.itertools.product(CASES.items(), [False, True]):
        if "same_as" in case:
            label, factor = f"{name} quality {quality}", case.get("factor", 1.0)
            other = renders[f"{case['same_as']} quality {quality}"] * factor
            g.assert_images(renders[label], other, label, 1 + (factor != 1.0))


def test_methods():
    """Lines, points, `to_array` parts and every live edit show up in the next render."""
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
        coverage = (255 - image[:, 110:210, 0].astype(np.float64)).sum(axis=0) / 255
        assert (np.abs(coverage - width) < 0.25).all(), f"line width {width} at {scale}"
    # lines have depth and an id too
    scene = framed(
        g.trimesh.creation.box(extents=[0.2] * 3), g.trimesh.load_path(SEGMENT)
    )
    scene.camera_transform = tf.translation_matrix([0, 0.5, 4.0])
    for magnification in [None, (2.0, 1.5)]:
        scene.camera.magnification = magnification
        _, depth, index = g.render(scene, True, True, resolution=[160, 120])
        line = index == 1
        assert line.sum() > 50, f"{magnification} line has pixels"
        assert np.allclose(depth[line], 4.0, rtol=1e-4), f"{magnification} line depth"

    points = g.trimesh.PointCloud(
        [[-1, -1, 0], [1, 0.5, 0], [0, 0, 0]], colors=[255, 0, 0, 255]
    )
    scene = framed(points)
    image = g.render(scene, point_size=8).color
    width, height = scene.camera.resolution
    # normalized device coordinates of every point through the camera
    ndc = g.project(scene, points.vertices)
    columns = ((ndc[:, 0] + 1) / 2 * width).astype(int)
    rows = ((1 - ndc[:, 1]) / 2 * height).astype(int)
    assert (image[rows, columns, 0] > 200).all(), "points drawn at projections"
    assert (image[rows, columns, 1] < 60).all(), "points drawn at projections"
    assert ndimage.label((image[..., :3] < 250).any(-1))[1] == 3, "separate dots"

    # each part is returned only when asked for
    viewer = g.offscreen_viewer(g.case_scene(CASES["instanced"]), resolution=[80, 60])
    color_only = viewer.to_array()
    everything = viewer.to_array(include_depth=True, include_segmentation=True)
    depth_only = viewer.to_array(include_depth=True)
    # hiding a node hides that instance but not others sharing its geometry
    viewer.hide_geometry("geometry_0_0")
    seen = set(np.unique(viewer.to_array(include_segmentation=True).segmentation))
    nodes = viewer.scene.graph.nodes_geometry
    viewer.close()
    assert color_only.depth is None and color_only.segmentation is None, "only color"
    assert depth_only.segmentation is None, "only depth when asked"
    g.assert_images(everything.color, color_only.color, "parts agree")
    assert (everything.depth == depth_only.depth).all(), "depth agrees"
    assert nodes.index("geometry_0_0") not in seen, "hide one instance"
    assert {nodes.index("geometry_0_1"), nodes.index("geometry_0_2")} <= seen, "others"

    # every edit of a live scene draws what a new viewer of it does
    line = g.trimesh.load_path(SEGMENT)
    box = g.trimesh.creation.box(extents=[0.5] * 3)
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
        assert np.abs(live.astype(int) - previous).max() > 1, f"{name} is visible"
        previous = live
    viewer.close()


def test_every_scene():
    """
    Every scene trimesh loads and every render case, at odd sizes, fast
    and with quality: something is drawn, a redraw and a window draw the
    same, powers of two scale exactly, and the scene is unchanged.
    """
    from trimesh.viewer.opengl.viewer import SceneViewer
    from trimesh.viewer.render import stage

    formats = set(g.trimesh.available_formats())
    paths = [
        p
        for p in sorted(g.data["model_paths"])
        if p.rsplit(".", 1)[-1].lower() in formats
    ]
    scenes = {g.os.path.basename(path): g.trimesh.load_scene(path) for path in paths}
    scenes.update({name: g.case_scene(case) for name, case in CASES.items()})
    scenes.update(
        {"box moves": g.dolly_scene("box"), "camera rides": g.dolly_scene("dolly")}
    )
    # render settings are keywords of each call, never scene content
    settings = {"exposure": 2.0, "bloom": 0.5, "ground_shadow": False}
    settings.update(show_environment=True, line_width=3.0, point_size=3.0)
    fast = {"resolution": (64, 48), "studio": True}
    for name, scene in scenes.items():
        if scene.is_empty:
            continue
        before = scene.export(file_type="glb")
        assert scene.export(file_type="glb") == before, f"{name} export is deterministic"
        state = (
            hash(scene.graph),
            scene.has_camera and [*scene.camera.fov],
            scene.has_lights,
            g.copy.deepcopy(scene.metadata),
            [id(item) for item in [*scene.cameras, *scene.animations]],
        )
        # a render reads the scene's own geometry, so values cached on
        # it like `smooth_shaded` are computed once for every render
        staged = stage(scene)[0]
        assert all(staged.geometry[k] is v for k, v in scene.geometry.items()), name
        scene.render(resolution=[64, 48], **settings)
        # size is only units, from millimeters to meters: powers of two scale
        # exactly, so they draw the same
        drawn = g.render(scene.copy(), **fast).color
        for factor in (1024.0, 1.0 / 16.0):
            scaled = g.render(scene.scaled(factor), **fast).color
            g.assert_images(scaled, drawn, f"{name} scaled {factor}")
        # vertices stored far from their node, which moves them back, draw the same
        shifted, parents = scene.copy(), scene.graph.transforms.parents
        away = tf.translation_matrix([2.0**22] * 3)
        for key in shifted.geometry:
            shifted.geometry[key] = shifted.geometry[key].copy().apply_transform(away)
        moved = set(shifted.graph.nodes_geometry)
        for node in [n for n in scene.graph.nodes if n in parents]:
            # a geometry node's frame moves back, which its children undo
            local = shifted.graph.get(node, parents[node])[0]
            if node in moved:
                local = local @ np.linalg.inv(away)
            if parents[node] in moved:
                local = away @ local
            shifted.graph.update(frame_to=node, frame_from=parents[node], matrix=local)
        if len(scene.skins) == 0:
            g.assert_images(g.render(shifted, **fast).color, drawn, f"{name} far away")
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
            first = hidden.to_array(include_segmentation=True)
            assert (first.segmentation >= 0).any(), f"{label} draws something"
            g.assert_images(hidden.to_array().color, first.color, f"{label} redraw", atol)
            # playing an animation draws it without posing the scene
            hidden.on_animate(0.37)
            hidden.to_array()
            hidden.close()
            window = SceneViewer(scene, **options)
            window.window.dispatch_events()
            g.assert_images(window.to_array().color, first.color, f"{label} window", atol)
            window.close()
        # the defaults `Scene.lights` and `Scene.camera` generate are the ones drawn
        read = scene.copy()
        images = [g.render(read, resolution=size).color for _ in range(2)]
        assert read.lights and read.camera, f"{label} defaults generated"
        g.assert_images(*images, f"{label} defaults read", atol)
        assert scene.export(file_type="glb") == before, f"{name} export unchanged"
        assert (
            hash(scene.graph),
            scene.has_camera and [*scene.camera.fov],
            scene.has_lights,
            scene.metadata,
            [id(item) for item in [*scene.cameras, *scene.animations]],
        ) == state, f"{name} state unchanged"


def test_interaction():
    """A window starts fast, idles, and drag then `z` resets; default lights stay put."""
    from trimesh.viewer.opengl.viewer import SceneViewer

    g.require_rendering()
    material = {"baseColorFactor": [180, 180, 180, 255], "roughnessFactor": 0.4}
    sphere = {"kind": "sphere", "material": material}
    scene = g.case_scene({"geometry": [sphere], "studio": True, "camera": "default"})
    window = SceneViewer(scene.copy(), start_loop=False)
    assert not window.state.quality, "a window starts fast"
    window.close()
    initial = scene.camera_transform.copy()
    fast = {"resolution": [200, 150], "flags": {"quality": False}}
    viewer = g.offscreen_viewer(scene, **fast)
    # pyglet's window module needs a context to exist before importing
    from pyglet.window import key, mouse

    full = viewer.to_array().color
    # nothing changed so nothing is drawn
    assert not any(viewer.on_draw() for _ in range(10)), "idle draws nothing"
    lights = viewer.light_transforms.copy()
    viewer.on_mouse_press(100, 75, mouse.LEFT, 0)
    viewer.on_mouse_drag(160, 95, 60, 20, mouse.LEFT, 0)
    assert not np.allclose(viewer.camera_transform, initial), "drag moves camera"
    dragged = viewer.camera_transform.copy()
    image = viewer.to_array().color[..., :3].astype(np.float64)
    started = (
        [*viewer.lights, viewer.environment],
        np.concatenate((viewer.light_transforms, [viewer.environment_transform])),
    )
    viewer.on_key_press(symbol=key.Z, modifiers=0)
    assert viewer.on_draw(), "reset redraws"
    assert np.allclose(viewer.camera_transform, initial), "reset camera"
    # dragging moved the viewer's camera and not the scene's
    assert np.allclose(scene.camera_transform, initial), "scene camera untouched"
    g.assert_images(viewer.to_array().color, full, "reset image")
    viewer.close()

    # the default lights are fixed in the world where the viewer started,
    # so moving the camera shows the same lighting from somewhere else
    assert np.allclose(started[1][:-1], lights), "lights stay put"
    moved = scene.copy()
    moved.camera_transform = dragged
    fixed = moved.copy()
    fixed.lights = g.copy.deepcopy(started[0])
    for light, pose in zip(fixed.lights, started[1]):
        fixed.graph[light.name] = pose
    assert np.abs(image - rgb(fixed, **fast)).max() <= 2.0, "lit from where it started"
    assert np.abs(image - rgb(moved, **fast)).max() > 10.0, "not lit from new camera"
    # settings are `ViewerState` fields passed as keywords or flags
    viewer = g.offscreen_viewer(scene, smooth=False, flags={"wireframe": True})
    assert not viewer.state.smooth and viewer.state.wireframe, "settings"
    viewer.close()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_scenes()
