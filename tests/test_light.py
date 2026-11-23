try:
    from . import generic as g
except BaseException:
    import generic as g


def harmonics(directions):
    """
    The real spherical harmonics up to l=2 of unit vectors as Babylon.js
    reads `EXT_lights_image_based`, with the Condon-Shortley phase.
    """
    x, y, z = g.np.moveaxis(directions, -1, 0)
    terms = [g.np.ones_like(x), y, z, x, x * y, y * z, 3 * z**2 - 1, x * z, x**2 - y**2]
    scale = g.np.sqrt(g.np.array([1, 3, 3, 3, 15, 15, 5 / 4, 15, 15 / 4]) / (4 * g.np.pi))
    return g.np.stack(terms, axis=-1) * scale * [1, -1, 1, -1, 1, -1, 1, -1, 1]


class LightTests(g.unittest.TestCase):
    def test_basic(self):
        for light_class in [
            g.trimesh.scene.lighting.DirectionalLight,
            g.trimesh.scene.lighting.PointLight,
            g.trimesh.scene.lighting.SpotLight,
        ]:
            light = light_class()
            assert isinstance(light.intensity, float)
            assert light.color.shape == (4,)
            assert light.color.dtype == g.np.uint8
            # the default color is the dim gray it has always been
            assert g.np.array_equal(light.color, [60, 60, 60, 255])

    def test_scene(self):
        from trimesh.scene.lighting import autolight

        s = g.get_mesh("duck.dae")
        # directional lights need no geometry, so even an empty scene is lit
        for name, scene in {"duck": s, "empty": g.trimesh.Scene()}.items():
            graph = hash(scene.graph)
            # asking for the studio a render uses doesn't change the scene
            kinds = [type(light).__name__ for light in autolight(scene)[0]]
            assert hash(scene.graph) == graph and not scene.has_lights, name
            assert kinds == ["DirectionalLight", "DirectionalLight", "EnvironmentLight"]
            # the default lights `Scene.lights` makes are kept with nodes, but
            # aren't the scene's so exports write no lights as before
            defaults = scene.lights
            assert scene.lights is defaults and not scene.has_lights, name
            assert all(light.name in scene.graph.nodes for light in defaults), name
            if scene.is_empty:
                continue
            trees = []
            scene.export(file_type="glb", tree_postprocessor=trees.append)
            assert "extensionsUsed" not in trees[0] or not any(
                "lights" in used for used in trees[0]["extensionsUsed"]
            ), f"{name} exports no default lights"
        assert isinstance(s.camera, g.trimesh.scene.cameras.Camera), "camera"
        # the environment's frame is a rotation whose +Y is the world axis
        # nearest the camera's up and whose +Z is level towards the camera
        camera = g.trimesh.transformations.euler_matrix(0.3, 0.8, 0.0)
        frame = autolight(s, camera)[1][2]
        assert (
            g.np.allclose(frame[:3, :3] @ frame[:3, :3].T, g.np.eye(3))
            and g.np.isclose(g.np.linalg.det(frame[:3, :3]), 1.0)
            and g.np.allclose(frame[:3, 1], [0, 1, 0])
            and frame[:3, 2] @ camera[:3, 2] > 0.5
        ), "environment frame"


def test_environment():
    """
    An environment's harmonics and mip levels match brute force over every
    texel, and as an `EXT_lights_image_based` light it is a fixed point of
    loading whose RGBD faces decode to the mirror level within a step.
    """
    from trimesh.exchange.gltf.lights import (
        ENVIRONMENT,
        RGBD_GAMMA,
        rgbd_pack,
        rgbd_unpack,
    )
    from trimesh.scene.lighting import (
        SPECULAR_SHARPNESS,
        EnvironmentLight,
        SpotLight,
        autolight,
        cubemap_directions,
        prefilter,
        spherical_harmonics,
    )
    from trimesh.viewer.opengl.shaders import LUMINANCE

    np = g.np
    rng = np.random.default_rng(0)
    studio = autolight(g.trimesh.Scene(), np.eye(4))[0][-1]
    shape = studio.cubemap.shape
    directions, solid_angle = cubemap_directions(shape[1])
    directions, solid_angle = directions.reshape((-1, 3)), solid_angle.reshape(-1)
    assert np.isclose(solid_angle.sum(), 4.0 * np.pi), "texels cover the sphere"
    # radiance which is harmonics up to l=2 has exactly those harmonics
    # times the clamped cosine's pi, 2 pi / 3 and pi / 4 for each band as
    # irradiance, which is what Babylon.js stores for `irradianceCoefficients`
    coefficients = rng.normal(size=(9, 3))
    exact = (harmonics(directions) @ coefficients).reshape(shape)
    bands = np.pi * np.array([1, *[2 / 3] * 3, *[1 / 4] * 5])
    assert np.allclose(
        spherical_harmonics(exact), coefficients * bands[:, None], atol=1e-10
    ), "harmonics are exact"
    normals = g.trimesh.util.unitize(rng.normal(size=(500, 3)))
    cosine = np.clip(normals @ directions.T, 0.0, None) * solid_angle
    assert np.allclose(
        harmonics(normals) @ spherical_harmonics(exact),
        cosine @ exact.reshape((-1, 3)),
        atol=1e-3,
    ), "exact irradiance"
    # every mip level is the cube map blurred by the spherical Gaussian of
    # its roughness, brute forced for a sample of each level's texels
    levels = prefilter(studio.cubemap)
    count = len(levels)
    assert [len(level[0]) for level in levels] == [shape[1] >> i for i in range(count)]
    for index, level in enumerate(levels[1:], 1):
        sharpness = np.exp(np.polyval(SPECULAR_SHARPNESS, np.log(index / (count - 1))))
        towards = cubemap_directions(len(level[0]))[0].reshape((-1, 3))
        sample = rng.choice(len(towards), size=min(512, len(towards)), replace=False)
        weights = np.exp(sharpness * (towards[sample] @ directions.T - 1.0)) * solid_angle
        truth = (weights @ studio.cubemap.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        model = level.reshape((-1, 3))[sample]
        error = np.abs((model - truth) @ LUMINANCE) / np.abs(truth @ LUMINANCE)
        assert error.mean() < 0.003 and error.max() < 0.01, f"level {index}"

    def bound(radiance):
        # RGBD stores `D = floor(255 / max(1, brightest)) / 255` as alpha
        # and each channel `(radiance * D) ** (1 / gamma)` to the nearest
        # 1/255, so decoded it is off by what half a step does at most
        radiance = np.asarray(radiance, dtype=np.float64)
        scale = np.floor(255.0 / np.maximum(radiance.max(axis=-1, keepdims=True), 1))
        encoded = (radiance * scale / 255.0) ** (1.0 / RGBD_GAMMA)
        step = (encoded + 0.5 / 255.0) ** RGBD_GAMMA - encoded**RGBD_GAMMA
        return step * 255.0 / scale + np.finfo(np.float32).eps * radiance

    # packing RGBD what it unpacked gives the same bytes, including
    # radiance exactly at the brightest each alpha holds
    brightest = 255.0 / np.arange(1, 256, dtype=np.float64)
    radiance = np.vstack(
        (rng.exponential(4.0, size=(4096, 3)), np.outer(brightest, [1.0, 0.5, 0.0]))
    )
    unpacked = rgbd_unpack(rgbd_pack(radiance))
    assert (np.abs(unpacked - radiance) <= bound(radiance)).all(), "rgbd within bound"
    assert np.array_equal(rgbd_pack(unpacked.astype(np.float32)), rgbd_pack(radiance))

    sky = g.gradient_environment(
        sky=[0.5, 0.6, 0.7],
        ground=[0.0, 0.0, 0.0],
        # off every axis, so a face flipped the wrong way changes it
        softboxes=[([0.5, 1, 0.2], 4.0, [9.0, 4.0, 1.0])],
    )
    environment = EnvironmentLight(name="sky", intensity=2.5, cubemap=sky.cubemap)
    scene = g.trimesh.Scene(g.trimesh.creation.box())
    scene.graph["sky"] = g.trimesh.transformations.random_rotation_matrix(seed=0)
    scene.graph["spot"] = np.eye(4)
    scene.lights = [SpotLight(name="spot"), environment]
    g.assert_glb_roundtrip(scene, "environment")
    trees = []
    exported = scene.export(file_type="glb", tree_postprocessor=trees.append)
    entry = trees[0]["extensions"][ENVIRONMENT]["lights"][0]
    reloaded = g.roundtrip(exported, "glb")
    # the light as the viewer and the file see it, a float32 mirror level,
    # whose faces are flipped about their vertical axis as the spec says
    mirror = prefilter(environment.cubemap)[0]
    found = reloaded.lights[-1].cubemap
    assert (np.abs(found - mirror) <= bound(mirror)).all(), "cube map within RGBD"
    assert np.allclose(
        entry["irradianceCoefficients"], spherical_harmonics(environment.cubemap)
    ), "harmonics stored"
    assert entry["specularImageSize"] == len(mirror[0]), "image size"
    assert [len(level) for level in entry["specularImages"]] == [6] * len(
        prefilter(mirror)
    ), "every mip level"
    assert "gltf_extensions" not in reloaded.metadata, "not in metadata"


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
