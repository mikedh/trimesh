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


# what a clamped cosine scales band l = 0, 1, 2 of radiance by into irradiance
BANDS = g.np.pi * g.np.array([1, *[2 / 3] * 3, *[1 / 4] * 5])


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
        from trimesh.scene.lighting import (
            autolight,
            cubemap_directions,
        )
        from trimesh.viewer.conversion import render_lights

        s = g.get_mesh("duck.dae")
        # directional lights need no geometry, so even an empty scene is lit
        for name, scene in {"duck": s, "empty": g.trimesh.Scene()}.items():
            graph = hash(scene.graph)
            # asking for the studio a render uses doesn't change the scene
            lights = autolight(scene)[0]
            assert hash(scene.graph) == graph, f"{name} studio leaves the graph"
            assert not scene.has_lights, f"{name} has no lights"
            kinds = [type(light).__name__ for light in lights]
            assert kinds == ["DirectionalLight", "DirectionalLight", "EnvironmentLight"]
            # the default lights `Scene.lights` generates are stored
            defaults = scene.lights
            assert scene.lights is defaults and scene.has_lights, name
            assert all(light.name in scene.graph.nodes for light in defaults), name
        assert isinstance(s.camera, g.trimesh.scene.cameras.Camera), "camera"

        # the key is above and to the right of the camera and the rim
        # is behind, measured in the camera's frame
        camera = g.trimesh.transformations.euler_matrix(0.3, 0.8, 0.0)
        camera[:3, 3] = [1, 2, 3]
        lights, transforms = autolight(s, camera)
        environment, frame = lights[2], transforms[2]
        towards = transforms[:2, :3, 2] @ camera[:3, :3]
        assert (towards @ camera[1, :3] > 0.3).all(), "key and rim above"
        assert towards[0, 2] > 0.3 and towards[0, 0] > 0.3, "key in front and right"
        assert towards[1, 2] < -0.3, "rim behind"
        # the environment's frame is a rotation whose +Y is the world axis
        # nearest the camera's up and whose +Z is level towards the camera
        assert (
            g.np.allclose(frame[:3, :3] @ frame[:3, :3].T, g.np.eye(3))
            and g.np.isclose(g.np.linalg.det(frame[:3, :3]), 1.0)
            and g.np.allclose(frame[:3, 1], [0, 1, 0])
            and frame[:3, 2] @ camera[:3, 2] > 0.5
        ), "environment frame"
        # a warm key and a cool rim under a slightly warm sky
        key, rim = (light.color[:3] for light in lights[:2])
        directions = cubemap_directions(environment.cubemap.shape[1])[0]
        sky = environment.cubemap.reshape((-1, 3))[
            directions.reshape((-1, 3))[:, 1].argmax()
        ]
        assert key[0] > key[1] > key[2] and rim[0] < rim[1] < rim[2], "warm key cool rim"
        assert sky[0] > sky[2], "sky is warm"
        # the lights are fixed in the world where the camera started
        moved = g.trimesh.transformations.rotation_matrix(1.0, [0, 1, 0]) @ camera
        assert not g.np.allclose(autolight(s, moved)[1], transforms), "world fixed"
        # explicit lights are used as they are with the default environment
        s.lights = [g.trimesh.scene.lighting.PointLight(name="bulb")]
        s.graph["bulb"] = g.np.eye(4)
        explicit = [type(light).__name__ for light in render_lights(s, camera)[0]]
        assert explicit == ["PointLight", "EnvironmentLight"], "explicit lights"
        assert isinstance(s.lights, list), "stored lights are a list"

    def test_environment_gltf(self):
        # an environment light is the `EXT_lights_image_based` light of the
        # scene: its cube map is the light and the mip levels and harmonics
        # the file stores are recomputed from it, so once the first export
        # has quantized it to RGBD every later export is the same bytes
        from PIL import Image

        from trimesh.exchange.gltf.lights import (
            ENVIRONMENT,
            RGBD_GAMMA,
            rgbd_pack,
            rgbd_unpack,
        )
        from trimesh.scene.lighting import (
            EnvironmentLight,
            SpotLight,
            cubemap_directions,
            prefilter,
            spherical_harmonics,
        )

        np = g.np
        tf = g.trimesh.transformations
        studio = g.gradient_environment(
            sky=[0.5, 0.6, 0.7],
            ground=[0.0, 0.0, 0.0],
            softboxes=[([0, 1, 0], 4.0, [9.0, 4.0, 1.0])],
        )
        environment = EnvironmentLight(name="sky", intensity=2.5, cubemap=studio.cubemap)
        scene = g.trimesh.Scene(g.trimesh.creation.box())
        rotation = tf.random_rotation_matrix(seed=0)
        scene.graph["sky"] = rotation
        scene.graph["spot"] = np.eye(4)
        scene.lights = [SpotLight(name="spot"), environment]
        exports, loaded = [scene.export(file_type="glb")], []
        for _ in range(2):
            loaded.append(g.roundtrip(exports[-1], "glb"))
            exports.append(loaded[-1].export(file_type="glb"))
        g.validate_glb(exports[0], "environment light")
        reloaded = loaded[0]
        found = reloaded.lights[-1]
        header, blob = g.glb_chunks(exports[0])
        entry = header["extensions"][ENVIRONMENT]["lights"][0]

        def bound(radiance):
            # RGBD stores `D = floor(255 / max(1, brightest)) / 255` as alpha
            # and each channel `(radiance * D) ** (1 / gamma)` to the nearest
            # 1/255, so decoded it is off by what half a step does at most
            radiance = np.asarray(radiance, dtype=np.float64)
            scale = np.floor(255.0 / np.maximum(radiance.max(axis=-1, keepdims=True), 1))
            encoded = (radiance * scale / 255.0) ** (1.0 / RGBD_GAMMA)
            step = (encoded + 0.5 / 255.0) ** RGBD_GAMMA - encoded**RGBD_GAMMA
            return step * 255.0 / scale + np.finfo(np.float32).eps * radiance

        # the light as the viewer and the file see it, a float32 mirror level
        original = prefilter(environment.cubemap)[0]
        tolerance = bound(original)
        # the faces of the mirror level as PNGs, flipped about their
        # vertical axis as the spec says
        faces = [
            np.asarray(
                Image.open(
                    g.trimesh.util.wrap_as_stream(
                        blob[view.get("byteOffset", 0) :][: view["byteLength"]]
                    )
                )
            )
            for view in (
                header["bufferViews"][header["images"][index]["bufferView"]]
                for index in np.ravel(entry["specularImages"])
            )
        ]
        # projecting radiance onto the harmonics is linear, so a texel's
        # error moves the coefficients by at most its weight times it
        directions, solid_angle = cubemap_directions(len(original[0]))
        weights = np.sqrt(solid_angle).reshape((-1, 1))
        projection = np.linalg.pinv(harmonics(directions.reshape((-1, 3))) * weights)
        harmonic_bound = np.abs(projection * weights.T) @ tolerance.reshape((-1, 3))

        # kind, intensity and pose round trip in `test_render_cases`, drawn
        assert exports[1] == exports[2], "second and third exports are identical"
        assert (np.abs(found.cubemap - original) <= tolerance).all(), (
            "cube map within RGBD"
        )
        assert np.allclose(
            entry["irradianceCoefficients"], spherical_harmonics(environment.cubemap)
        ), "harmonics stored"
        assert (
            np.abs(spherical_harmonics(found.cubemap) - spherical_harmonics(original))
            <= harmonic_bound * BANDS[:, None]
        ).all(), "harmonics within RGBD"
        assert all(
            np.abs(after - before).max() <= tolerance.max()
            for after, before in zip(prefilter(found.cubemap), prefilter(original))
        ), "levels within RGBD"
        assert entry["specularImageSize"] == len(original[0]), "image size"
        assert [len(level) for level in entry["specularImages"]] == [6] * len(
            prefilter(original)
        ) and [face.shape for face in faces] == [
            (len(original[0]) >> level, len(original[0]) >> level, 4)
            for level in range(len(entry["specularImages"]))
            for _ in range(6)
        ], "mip levels halve"
        assert all(
            (np.abs(rgbd_unpack(face)[:, ::-1] - texels) <= tolerance[index]).all()
            for index, (face, texels) in enumerate(zip(faces, original))
        ), "faces are RGBD flipped"
        assert "gltf_extensions" not in reloaded.metadata, "not in metadata"

        # packing RGBD what it unpacked gives the same bytes, including
        # radiance exactly at the brightest each alpha holds
        rng = np.random.default_rng(0)
        brightest = 255.0 / np.arange(1, 256, dtype=np.float64)
        radiance = np.vstack(
            (rng.exponential(4.0, size=(4096, 3)), np.outer(brightest, [1.0, 0.5, 0.0]))
        )
        packed = rgbd_pack(radiance)
        unpacked = rgbd_unpack(packed)
        assert (np.abs(unpacked - radiance) <= bound(radiance)).all(), "rgbd within bound"
        assert np.array_equal(rgbd_pack(unpacked.astype(np.float32)), packed), (
            "rgbd is idempotent"
        )

        # an environment is white unless tinted
        assert (EnvironmentLight(cubemap=np.ones((6, 2, 2, 3))).color == 255).all(), (
            "white"
        )

        plain = g.trimesh.Scene(g.trimesh.creation.box())
        assert ENVIRONMENT.encode() not in plain.export(file_type="glb"), (
            "default scene has no environment"
        )


def test_environment():
    # an environment's harmonics, the mip levels the viewer reflects and
    # equirectangular images match brute force over every texel
    from trimesh.scene.lighting import (
        SPECULAR_SHARPNESS,
        EnvironmentLight,
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
    normals = g.trimesh.util.unitize(rng.normal(size=(500, 3)))
    cosine = normals @ directions.T

    def relative(model, truth):
        return np.abs((model - truth) @ LUMINANCE) / np.abs(truth @ LUMINANCE)

    def irradiance(cubemap):
        # light over each normal's hemisphere weighted by cosine
        return (np.clip(cosine, 0.0, None) * solid_angle) @ cubemap.reshape((-1, 3))

    assert np.isclose(solid_angle.sum(), 4.0 * np.pi), "texels cover the sphere"
    # radiance which is harmonics up to l=2 has exactly those harmonics
    # times the clamped cosine's pi, 2 pi / 3 and pi / 4 for each band as
    # irradiance, which is what Babylon.js, the reader of
    # `EXT_lights_image_based`, stores: its own cube maps are projected and
    # scaled by `SHCosKernelConvolution` before the steps its loader applies
    # to `irradianceCoefficients`, in core/src/Maths/sphericalPolynomial.pure.ts
    coefficients = rng.normal(size=(9, 3))
    exact = EnvironmentLight(
        cubemap=(harmonics(directions) @ coefficients).reshape(shape)
    )
    assert np.allclose(
        spherical_harmonics(exact.cubemap), coefficients * BANDS[:, None], atol=1e-10
    ), "harmonics are exact"
    assert np.allclose(
        harmonics(normals) @ spherical_harmonics(exact.cubemap),
        irradiance(exact.cubemap),
        atol=1e-3,
    ), "exact irradiance"
    # the studio has sharper light which l=2 harmonics blur a little
    error = relative(
        harmonics(normals) @ spherical_harmonics(studio.cubemap),
        irradiance(studio.cubemap),
    )
    assert error.mean() < 0.015, "studio irradiance mean"
    assert error.max() < 0.05, "studio irradiance max"

    # every mip level is the cube map blurred by the spherical Gaussian of
    # its roughness, and blending them is close to a GGX reflection seen
    # head on, where the squared cosine of the half vector is (1 + cosine) / 2
    levels = prefilter(studio.cubemap)
    count = len(levels)
    assert [len(level[0]) for level in levels] == [
        shape[1] >> index for index in range(count)
    ], "levels halve"
    # brute force over every source texel for a fixed sample of the
    # texels of each level, as each is computed independently
    for index, level in enumerate(levels[1:], 1):
        sharpness = np.exp(np.polyval(SPECULAR_SHARPNESS, np.log(index / (count - 1))))
        towards = cubemap_directions(len(level[0]))[0].reshape((-1, 3))
        sample = rng.choice(len(towards), size=min(512, len(towards)), replace=False)
        weights = np.exp(sharpness * (towards[sample] @ directions.T - 1.0)) * solid_angle
        truth = (weights @ studio.cubemap.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        error = relative(level.reshape((-1, 3))[sample], truth)
        assert error.mean() < 0.003, f"level {index} mean"
        assert error.max() < 0.01, f"level {index} max"
    for roughness in [0.1, 0.3, 0.6, 1.0]:
        level = roughness * (count - 1)
        low, fraction = int(level), level % 1.0
        high = min(low + 1, count - 1)
        # blended at the middle of each texel of the smaller level, which
        # is the corner of the four of the larger one linear filtering averages
        size = len(levels[high][0])
        step = len(levels[low][0]) // size
        lower = levels[low].reshape((6, size, step, size, step, 3)).mean(axis=(2, 4))
        model = ((1 - fraction) * lower + fraction * levels[high]).reshape((-1, 3))
        towards = cubemap_directions(size)[0].reshape((-1, 3))
        sample = rng.choice(len(towards), size=min(512, len(towards)), replace=False)
        model = model[sample]
        facing = towards[sample] @ directions.T
        alpha = roughness**4
        weights = np.clip(facing, 0.0, 1.0) * solid_angle
        weights *= alpha / ((1 + facing) / 2 * (alpha - 1) + 1) ** 2
        truth = (weights @ studio.cubemap.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        error = relative(model, truth)
        assert error.mean() < 0.04, f"specular {roughness} mean"
        assert error.max() < 0.18, f"specular {roughness} max"

    # an equirectangular image's top row is up and `atan2(z, x)` goes
    # across it from -pi, so a smooth environment's faces are it
    height = 64
    polar, azimuth = np.meshgrid(
        (np.arange(height) + 0.5) / height * np.pi,
        (np.arange(2 * height) + 0.5) / height * np.pi - np.pi,
        indexing="ij",
    )
    pixels = np.stack(
        (np.sin(polar) * np.cos(azimuth), np.cos(polar), np.sin(polar) * np.sin(azimuth)),
        axis=-1,
    )

    def smooth(direction):
        # a broad light low on one side over a gradient
        lobe = np.exp(3.0 * (direction @ g.trimesh.util.unitize([1.0, -0.3, 2.0]) - 1))
        return np.outer(lobe, [1.0, 0.5, 0.2]) + direction[..., 1:2] * 0.1 + 0.2

    image = smooth(pixels.reshape((-1, 3))).reshape((height, 2 * height, 3))
    cubemap = EnvironmentLight(cubemap=image).cubemap
    faces = smooth(cubemap_directions(len(cubemap[0]))[0].reshape((-1, 3)))
    assert np.allclose(cubemap.reshape((-1, 3)), faces, rtol=5e-3), (
        "equirectangular to faces"
    )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
