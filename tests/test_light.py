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
        from trimesh.viewer.opengl.shaders import LUMINANCE

        s = g.get_mesh("duck.dae")
        checks = {}
        # directional lights need no geometry, so even an empty scene is lit
        for name, scene in {"duck": s, "empty": g.trimesh.Scene()}.items():
            graph = hash(scene.graph)
            # asking for the default lights doesn't change the scene
            autolight(scene)
            checks[f"{name} autolight leaves the graph"] = hash(scene.graph) == graph
            checks[f"{name} has no lights"] = not scene.has_lights
            lights = scene.lights
            kinds = [type(light).__name__ for light in lights]
            # in glTF units a white surface facing a light reflects its
            # linear color times its intensity over pi
            reflected = [
                g.np.dot(light.color[:3] / 255.0, LUMINANCE) * light.intensity / g.np.pi
                for light in lights[:2]
            ]
            current = {
                # a key and a rim light, which a white surface facing them
                # reflects 0.58 and 0.28 of as luminance at any scale, and
                # the studio's environment
                "kinds": kinds
                == ["DirectionalLight", "DirectionalLight", "EnvironmentLight"],
                "reflected": g.np.allclose(reflected, [0.58, 0.28]),
                # the default lights are generated once into the scene
                "a list": isinstance(lights, list),
                # stored, but only the scene's own once they're changed
                "stored": scene.lights is lights and not scene.has_lights,
                "same objects": all(a is b for a, b in zip(scene.lights, lights)),
                "in the graph": all(light.name in scene.graph.nodes for light in lights),
            }
            checks.update({f"{name} {check}": ok for check, ok in current.items()})
        checks["camera"] = isinstance(s.camera, g.trimesh.scene.cameras.Camera)

        # the key is above and to the right of the camera and the rim
        # is behind, measured in the camera's frame
        camera = g.trimesh.transformations.euler_matrix(0.3, 0.8, 0.0)
        camera[:3, 3] = [1, 2, 3]
        lights, transforms = autolight(s, camera)
        environment, frame = lights[2], transforms[2]
        towards = transforms[:2, :3, 2] @ camera[:3, :3]
        angle = g.np.degrees(g.np.arcsin(towards @ camera[1, :3]))
        checks["key elevation"], checks["rim elevation"] = g.np.isclose(angle, [35, 65])
        checks["key in front and right"] = towards[0, 2] > 0.3 and towards[0, 0] > 0.3
        checks["rim behind"] = towards[1, 2] < -0.3
        # the environment's frame is a rotation whose +Y is the world axis
        # nearest the camera's up and whose +Z is level towards the camera
        checks["environment frame"] = (
            g.np.allclose(frame[:3, :3] @ frame[:3, :3].T, g.np.eye(3))
            and g.np.isclose(g.np.linalg.det(frame[:3, :3]), 1.0)
            and g.np.allclose(frame[:3, 1], [0, 1, 0])
            and frame[:3, 2] @ camera[:3, 2] > 0.5
        )
        # a warm key and a cool rim under a slightly warm sky, with the
        # softbox on the key's side warm and the one opposite cool, which
        # are as bright as the white they're tinted from, so the warm
        # studio is lit as brightly as a white one: every texel is a
        # gradient from a floor of 0.096 to a sky of 0.31 plus softboxes
        # overhead, on the key's side and the rim's of luminance 0.35, 2
        # and 1 with sharpness 4, 30 and 30, at (azimuth, elevation)
        azimuth, elevation = g.np.radians([[0, 90], [60, 20], [-70, 12]]).T
        softboxes = g.np.column_stack(
            (
                g.np.sin(azimuth) * g.np.cos(elevation),
                g.np.sin(elevation),
                g.np.cos(azimuth) * g.np.cos(elevation),
            )
        )
        directions = cubemap_directions(environment.cubemap.shape[1])[0]
        lobes = g.np.exp([4.0, 30.0, 30.0] * (directions @ softboxes.T - 1.0))
        luminance = 0.096 + 0.214 * (0.5 + 0.5 * directions[..., 1])
        luminance += lobes @ [0.35, 2.0, 1.0]
        checks["environment luminance"] = g.np.allclose(
            environment.cubemap @ LUMINANCE, luminance
        )
        key, rim = (light.color[:3] for light in lights[:2])
        # the texel nearest each softbox
        nearest = (directions.reshape((-1, 3)) @ softboxes.T).argmax(axis=0)
        sky, near, far = environment.cubemap.reshape((-1, 3))[nearest]
        for name, color in {"key": key, "near softbox": near}.items():
            checks[f"{name} is warm"] = color[0] > color[1] > color[2]
        for name, color in {"rim": rim, "far softbox": far}.items():
            checks[f"{name} is cool"] = color[0] < color[1] < color[2]
        checks["sky is warm"] = sky[0] > sky[2]
        # the lights are fixed in the world where the camera started
        moved = g.trimesh.transformations.rotation_matrix(1.0, [0, 1, 0]) @ camera
        checks["world fixed"] = not g.np.allclose(autolight(s, moved)[1], transforms)
        # explicit lights are used as they are with the default environment
        s.lights = [g.trimesh.scene.lighting.PointLight(name="bulb")]
        s.graph["bulb"] = g.np.eye(4)
        explicit = [type(light).__name__ for light in render_lights(s, camera)[0]]
        checks["explicit lights"] = explicit == ["PointLight", "EnvironmentLight"]
        checks["stored lights are a list"] = isinstance(s.lights, list)
        for name, ok in checks.items():
            assert ok, name

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
            DirectionalLight,
            EnvironmentLight,
            SpotLight,
            cubemap_directions,
            prefilter,
            spherical_harmonics,
        )
        from trimesh.viewer.conversion import render_lights

        np = g.np
        tf = g.trimesh.transformations
        studio = g.gradient_environment(
            sky=[0.5, 0.6, 0.7],
            ground=[0.0, 0.0, 0.0],
            softboxes=[([0, 1, 0], 4.0, [9.0, 4.0, 1.0])],
        )
        environment = EnvironmentLight(name="sky", intensity=2.5, cubemap=studio.cubemap)
        scene = g.trimesh.Scene(g.trimesh.creation.box())
        rotation = tf.random_rotation_matrix()
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

        checks = {
            "second and third exports are identical": exports[1] == exports[2],
            "lights round trip": [type(light).__name__ for light in reloaded.lights]
            == ["SpotLight", "EnvironmentLight"],
            "name": found.name == "sky",
            "intensity is exact": found.intensity == 2.5,
            "node pose": np.allclose(reloaded.graph.get("sky")[0], rotation),
            # glTF quaternions are xyzw and trimesh's wxyz
            "rotation": np.allclose(
                tf.quaternion_matrix(np.roll(entry["rotation"], 1)), rotation
            ),
            "cube map within RGBD": (np.abs(found.cubemap - original) <= tolerance).all(),
            "harmonics stored": np.allclose(
                entry["irradianceCoefficients"], spherical_harmonics(environment.cubemap)
            ),
            "harmonics within RGBD": (
                np.abs(spherical_harmonics(found.cubemap) - spherical_harmonics(original))
                <= harmonic_bound * BANDS[:, None]
            ).all(),
            # each level is a weighted mean of the mirror level's texels
            "levels within RGBD": all(
                np.abs(after - before).max() <= tolerance.max()
                for after, before in zip(prefilter(found.cubemap), prefilter(original))
            ),
            "image size": entry["specularImageSize"] == len(original[0]),
            "mip levels halve": [len(level) for level in entry["specularImages"]]
            == [6] * len(prefilter(original))
            and [face.shape for face in faces]
            == [
                (len(original[0]) >> level, len(original[0]) >> level, 4)
                for level in range(len(entry["specularImages"]))
                for _ in range(6)
            ],
            "faces are RGBD flipped": all(
                (np.abs(rgbd_unpack(face)[:, ::-1] - texels) <= tolerance[index]).all()
                for index, (face, texels) in enumerate(zip(faces, original))
            ),
            "the scene uses it": header["scenes"][0]["extensions"][ENVIRONMENT]
            == {"light": 0},
            "extension used": ENVIRONMENT in header["extensionsUsed"],
            "not in metadata": "gltf_extensions" not in reloaded.metadata,
        }

        # packing RGBD what it unpacked gives the same bytes, including
        # radiance exactly at the brightest each alpha holds
        rng = np.random.default_rng(0)
        brightest = 255.0 / np.arange(1, 256, dtype=np.float64)
        radiance = np.vstack(
            (rng.exponential(4.0, size=(4096, 3)), np.outer(brightest, [1.0, 0.5, 0.0]))
        )
        packed = rgbd_pack(radiance)
        unpacked = rgbd_unpack(packed)
        checks["rgbd within bound"] = (
            np.abs(unpacked - radiance) <= bound(radiance)
        ).all()
        checks["rgbd is idempotent"] = np.array_equal(
            rgbd_pack(unpacked.astype(np.float32)), packed
        )

        # the light is its cube map and nothing derived from it, white
        # and posed by its node like every light
        shape = EnvironmentLight(cubemap=np.ones((6, 2, 2, 3)))
        checks["only adds a cube map"] = set(vars(shape)) - set(
            vars(DirectionalLight())
        ) == {"cubemap"}
        checks["only defines __init__"] = {
            name
            for name, value in vars(EnvironmentLight).items()
            if callable(value) or isinstance(value, property)
        } == {"__init__"}
        checks["white"] = (shape.color == 255).all()

        # untouched default lights and the studio a render uses are not
        # the scene's, so they are never exported
        plain = g.trimesh.Scene(g.trimesh.creation.box())
        before = plain.export(file_type="glb")
        render_lights(plain, np.eye(4))
        checks["default lights generated"] = len(plain.lights) == 3
        after = plain.export(file_type="glb")
        checks["default scene has no environment"] = ENVIRONMENT.encode() not in before
        checks["rendering doesn't change the export"] = before == after
        for name, ok in checks.items():
            assert ok, name


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

    checks = {"texels cover the sphere": np.isclose(solid_angle.sum(), 4.0 * np.pi)}
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
    checks["harmonics are exact"] = np.allclose(
        spherical_harmonics(exact.cubemap), coefficients * BANDS[:, None], atol=1e-10
    )
    checks["exact irradiance"] = np.allclose(
        harmonics(normals) @ spherical_harmonics(exact.cubemap),
        irradiance(exact.cubemap),
        atol=1e-3,
    )
    # the studio has sharper light which l=2 harmonics blur a little
    error = relative(
        harmonics(normals) @ spherical_harmonics(studio.cubemap),
        irradiance(studio.cubemap),
    )
    checks["studio irradiance mean"] = error.mean() < 0.015
    checks["studio irradiance max"] = error.max() < 0.05

    # every mip level is the cube map blurred by the spherical Gaussian of
    # its roughness, and blending them is close to a GGX reflection seen
    # head on, where the squared cosine of the half vector is (1 + cosine) / 2
    levels = prefilter(studio.cubemap)
    count = len(levels)
    checks["levels halve"] = [len(level[0]) for level in levels] == [
        shape[1] >> index for index in range(count)
    ]
    for index, level in enumerate(levels[1:], 1):
        sharpness = np.exp(np.polyval(SPECULAR_SHARPNESS, np.log(index / (count - 1))))
        towards = cubemap_directions(len(level[0]))[0].reshape((-1, 3))
        weights = np.exp(sharpness * (towards @ directions.T - 1.0)) * solid_angle
        truth = (weights @ studio.cubemap.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        error = relative(level.reshape((-1, 3)), truth)
        checks[f"level {index} mean"] = error.mean() < 0.003
        checks[f"level {index} max"] = error.max() < 0.01
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
        facing = cubemap_directions(size)[0].reshape((-1, 3)) @ directions.T
        alpha = roughness**4
        weights = np.clip(facing, 0.0, 1.0) * solid_angle
        weights *= alpha / ((1 + facing) / 2 * (alpha - 1) + 1) ** 2
        truth = (weights @ studio.cubemap.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        error = relative(model, truth)
        checks[f"specular {roughness} mean"] = error.mean() < 0.04
        checks[f"specular {roughness} max"] = error.max() < 0.18

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
    checks["equirectangular to faces"] = np.allclose(
        cubemap.reshape((-1, 3)), faces, rtol=5e-3
    )
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
