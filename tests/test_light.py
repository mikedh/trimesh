try:
    from . import generic as g
except BaseException:
    import generic as g


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
            LUMINANCE,
            autolight,
            cubemap_directions,
            cubemap_sample,
            studio_environment,
        )
        from trimesh.viewer.conversion import render_lights

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
                # reflects 0.58 and 0.28 of as luminance at any scale
                # only standard `KHR_lights_punctual` lights are stored
                "kinds": kinds == ["DirectionalLight", "DirectionalLight"],
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
        environment, frame = studio_environment(s, camera)
        towards = transforms[:, :3, 2] @ camera[:3, :3]
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
        key, rim = (light.color[:3] for light in lights)
        sky, near, far = cubemap_sample(environment.cubemap, softboxes)
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
        # an environment light is only for the viewer: a GLB with one is
        # byte identical to one without it, and loading finds none
        from trimesh.scene.lighting import EnvironmentLight, SpotLight, studio_cubemap

        scene = g.trimesh.Scene(g.trimesh.creation.box())
        scene.graph["sky"] = g.trimesh.transformations.random_rotation_matrix()
        scene.graph["spot"] = g.np.eye(4)
        environment = EnvironmentLight(
            name="sky",
            cubemap=studio_cubemap([0.5, 0.6, 0.7], [0, 0, 0], [([0, 1, 0], 4, [1] * 3)]),
        )
        exports = {}
        for name, lights in {
            "spot": [SpotLight(name="spot")],
            "spot and sky": [SpotLight(name="spot"), environment],
            "none": [],
            "sky": [environment],
        }.items():
            scene.lights = lights
            exports[name] = scene.export(file_type="glb")
        g.validate_glb(exports["spot and sky"], "environment light")
        reloaded = g.roundtrip(exports["spot and sky"], "glb")
        checks = {
            "spot identical": exports["spot and sky"] == exports["spot"],
            "alone identical": exports["sky"] == exports["none"],
            "loads only the spot": [type(light).__name__ for light in reloaded.lights]
            == ["SpotLight"],
            "no extension alone": b"KHR_lights_punctual" not in exports["sky"],
        }
        for name, ok in checks.items():
            assert ok, name


def test_environment():
    # an environment's harmonics, the mip levels the viewer reflects and
    # equirectangular images match brute force over every texel
    from trimesh.scene.lighting import (
        LUMINANCE,
        EnvironmentLight,
        cubemap_directions,
        cubemap_sample,
        spherical_harmonics,
        studio_environment,
    )
    from trimesh.viewer.conversion import SPECULAR_SHARPNESS, environment_levels

    np = g.np
    rng = np.random.default_rng(0)
    studio = studio_environment(g.trimesh.Scene(), np.eye(4))[0]
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
    # times the clamped cosine's pi, 2 pi / 3 and pi / 4 for each band
    coefficients = rng.normal(size=(9, 3))
    exact = EnvironmentLight(
        cubemap=(spherical_harmonics(directions) @ coefficients).reshape(shape)
    )
    bands = np.pi * np.array([1, 2, 2, 2, 0.75, 0.75, 0.75, 0.75, 0.75]) / [1, *[3] * 8]
    checks["harmonics are exact"] = np.allclose(
        exact.irradiance, coefficients * bands[:, None], atol=1e-10
    )
    checks["exact irradiance"] = np.allclose(
        spherical_harmonics(normals) @ exact.irradiance,
        irradiance(exact.cubemap),
        atol=1e-3,
    )
    # the studio has sharper light which l=2 harmonics blur a little
    error = relative(
        spherical_harmonics(normals) @ studio.irradiance, irradiance(studio.cubemap)
    )
    checks["studio irradiance mean"] = error.mean() < 0.015
    checks["studio irradiance max"] = error.max() < 0.05

    # every mip level is the cube map blurred by the spherical Gaussian of
    # its roughness, and blending them is close to a GGX reflection seen
    # head on, where the squared cosine of the half vector is (1 + cosine) / 2
    levels = environment_levels(studio.cubemap)
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
        alpha = roughness**4
        weights = np.clip(cosine, 0.0, 1.0) * solid_angle
        weights *= alpha / ((1 + cosine) / 2 * (alpha - 1) + 1) ** 2
        truth = (weights @ studio.cubemap.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        level = roughness * (count - 1)
        low, fraction = int(level), level % 1.0
        model = (1 - fraction) * cubemap_sample(levels[low], normals)
        model += fraction * cubemap_sample(levels[min(low + 1, count - 1)], normals)
        error = relative(model, truth)
        checks[f"specular {roughness} mean"] = error.mean() < 0.04
        checks[f"specular {roughness} max"] = error.max() < 0.18

    # an equirectangular image's top row is up and `atan2(z, x)` goes
    # across it from -pi, and of a smooth environment it round trips
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
    checks["equirectangular round trip"] = np.allclose(
        cubemap_sample(cubemap, pixels), image, rtol=5e-3
    )
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
