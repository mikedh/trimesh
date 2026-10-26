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
        from trimesh.scene.lighting import LUMINANCE, autolight
        from trimesh.viewer.conversion import render_lights, shadow_floor

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
                "kinds": kinds
                == ["DirectionalLight", "DirectionalLight", "EnvironmentLight"],
                "reflected": g.np.allclose(reflected, [0.58, 0.28]),
                # the default lights are generated once into the scene
                "a list": isinstance(lights, list),
                "stored": scene.has_lights and scene.lights is lights,
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
        towards = transforms[:2, :3, 2] @ camera[:3, :3]
        angle = g.np.degrees(g.np.arcsin(towards @ camera[1, :3]))
        checks["key elevation"], checks["rim elevation"] = g.np.isclose(angle, [35, 65])
        checks["key in front and right"] = towards[0, 2] > 0.3 and towards[0, 0] > 0.3
        checks["rim behind"] = towards[1, 2] < -0.3
        # the environment's frame is a rotation whose +Y is the world axis
        # nearest the camera's up and whose +Z is level towards the camera
        frame = transforms[2]
        checks["environment frame"] = (
            g.np.allclose(frame[:3, :3] @ frame[:3, :3].T, g.np.eye(3))
            and g.np.isclose(g.np.linalg.det(frame[:3, :3]), 1.0)
            and g.np.allclose(frame[:3, 1], [0, 1, 0])
            and frame[:3, 2] @ camera[:3, 2] > 0.5
        )
        # a warm key and a cool rim under a slightly warm sky, with the
        # softbox on the key's side warm and the one opposite cool, which
        # are as bright as the white they're tinted from, so the warm
        # studio is lit as brightly as a white one
        environment = lights[2]
        key, rim = (light.color[:3] for light in lights[:2])
        overhead, near, far = environment.softboxes["color"]
        for name, color in {"key": key, "near softbox": near}.items():
            checks[f"{name} is warm"] = color[0] > color[1] > color[2]
        for name, color in {"rim": rim, "far softbox": far}.items():
            checks[f"{name} is cool"] = color[0] < color[1] < color[2]
        checks["sky is warm"] = environment.sky[0] > environment.sky[2]
        checks["environment luminance"] = g.np.allclose(
            [environment.sky, environment.ground, overhead, near, far] @ LUMINANCE,
            [0.31, 0.096, 0.35, 2.0, 1.0],
        )
        # the lights are fixed in the world where the camera started
        moved = g.trimesh.transformations.rotation_matrix(1.0, [0, 1, 0]) @ camera
        checks["world fixed"] = not g.np.allclose(autolight(s, moved)[1], transforms)
        # an environment whose up is +Z still gets a whole floor under it
        overhead = g.trimesh.transformations.rotation_matrix(g.np.pi / 2, [1, 0, 0])
        floor = shadow_floor(s, overhead)
        checks["floor faces up"] = g.np.allclose(floor[:, 2], floor[0, 2]) and (
            g.np.ptp(floor, axis=0)[:2].min() > 0
        )
        # explicit lights are used as they are with the default environment
        s.lights = [g.trimesh.scene.lighting.PointLight(name="bulb")]
        s.graph["bulb"] = g.np.eye(4)
        explicit = [type(light).__name__ for light in render_lights(s, camera)[0]]
        checks["explicit lights"] = explicit == ["PointLight", "EnvironmentLight"]
        checks["stored lights are a list"] = isinstance(s.lights, list)
        for name, ok in checks.items():
            assert ok, name

    def test_environment_gltf(self):
        # an environment light round trips exactly through a GLB, which
        # stays spec clean as `KHR_lights_punctual` has no such light
        from trimesh.scene.lighting import EnvironmentLight, SpotLight

        scene = g.trimesh.Scene(g.trimesh.creation.box())
        environment = EnvironmentLight(
            name="sky",
            color=[255, 200, 100, 255],
            intensity=1.7,
            sky=[0.5, 0.6, 0.7],
            ground=[0.1, 0.05, 0.0],
            softboxes=list(
                zip(
                    g.trimesh.util.unitize(g.random((3, 3)) - 0.5),
                    [4.0, 30.0, 1.0 / 3.0],
                    g.random((3, 3)) * 5.0,
                )
            ),
        )
        scene.lights = [SpotLight(name="spot"), environment]
        pose = g.trimesh.transformations.random_rotation_matrix()
        scene.graph["sky"] = pose
        scene.graph["spot"] = g.np.eye(4)
        export = scene.export(file_type="glb")
        g.validate_glb(export, "environment light")
        header = g.glb_chunks(export)[0]
        reloaded = g.roundtrip(export, "glb")
        loaded = {type(light).__name__: light for light in reloaded.lights}
        after = loaded.get("EnvironmentLight")
        checks = {
            "kinds": sorted(loaded) == ["EnvironmentLight", "SpotLight"],
            "only punctual in the extension": [
                light["type"]
                for light in header["extensions"]["KHR_lights_punctual"]["lights"]
            ]
            == ["spot"],
            "name": after.name == "sky",
            "pose": g.np.allclose(reloaded.graph[after.name][0], pose),
            "intensity": after.intensity == environment.intensity,
            **{
                key: g.np.array_equal(getattr(after, key), getattr(environment, key))
                for key in ["color", "sky", "ground", "softboxes"]
            },
        }
        # with only an environment the extension isn't written at all
        scene.lights = [environment]
        alone = g.glb_chunks(scene.export(file_type="glb"))[0]
        checks["no empty extension"] = "KHR_lights_punctual" not in alone.get(
            "extensionsUsed", []
        ) and "KHR_lights_punctual" not in alone.get("extensions", {})
        for name, ok in checks.items():
            assert ok, name


def test_environment():
    # the environment's spherical Gaussian blurs match brute force
    # integrals over the sphere of what diffuse and GGX surfaces reflect
    from trimesh.scene.lighting import LUMINANCE, EnvironmentLight, autolight
    from trimesh.viewer.conversion import DIFFUSE_SHARPNESS, environment_light

    np = g.np
    # evenly spread directions over the sphere for the integrals
    index = np.arange(200000) + 0.5
    azimuth, polar = np.pi * (1 + 5**0.5) * index, np.arccos(1 - index / 1e5)
    directions = g.trimesh.util.spherical_to_vector(np.column_stack((azimuth, polar)))
    normals = g.trimesh.util.unitize(np.random.default_rng(0).normal(size=(48, 3)))
    cosine = normals @ directions.T
    # how wide a GGX reflection is, which the viewer's shader fits
    specular = (0.131, 1.0381, -0.7256, 1.0728)

    def errors(environment, transform, roughness=None):
        # relative error of how bright every normal looks against the brute
        # force integral of the environment's definition in every direction,
        # which is the same for any tint of the same luminance
        rotation = transform[:3, :3]
        light = environment.ground + np.outer(
            0.5 + 0.5 * directions @ rotation[:, 1], environment.sky - environment.ground
        )
        softboxes = environment.softboxes
        boxes = softboxes["direction"] @ rotation.T
        lobes = np.exp(softboxes["sharpness"] * (directions @ boxes.T - 1.0))
        light += lobes @ softboxes["color"]
        weights, blur = np.clip(cosine, 0.0, 1.0), DIFFUSE_SHARPNESS
        if roughness is not None:
            # the GGX lobe around the normal seen straight on, where the
            # squared cosine of the half vector is (1 + cosine) / 2
            alpha = roughness**4
            weights *= alpha / (np.pi * ((1 + cosine) / 2 * (alpha - 1) + 1) ** 2)
            blur = np.exp(np.polyval(specular, np.log(roughness)))
        truth = (weights @ light) / weights.sum(axis=1)[:, None]
        model = environment_light(environment, transform, normals, blur)
        return np.abs((model - truth) @ LUMINANCE) / (truth @ LUMINANCE)

    # sky and ground, with up along +Z
    gradient = EnvironmentLight(sky=[1, 0.8, 0.6], ground=[0.1, 0.2, 0.3])
    up = g.trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    lights, transforms = autolight(g.trimesh.Scene())
    diffuse = errors(lights[-1], transforms[-1])
    checks = {
        "gradient diffuse is exact": errors(gradient, up).max() < 0.01,
        "studio diffuse mean": diffuse.mean() < 0.04,
        "studio diffuse max": diffuse.max() < 0.13,
    }
    for roughness in [0.1, 0.3, 0.6, 1.0]:
        specular_error = errors(lights[-1], transforms[-1], roughness)
        checks[f"specular {roughness} mean"] = specular_error.mean() < 0.04
        checks[f"specular {roughness} max"] = specular_error.max() < 0.17
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
