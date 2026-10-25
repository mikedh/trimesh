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
        from trimesh.scene.metadata import LUMINANCE, Environment, RenderSettings

        s = g.get_mesh("duck.dae")
        checks = {}
        before = s.export(file_type="glb")
        # directional lights need no geometry, so even an empty scene is lit
        for name, scene in {"duck": s, "empty": g.trimesh.Scene()}.items():
            nodes, metadata = set(scene.graph.nodes), g.copy.deepcopy(scene.metadata)
            lights = scene.lights
            kinds = [type(light).__name__ for light in lights]
            # in glTF units a white surface facing a light reflects its
            # linear color times its intensity over pi
            reflected = [
                g.np.dot(light.color[:3] / 255.0, LUMINANCE) * light.intensity / g.np.pi
                for light in lights
            ]
            current = {
                # a key and a rim light, which a white surface facing them
                # reflects 0.58 and 0.28 of as luminance at any scale
                "two directional": kinds == ["DirectionalLight"] * 2,
                "reflected": g.np.allclose(reflected, [0.58, 0.28]),
                # asking for the default lights stores nothing
                "not stored": not scene.has_lights and not scene.has_camera,
                "not in the graph": not any(
                    light.name in scene.graph.nodes for light in lights
                ),
                "graph unchanged": set(scene.graph.nodes) == nodes,
                "metadata unchanged": scene.metadata == metadata,
                # so editing them raises rather than doing nothing
                "default lights are a tuple": isinstance(lights, tuple),
            }
            checks.update({f"{name} {check}": ok for check, ok in current.items()})
        checks["export unchanged by reading lights"] = s.export(file_type="glb") == before
        checks["camera"] = isinstance(s.camera, g.trimesh.scene.cameras.Camera)

        # the key is above and to the right of the camera and the rim
        # is behind, measured in the camera's frame
        camera = g.trimesh.transformations.euler_matrix(0.3, 0.8, 0.0)
        camera[:3, 3] = [1, 2, 3]
        s.camera_transform = camera
        studio = RenderSettings().resolve(s, camera)
        towards = studio.transforms[:, :3, 2] @ camera[:3, :3]
        angle = g.np.degrees(g.np.arcsin(towards @ camera[1, :3]))
        checks["key elevation"], checks["rim elevation"] = g.np.isclose(angle, [35, 65])
        checks["key in front and right"] = towards[0, 2] > 0.3 and towards[0, 0] > 0.3
        checks["rim behind"] = towards[1, 2] < -0.3
        # a warm key and a cool rim under a slightly warm sky, with the
        # softbox on the key's side warm and the one opposite cool, which
        # are as bright as the white they're tinted from, so the warm
        # studio is lit as brightly as a white one
        environment = studio.environment
        key, rim = (light.color[:3] for light in studio.lights)
        overhead, near, far = (box.color for box in environment.softboxes)
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
        resolved = RenderSettings().resolve(s, moved).transforms
        checks["world fixed"] = not g.np.allclose(resolved, studio.transforms)
        # a camera looking straight down still gets a whole floor
        overhead = RenderSettings(environment=Environment(up=[0, 0, 1]))
        down = overhead.resolve(s, g.np.eye(4))
        checks["floor from straight above"] = g.np.ptp(down.ground, axis=0)[:2].min() > 0
        # explicit lights are used as they are
        s.lights = [g.trimesh.scene.lighting.PointLight(name="bulb")]
        s.graph["bulb"] = g.np.eye(4)
        explicit = [light.name for light in RenderSettings().resolve(s).lights]
        checks["explicit lights"] = explicit == ["bulb"]
        checks["stored lights are a list"] = isinstance(s.lights, list)
        for name, ok in checks.items():
            assert ok, name

    def test_settings(self):
        from trimesh.scene.metadata import Environment, RenderSettings, Softbox

        box = Softbox(direction=[1, 0, 0], sharpness=8, color=[2, 2, 2])
        sky, up = g.np.array([0.5, 0.6, 0.7]), g.np.array([0, 0, 1])
        environment = Environment(up=up, sky=sky, ground=[0.1] * 3, softboxes=[box])
        custom = RenderSettings(environment=environment, ground_shadow=False)
        stored = g.json.loads(g.json.dumps(custom.to_metadata()))
        checks = {
            "defaults store nothing": RenderSettings().to_metadata() == {},
            "custom survives json": RenderSettings.from_metadata(stored) == custom,
            "arrays become floats": custom.environment.sky == [0.5, 0.6, 0.7],
        }
        for name, metadata in [("missing", {}), ("garbage", {"render": 7})]:
            default = RenderSettings.from_metadata(metadata) == RenderSettings()
            checks[f"{name} is the default"] = default
        for name, ok in checks.items():
            assert ok, name


def test_environment():
    # the environment's spherical Gaussian blurs match brute force
    # integrals over the sphere of what diffuse and GGX surfaces reflect
    from trimesh.scene import metadata

    np = g.np
    # evenly spread directions over the sphere for the integrals
    index = np.arange(200000) + 0.5
    azimuth, polar = np.pi * (1 + 5**0.5) * index, np.arccos(1 - index / 1e5)
    directions = g.trimesh.util.spherical_to_vector(np.column_stack((azimuth, polar)))
    normals = g.trimesh.util.unitize(np.random.default_rng(0).normal(size=(48, 3)))
    cosine = normals @ directions.T

    def errors(environment, roughness=None):
        # relative error of how bright every normal looks against the brute
        # force integral of the environment's definition in every direction,
        # which is the same for any tint of the same luminance
        sky, ground = np.array(environment.sky), np.array(environment.ground)
        light = ground + np.outer(0.5 + 0.5 * directions @ environment.up, sky - ground)
        for box in environment.softboxes:
            lobe = np.exp(box.sharpness * (directions @ box.direction - 1.0))
            light += np.outer(lobe, box.color)
        weights, blur = np.clip(cosine, 0.0, 1.0), metadata.DIFFUSE_SHARPNESS
        if roughness is not None:
            # the GGX lobe around the normal seen straight on, where the
            # squared cosine of the half vector is (1 + cosine) / 2
            alpha = roughness**4
            weights *= alpha / (np.pi * ((1 + cosine) / 2 * (alpha - 1) + 1) ** 2)
            blur = metadata.specular_sharpness(roughness)
        truth = (weights @ light) / weights.sum(axis=1)[:, None]
        model = environment.light(normals, blur)
        luminance = metadata.LUMINANCE
        return np.abs((model - truth) @ luminance) / (truth @ luminance)

    # up, sky and ground
    gradient = metadata.Environment([0, 0, 1], [1, 0.8, 0.6], [0.1, 0.2, 0.3])
    studio = metadata.RenderSettings().resolve(g.trimesh.Scene()).environment
    diffuse = errors(studio)
    checks = {
        "gradient diffuse is exact": errors(gradient).max() < 0.01,
        "studio diffuse mean": diffuse.mean() < 0.04,
        "studio diffuse max": diffuse.max() < 0.13,
    }
    for roughness in [0.1, 0.3, 0.6, 1.0]:
        specular = errors(studio, roughness)
        checks[f"specular {roughness} mean"] = specular.mean() < 0.04
        checks[f"specular {roughness} max"] = specular.max() < 0.17
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
