"""
Run the examples which make scenes, and check that what they make is
correct and survives a GLB round trip.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g

import importlib.util


def load_example(name):
    """Import `examples/{name}.py` without running its `__main__`."""
    path = g.os.path.join(g.dir_current, "..", "examples", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cycloidal():
    """The hero film survives a GLB round trip pose for pose."""
    hero = load_example("animation_cycloidal")
    scene = hero.animate(g.get_scene("cycloidal.3DXML"), fps=4)
    export = scene.export(file_type="glb")
    g.validate_glb(export, "cycloidal.glb")
    reloaded = g.roundtrip(export, "glb")
    once = reloaded.export(file_type="glb")
    twice = g.roundtrip(once, "glb").export(file_type="glb")

    nodes = scene.graph.nodes_geometry
    # a viewer that isn't playing shows the file at rest, which is the first frame
    rest = {n: reloaded.graph[n][0] for n in nodes}
    scene.animate(0.0)
    checks = {
        "rest is the first frame": all(
            g.np.allclose(scene.graph[n][0], rest[n], atol=1e-4) for n in nodes
        ),
        # one glTF animation, as other importers only play the first of
        # several sharing a name
        "one clip": [a["name"] for a in g.glb_chunks(export)[0]["animations"]]
        == ["cycloidal"],
        # the reloaded film exports to a file which loads and exports to
        # the same animation, to float32 rounding
        "fixed point": g.same_animations(once, twice),
        "same keyframes": g.np.array_equal(
            reloaded.animations[0].times,
            g.np.float32(scene.animations[0].times),
        ),
        "same duration": g.np.isclose(reloaded.duration, hero.DURATION),
        "same nodes": set(reloaded.graph.nodes_geometry) == set(nodes),
        # the outlines are lines which keep their color, and the cover
        # stays glass
        "outlines": isinstance(reloaded.geometry["outline"], g.trimesh.path.Path3D)
        and g.np.isclose(
            reloaded.geometry["outline"].length, scene.geometry["outline"].length
        )
        and g.np.array_equal(
            g.np.unique(reloaded.geometry["outline"].colors, axis=0),
            g.np.unique(scene.geometry["outline"].colors, axis=0),
        ),
        "glass": g.np.isclose(
            reloaded.geometry[hero.GLASS].visual.material.transmissionFactor, 0.9
        ),
    }
    # every node of the file is where the example put it at every beat
    for time in [0.0, 2.75, 5.0, 8.0, 12.0, hero.DURATION]:
        scene.animate(time)
        reloaded.animate(time)
        checks[f"pose at {time}"] = all(
            g.np.allclose(scene.graph[n][0], reloaded.graph[n][0], atol=1e-4)
            for n in nodes
        )

    for name, ok in checks.items():
        assert ok, name


def test_pyrender_scene():
    """
    pyrender's example scene keeps its standard lights and camera
    through a GLB, and the file renders exactly what its rays hit.
    """
    example = load_example("pyrender_scene")
    # the same files as pyrender's but made from the models here
    stand_ins = {
        "drill.obj": g.trimesh.creation.capsule(height=0.1, radius=0.03),
        "wood.obj": g.trimesh.creation.box(extents=[0.6, 0.6, 0.01]),
        "WaterBottle.glb": g.trimesh.creation.cylinder(radius=0.04, height=0.2),
    }
    with g.tempfile.TemporaryDirectory() as directory:
        for name in ["fuze.obj", "fuze.obj.mtl", "fuze uv.jpg"]:
            g.shutil.copy(g.os.path.join(g.dir_models, name), directory)
        for name, mesh in stand_ins.items():
            mesh.export(g.os.path.join(directory, name))
        scene = example.build(g.trimesh.resolvers.FilePathResolver(directory))
    export = scene.export(file_type="glb")
    g.validate_glb(export, "pyrender_scene.glb")
    reloaded = g.roundtrip(export, "glb")

    # what the file renders is exactly what its camera's rays hit
    arrays = g.render(
        reloaded, resolution=[160, 120], include_depth=True, include_segmentation=True
    )
    index = arrays.segmentation
    checks = {
        # its ambient light is an `EXT_lights_image_based` environment
        "lights": [(type(light), light.intensity) for light in reloaded.lights]
        == [(type(light), light.intensity) for light in scene.lights]
        == [
            (getattr(g.trimesh.scene.lighting, kind), intensity)
            for kind, intensity in [
                ("DirectionalLight", 1.0),
                ("SpotLight", 10.0),
                ("EnvironmentLight", 1.0),
            ]
        ],
        # pyrender's spot light cone
        "cone": g.np.allclose(
            [reloaded.lights[1].innerConeAngle, reloaded.lights[1].outerConeAngle],
            [g.np.pi / 16.0, g.np.pi / 6.0],
        ),
        "light poses": all(
            g.np.allclose(reloaded.graph[light.name][0], example.CAMERA_POSE)
            for light in reloaded.lights
        ),
        "camera pose": g.np.allclose(reloaded.camera_transform, example.CAMERA_POSE),
        "fov": g.np.allclose(reloaded.camera.fov, scene.camera.fov),
        # render settings are keywords rather than scene content
        "no settings": "render" not in reloaded.metadata,
        "every node drawn": len(g.np.unique(index[index >= 0]))
        == len(reloaded.graph.nodes_geometry),
    }
    checks.update(g.ray_checks(reloaded, arrays))
    for name, ok in checks.items():
        assert ok, name
