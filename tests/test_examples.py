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
    """The animated drive is one clip which survives a GLB round trip."""
    hero = load_example("animation_cycloidal")
    scene = hero.animate(g.get_scene("cycloidal.3DXML"), fps=4)
    header = g.glb_chunks(scene.export(file_type="glb"))[0]
    assert [a["name"] for a in header["animations"]] == ["cycloidal"], "one clip"
    assert g.np.isclose(scene.duration, hero.DURATION), "duration"
    g.assert_glb_roundtrip(scene, "cycloidal")


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
    assert (
        [(type(light), light.intensity) for light in reloaded.lights]
        == [(type(light), light.intensity) for light in scene.lights]
        == [
            (getattr(g.trimesh.scene.lighting, kind), intensity)
            for kind, intensity in [
                ("DirectionalLight", 1.0),
                ("SpotLight", 10.0),
                ("EnvironmentLight", 1.0),
            ]
        ]
    ), "lights"
    assert g.np.allclose(
        [reloaded.lights[1].innerConeAngle, reloaded.lights[1].outerConeAngle],
        [g.np.pi / 16.0, g.np.pi / 6.0],
    ), "cone"
    assert all(
        g.np.allclose(reloaded.graph[light.name][0], example.CAMERA_POSE)
        for light in reloaded.lights
    ), "light poses"
    assert g.np.allclose(reloaded.camera_transform, example.CAMERA_POSE), "camera pose"
    assert g.np.allclose(reloaded.camera.fov, scene.camera.fov), "fov"
    assert "render" not in reloaded.metadata, "no settings"
    assert len(g.np.unique(index[index >= 0])) == len(reloaded.graph.nodes_geometry), (
        "every node drawn"
    )
    g.assert_rays(reloaded, arrays)
