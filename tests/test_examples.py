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


def overlaps(scene, target, others, height):
    """Area of `target` overlapping each instance of `others` posed, on a Z plane."""
    from shapely.ops import unary_union

    sections = {}
    for node in scene.graph.nodes_geometry:
        part = str(node).split("#")[0]
        if part != target and part not in others:
            continue
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy().apply_transform(transform)
        section = mesh.section(plane_origin=[0, 0, height], plane_normal=[0, 0, 1])
        if section is not None:
            polygons = section.to_2D(to_2D=g.np.eye(4))[0].polygons_full
            sections.setdefault(part == target, []).append(unary_union(polygons))
    checked = unary_union(sections[True])
    return g.np.array([checked.intersection(other).area for other in sections[False]])


def test_cycloidal():
    """
    The hero film survives a GLB round trip pose for pose, runs the drive
    without the disc going into the rollers or bearings, and the file
    draws what the storyboard says when.
    """
    hero = load_example("animation_cycloidal")
    scene = hero.animate(g.get_scene("cycloidal.3DXML"), fps=4)
    export = scene.export(file_type="glb")
    g.validate_glb(export, "cycloidal.glb")
    reloaded = g.roundtrip(export, "glb")

    nodes = scene.graph.nodes_geometry
    # a viewer that isn't playing shows the file at rest, which is the first
    # frame: only the outlines, as the drive and the cloud are scaled to nothing
    rest = {n: reloaded.graph[n][0] for n in nodes}
    scene.animate(0.0)
    hidden = [n for n in nodes if not str(n).startswith("outline_")]
    checks = {
        "rest is the first frame": all(
            g.np.allclose(scene.graph[n][0], rest[n], atol=1e-4) for n in nodes
        ),
        "hidden at rest": all(g.np.allclose(rest[n][:3, :3], 0.0) for n in hidden),
        # one glTF animation, as other importers only play the first of
        # several sharing a name
        "one clip": [a["name"] for a in g.glb_chunks(export)[0]["animations"]]
        == ["cycloidal"],
        "same edges": sum(len(a.edges) for a in reloaded.animations)
        == sum(len(a.edges) for a in scene.animations),
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
    # the cloud bursts out from the drive faster at every keyframe, to
    # many times the drive's size before it is gone
    keys = g.np.arange(hero.BURST[0], hero.BURST[1], 0.25)[1:]
    sizes = []
    for time in keys:
        reloaded.animate(time)
        sizes.append(g.np.linalg.norm(reloaded.graph["cloud"][0][:3, 0]))
    checks["cloud accelerates"] = (g.np.diff(sizes, 2) > 0).all()
    checks["cloud flies far"] = sizes[-1] > 50.0

    # every node of the file is where the example put it at every beat
    for time in [0.0, 2.75, 5.0, 8.0, 12.0, hero.DURATION]:
        scene.animate(time)
        reloaded.animate(time)
        checks[f"pose at {time}"] = all(
            g.np.allclose(scene.graph[n][0], reloaded.graph[n][0], atol=1e-4)
            for n in nodes
        )

    # while the crank turns through every phase the disc gets no further
    # into the rollers or the bearings on its journals than it is when
    # fully grown at a whole crank turn, which is the modeled arrangement
    # and is a fraction of a square millimeter
    parts = {"rollers": {"bushing_roller"}, "bearings": {"vxb-6800-2rs"}}
    grown = hero.GROW[1]
    reloaded.animate(grown)
    full = {k: overlaps(reloaded, "disc_cam_A", v, -14.5) for k, v in parts.items()}
    checks["rest rollers touch"] = 0.0 < full["rollers"].max() < 0.1
    # keyframes while the drive runs alone, which aren't interpolated
    for time in grown + g.np.arange(8) / 4.0:
        reloaded.animate(time)
        area = {k: overlaps(reloaded, "disc_cam_A", v, -14.5) for k, v in parts.items()}
        checks[f"rollers at {time}"] = area["rollers"].max() < 0.1
        checks[f"bearings at {time}"] = g.np.isclose(
            area["bearings"].sum(), full["bearings"].sum(), rtol=0.01
        )

    # the file as any viewer draws it: overlays scaled to zero by the film
    # aren't drawn at all, and appear when the storyboard says they do
    names = g.np.array(reloaded.graph.nodes_geometry)
    cloud = names == "cloud"
    outlines = g.np.char.startswith(names.astype(str), "outline_")
    masks = {
        "cloud": cloud,
        "outlines": outlines,
        "drive": ~(cloud | outlines),
        "glass": g.np.array([reloaded.graph[n][1] == hero.GLASS for n in names]),
    }
    # {time: (name, {mask: fewest nodes drawn}, masks hidden, fewest rows of outlines)}
    storyboard = {
        0.0: (
            "outlines alone filling the frame",
            {"outlines": 1},
            ["cloud", "drive"],
            0.9,
        ),
        2.75: ("outlines fall as the drive grows", {"outlines": 1, "drive": 11}, [], 0),
        5.0: (
            "drive runs alone under glass",
            {"drive": 11, "glass": 1},
            ["cloud", "outlines"],
            0,
        ),
        8.0: ("cloud bursts from the drive", {"cloud": cloud.sum(), "drive": 11}, [], 0),
        12.0: ("ends on the drive alone", {"drive": 11}, ["cloud", "outlines"], 0),
    }
    for time, (name, drawn, hidden, span) in storyboard.items():
        reloaded.animate(time)
        # points as small as the pixels, which are few
        index = g.render(
            reloaded, resolution=[160, 90], point_size=1.0, include_segmentation=True
        ).segmentation
        seen = g.np.zeros(len(names), dtype=bool)
        seen[index[index >= 0]] = True
        # how much of the height the outlines cover, top row to bottom row
        rows = g.np.flatnonzero(g.np.isin(index, g.np.flatnonzero(outlines)).any(axis=1))
        # scaled to zero is hidden, which is how glTF hides things
        checks[name] = (
            all(seen[masks[k]].sum() >= count for k, count in drawn.items())
            and not any(seen[masks[k]].any() for k in hidden)
            and (rows.max(initial=0) - rows.min(initial=0) + 1) / len(index) > span
        )

    for name, ok in checks.items():
        assert ok, name


def test_pyrender_scene():
    """
    pyrender's example scene keeps its lights, camera and settings
    through a GLB, and the file renders exactly what its rays hit.
    """
    from scipy import ndimage

    from trimesh.scene.metadata import RenderSettings

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
    _, depth, index = g.render(
        reloaded, resolution=[160, 120], include_depth=True, include_segmentation=True
    )
    reference_depth, reference = g.camera_ray_reference(reloaded)
    interior = ndimage.maximum_filter(reference, 3) == ndimage.minimum_filter(
        reference, 3
    )
    covered = interior & (reference >= 0)
    checks = {
        "lights": [(type(light), light.intensity) for light in reloaded.lights]
        == [(type(light), light.intensity) for light in scene.lights],
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
        # the example changes pyrender's settings away from the defaults
        "settings": RenderSettings.from_metadata(reloaded.metadata)
        == RenderSettings.from_metadata(scene.metadata)
        != RenderSettings(),
        "segmentation is rays": (index == reference)[interior].all(),
        "depth is rays": g.np.allclose(
            depth[covered], reference_depth[covered], rtol=1e-4
        ),
        "every node drawn": len(g.np.unique(index[index >= 0]))
        == len(reloaded.graph.nodes_geometry),
    }
    for name, ok in checks.items():
        assert ok, name
