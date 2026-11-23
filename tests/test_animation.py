try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.animation import INTERPOLATIONS, Animation, Skin

tf = g.trimesh.transformations


def transforms(count, seed=0, rigid=False):
    """An array of `count` random transforms, mirrored and stretched unless `rigid`."""
    return g.np.array(list(g.random_transforms(count, 20.0, seed, rigid)))


def test_sample():
    """
    Every node samples as the glTF spec samples its parts, or as the textbook
    monotone cubic Hermite through them, with quaternions in either hemisphere.
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    np = g.np
    rng = np.random.default_rng(0)
    keys, count = 7, 5
    times = np.sort(rng.uniform(0.0, 4.0, keys))
    # every node stretched and the first mirrored, quaternions `x, y, z, w`
    translation = rng.normal(size=(keys, count, 3)) * 5.0
    rotation = rng.normal(size=(keys, count, 4))
    rotation /= np.linalg.norm(rotation, axis=2)[..., None]
    scale = rng.uniform(0.2, 3.0, size=(keys, count, 3))
    scale[:, 0, 0] *= -1.0

    def compose(translation, quaternion, scale):
        """Matrices which scale, rotate by `x, y, z, w` quaternions, then translate."""
        stack = translation.shape[:-1]
        result = np.tile(np.eye(4), (*stack, 1, 1))
        turn = Rotation.from_quat(quaternion.reshape((-1, 4))).as_matrix()
        result[..., :3, :3] = turn.reshape((*stack, 3, 3)) * scale[..., None, :]
        result[..., :3, 3] = translation
        return result

    middles = (times[1:] + times[:-1]) / 2
    queries = np.array([-1.0, *times, *middles, *np.linspace(-0.5, 4.5, 23)])
    clamped = np.clip(queries, times[0], times[-1])
    upper = np.clip(np.searchsorted(times, clamped), 1, keys - 1)
    lower = upper - 1

    def hermite(part):
        """The textbook cubic Hermite spline with Fritsch-Butland tangents."""
        slope = np.diff(part, axis=0) / np.diff(times)[:, None, None]
        a, b = slope[:-1], slope[1:]
        with np.errstate(all="ignore"):
            middle = np.where(a * b > 0, 2 * a * b / (a + b), 0.0)
        tangent = np.concatenate((slope[:1], middle, slope[-1:]))
        h = (times[upper] - times[lower])[:, None, None]
        s = (clamped - times[lower])[:, None, None] / h
        return (
            (2 * s**3 - 3 * s**2 + 1) * part[lower]
            + (s**3 - 2 * s**2 + s) * h * tangent[lower]
            + (-2 * s**3 + 3 * s**2) * part[upper]
            + (s**3 - s**2) * h * tangent[upper]
        )

    # the spline runs through quaternions in the hemisphere of the last
    flips = (rotation[1:] * rotation[:-1]).sum(axis=2) < 0
    turns = rotation.copy()
    turns[1:] *= np.cumprod(np.where(flips, -1, 1), axis=0)[..., None]
    held = np.clip(np.searchsorted(times, clamped, side="right") - 1, 0, keys - 1)
    slerped = [
        Slerp(times, Rotation.from_quat(rotation[:, node]))(clamped).as_quat()
        for node in range(count)
    ]
    expected = {
        "step": compose(translation[held], rotation[held], scale[held]),
        "linear": compose(
            interp1d(times, translation, axis=0)(clamped),
            np.stack(slerped, axis=1),
            interp1d(times, scale, axis=0)(clamped),
        ),
        "cubic": compose(hermite(translation), hermite(turns), hermite(scale)),
    }
    matrices = compose(translation, rotation, scale)
    nodes = [f"node_{index}" for index in range(count)]
    for interpolation in INTERPOLATIONS:
        animation = Animation(
            name=interpolation,
            nodes=nodes,
            times=times,
            matrices=matrices.copy(),
            interpolation=interpolation,
        )
        sampled = np.array([animation.interpolate(time) for time in queries])
        assert animation.duration == times[-1], f"{interpolation} duration"
        assert np.allclose(sampled[1 : keys + 1], matrices, atol=1e-12), interpolation
        assert np.allclose(sampled, expected[interpolation]), f"{interpolation} samples"
        # editing the keyframes in place moves what is sampled
        animation.trs[..., :3] += [1.0, 2.0, 3.0]
        moved = np.array([animation.interpolate(time) for time in queries])
        assert np.allclose(moved[..., :3, 3], sampled[..., :3, 3] + [1, 2, 3])

    # one keyframe holds forever, a repeated time is finite, bad input raises
    single = Animation(name="single", nodes=nodes, times=[2.0], matrices=matrices[:1])
    assert all(np.allclose(single.interpolate(t), matrices[0]) for t in queries)
    repeated = Animation(
        name="repeated", nodes=nodes[:1], times=[0.0, 0.0, 1.0], matrices=matrices[:3, :1]
    )
    assert all(np.isfinite(repeated.interpolate(t)).all() for t in queries)
    for kwargs in (
        {"times": times[::-1], "matrices": matrices},
        {"times": times, "matrices": matrices[:, :2]},
        {"times": times, "matrices": matrices, "interpolation": "LINEAR"},
    ):
        with g.pytest.raises(ValueError):
            Animation(name="bad", nodes=nodes, **kwargs)


def posed_dump(scene, time):
    """The {node: world vertices} of a copy of `scene` animated to `time`."""
    scene = scene.copy()
    scene.animate(time)
    return {mesh.metadata["node"]: mesh.vertices for mesh in scene.dump()}


def assert_operations(scene, times, label):
    """
    Copying, scaling and appending a scene keep every node and every posed
    vertex at every time, and copies share nothing with the original.
    """
    truth = {time: posed_dump(scene, time) for time in times}
    stretch = g.np.diag([2.5, 2.5, 2.5, 1.0])
    for factor in [2.5, [2.5] * 3]:
        scaled = scene.scaled(factor)
        # not rebuilt flat, which would drop the nodes animations drive
        assert scaled.graph.transforms.parents == scene.graph.transforms.parents, label
        # every world transform is conjugated by the scale, rotations untouched
        assert all(
            g.np.allclose(
                scaled.graph[n][0], stretch @ scene.graph[n][0] @ g.np.linalg.inv(stretch)
            )
            for n in scene.graph.nodes
        ), f"{label} scaled {factor} conjugates"
        for time in times:
            assert all(
                g.np.allclose(vertices, truth[time][node] * 2.5, atol=1e-9)
                for node, vertices in posed_dump(scaled, time).items()
            ), f"{label} scaled {factor} at {time}"
    joined = g.trimesh.scene.scene.append_scenes([scene, scene.copy()])
    for time in times:
        both = sorted(g.np.vstack(list(posed_dump(joined, time).values())).tolist())
        twice = sorted(g.np.vstack([*truth[time].values()] * 2).tolist())
        assert g.np.allclose(both, twice, atol=1e-12), f"{label} appended at {time}"
    copied = scene.copy()
    for animation in copied.animations:
        animation.trs[...] = 99.0
    for skin in copied.skins.values():
        skin.inverse_bind_matrices[...] = 99.0
    assert all(
        g.np.allclose(posed_dump(scene, time)[node], vertices)
        for time in times
        for node, vertices in truth[time].items()
    ), f"{label} copy shares nothing"


def rigid_animation(name, nodes, times, seed):
    """An animation of random rigid motion of every node at every keyframe."""
    matrices = transforms(len(times) * len(nodes), seed=seed, rigid=True)
    matrices = matrices.reshape((len(times), len(nodes), 4, 4))
    return Animation(name=name, nodes=nodes, times=times, matrices=matrices)


def rigid_scene():
    """A box under a moving parent, beside a longer animation of its own."""
    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="child",
        parent_node_name="parent",
        transform=tf.translation_matrix([5, 0, 0]),
    )
    # an animation only moves nodes the graph has, it never adds them
    for node in ("parent", "solo", "slow"):
        scene.graph.update(frame_to=node)
    scene.animations.append(
        rigid_animation("spin", ["child", "solo", "parent"], g.np.linspace(0, 1, 9), 3)
    )
    scene.animations.append(rigid_animation("slow", ["slow"], g.np.linspace(0, 3, 7), 9))
    return scene


def bent_tube():
    """An open tube on Z 0-2 whose knee at Z=1 bends while the hip slides on X."""
    tube = g.trimesh.creation.revolve(
        g.np.column_stack((g.np.full(21, 0.2), g.np.linspace(0.0, 2.0, 21))),
        sections=16,
    )
    # the knee takes over from the hip between Z of 0.8 and 1.2
    knee_weight = g.np.clip((tube.vertices[:, 2] - 0.8) / 0.4, 0.0, 1.0)
    # each joint split unevenly over two of the four influences
    tube.vertex_attributes["joints"] = g.np.tile([0, 1, 1, 0], (len(knee_weight), 1))
    split = g.np.outer(1.0 - knee_weight, [0.3, 0.7]), g.np.outer(knee_weight, [0.6, 0.4])
    tube.vertex_attributes["weights"] = g.np.column_stack(split)[:, [0, 2, 3, 1]]
    scene = g.trimesh.Scene()
    scene.graph.update(frame_to="hip", matrix=g.np.eye(4))
    scene.graph.update(
        frame_from="hip", frame_to="knee", matrix=tf.translation_matrix([0, 0, 1])
    )
    # glTF says the transform of a skinned mesh's own node must not move it
    scene.graph.update(frame_to="holder", matrix=transforms(2, seed=11)[0])
    scene.add_geometry(
        tube, geom_name="tube", node_name="tube", parent_node_name="holder"
    )
    scene.skins["tube"] = Skin(
        joints=["hip", "knee"],
        inverse_bind_matrices=[g.np.eye(4), tf.translation_matrix([0, 0, -1])],
    )
    times = g.np.linspace(0.0, 1.0, 11)
    # the hip slides along X as the knee makes a quarter turn around X
    hip = [tf.translation_matrix([0.5 * time, 0, 0]) for time in times]
    knee = [
        tf.translation_matrix([0, 0, 1])
        @ tf.rotation_matrix(time * g.np.pi / 2, [1, 0, 0])
        for time in times
    ]
    scene.animations.append(
        Animation(
            name="bend",
            nodes=["hip", "knee"],
            times=times,
            matrices=g.np.stack((hip, knee), axis=1),
        )
    )
    return scene, knee_weight


def test_animated_scenes():
    """
    At keys, between them and outside of them every scene poses as each
    edge written by `SceneGraph.update`, survives copy, scale and append,
    and the tube bends as the analytic answer, also loaded back.
    """
    from scipy.spatial import cKDTree

    tube, knee_weight = bent_tube()
    rest = tube.geometry["tube"].vertices.copy()

    def bent(time):
        """The analytic answer: the hip slides and the knee bends about Z=1."""
        time = g.np.clip(time, 0.0, 1.0)
        hip = tf.translation_matrix([0.5 * time, 0, 0])
        knee = hip @ tf.rotation_matrix(time * g.np.pi / 2, [1, 0, 0], [0, 0, 1])
        expected = (1 - knee_weight)[:, None] * tf.transform_points(rest, hip)
        return expected + knee_weight[:, None] * tf.transform_points(rest, knee)

    def unmoved(edges):
        """Edge data other than the matrix an animation drives."""
        return {
            e: {k: v for k, v in d.items() if k != "matrix"} for e, d in edges.items()
        }

    for name, scene in {"rigid": rigid_scene(), "tube": tube}.items():
        keys = g.np.unique(g.np.concatenate([a.times for a in scene.animations]))
        times = [-1.0, *keys, *(keys[1:] + keys[:-1]) / 2, keys[-1] + 1.0]
        reference, graph = scene.copy().graph, scene.graph
        for time in times:
            scene.animate(time)
            for animation in scene.animations:
                for node, matrix in zip(animation.nodes, animation.interpolate(time)):
                    parent = reference.transforms.parents[node]
                    data = reference.transforms.edge_data[(parent, node)]
                    reference.update(node, parent, **{**data, "matrix": matrix})
            assert reference.transforms.parents == graph.transforms.parents, name
            assert unmoved(reference.transforms.edge_data) == unmoved(
                graph.transforms.edge_data
            ), name
            assert all(
                g.np.array_equal(reference[node][0], graph[node][0])
                for node in graph.nodes
            ), f"{name} same as update at {time}"
        assert g.np.isclose(scene.duration, keys[-1]), f"{name} duration"
        assert_operations(scene, keys[:: max(len(keys) // 4, 1)], name)
    assert g.trimesh.Scene().duration == 0.0, "empty duration"

    # the tube bends as the analytic answer, and its own node doesn't move it
    moved = tube.copy()
    moved.graph.update(frame_to="holder", matrix=transforms(2, seed=99)[0])
    reloaded = g.roundtrip(tube.export(file_type="glb"), "glb")
    for time in [-1.0, 0.0, 0.13, 0.5, 0.77, 1.0, 5.0]:
        assert g.np.allclose(posed_dump(tube, time)["tube"], bent(time), atol=1e-12)
        assert g.np.allclose(posed_dump(moved, time)["tube"], bent(time), atol=1e-12)
        # the loaded file to its float32, as vertex order may change on load
        loaded = cKDTree(posed_dump(reloaded, time)["tube"])
        assert loaded.query(bent(time))[0].max() < 1e-5, f"reloaded {time}"
    joined = g.trimesh.scene.scene.append_scenes([tube, tube.copy()])
    joints = {joint for skin in joined.skins.values() for joint in skin.joints}
    assert len(joints) == 4, "append renames joints"
    # a subscene without the joints can't be skinned but still works
    holder = tube.subscene("holder")
    assert not holder.skins and len(holder.dump()) == 1, "subscene drops orphan skin"
    tube.animate(1.0)
    assert g.np.allclose(tube.bounds, [bent(1.0).min(0), bent(1.0).max(0)]), "bounds"

    # a subscene keeps the nodes it has but not its root, and units convert
    rigid = rigid_scene()
    spin = rigid.animations[0]
    sub = rigid.subscene("parent").animations[0]
    assert sub.nodes == ["child"] and g.np.array_equal(
        sub.matrices[:, 0], spin.matrices[:, spin.nodes.index("child")]
    ), "subscene nodes"
    rigid.units = "mm"
    assert "parent" in rigid.convert_units("m").graph.nodes, "convert units"


def test_skin_render():
    """
    One mesh drawn by two nodes with their own skins bending opposite ways
    covers and is as deep as the scene posed by the numpy skinning of `dump`.
    """
    scene = bent_tube()[0]
    scene.animate(1.0)
    scene.graph.update(frame_to="hip_2", matrix=tf.translation_matrix([1.5, 0, 0]))
    scene.graph.update(
        frame_from="hip_2",
        frame_to="knee_2",
        matrix=tf.translation_matrix([0, 0, 1]) @ tf.rotation_matrix(-1.2, [1, 0, 0]),
    )
    scene.graph.update(frame_from="holder", frame_to="tube_2", geometry="tube")
    scene.skins["tube_2"] = Skin(
        joints=["hip_2", "knee_2"],
        inverse_bind_matrices=[tf.translation_matrix([-1.5, 0, z]) for z in [0, -1]],
    )
    scene.camera.resolution = (200, 150)
    scene.camera_transform = scene.camera.look_at(
        scene.bounds, rotation=tf.euler_matrix(0.2, 1.2, 0.0)
    )
    posed = g.trimesh.Scene(scene.to_geometry())
    posed.camera, posed.camera_transform = scene.camera, scene.camera_transform
    arrays, reference = (
        g.render(
            source,
            resolution=[200, 150],
            include_depth=True,
            include_segmentation=True,
        )
        for source in (scene, posed)
    )
    g.assert_rays(scene, arrays, "twins", depth=False)
    covered = arrays.segmentation >= 0
    assert g.np.array_equal(covered, reference.segmentation >= 0), "coverage"
    # float32 depth is rounded to half an epsilon of its size: allow 4
    tolerance = 4.0 * g.np.finfo(g.np.float32).eps * reference.depth.max()
    assert (g.np.abs(arrays.depth - reference.depth) <= tolerance)[covered].all()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_animated_scenes()
