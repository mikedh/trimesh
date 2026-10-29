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
    Every node of an animation samples as the glTF spec samples its parts
    or as a spline through them, which may be in either hemisphere.
    """
    from scipy.interpolate import CubicSpline, interp1d
    from scipy.spatial.transform import Rotation, Slerp

    rng = g.np.random.default_rng(0)
    keys, count = 7, 5
    times = g.np.sort(rng.uniform(0.0, 4.0, keys))
    # every node stretched and the first mirrored, with the spec's
    # `x, y, z, w` quaternions in any hemisphere
    parts = {
        "translation": rng.normal(size=(keys, count, 3)) * 5.0,
        "rotation": rng.normal(size=(keys, count, 4)),
        "scale": rng.uniform(0.2, 3.0, size=(keys, count, 3)),
    }
    parts["scale"][:, 0, 0] *= -1.0
    parts["rotation"] /= g.np.linalg.norm(parts["rotation"], axis=2)[..., None]

    def compose(translation, quaternion, scale):
        """Matrices from `x, y, z, w` quaternions which scale, rotate, then translate."""
        stack = translation.shape[:-1]
        result = g.np.tile(g.np.eye(4), (*stack, 1, 1))
        rotation = Rotation.from_quat(quaternion.reshape((-1, 4))).as_matrix()
        result[..., :3, :3] = rotation.reshape((*stack, 3, 3)) * scale[..., None, :]
        result[..., :3, 3] = translation
        return result

    matrices = compose(parts["translation"], parts["rotation"], parts["scale"])
    nodes = [f"node_{index}" for index in range(count)]
    # a spline through quaternions in the same hemisphere as the last
    flips = (parts["rotation"][1:] * parts["rotation"][:-1]).sum(axis=2) < 0
    turns = parts["rotation"].copy()
    turns[1:] *= g.np.cumprod(g.np.where(flips, -1, 1), axis=0)[..., None]

    queries = g.np.array(
        [-1.0, *times, *(times[1:] + times[:-1]) / 2, *g.np.linspace(-0.5, 4.5, 23)]
    )
    clamped = g.np.clip(queries, times[0], times[-1])
    # the keyframe at or before each query, blends of the two around it
    # by scipy, and a spline through every keyframe
    held = g.np.clip(g.np.searchsorted(times, clamped, side="right") - 1, 0, keys - 1)
    slerped = g.np.stack(
        [
            Slerp(times, Rotation.from_quat(parts["rotation"][:, node]))(
                clamped
            ).as_quat()
            for node in range(count)
        ],
        axis=1,
    )
    expected = {
        "step": compose(*(part[held] for part in parts.values())),
        "linear": compose(
            interp1d(times, parts["translation"], axis=0)(clamped),
            slerped,
            interp1d(times, parts["scale"], axis=0)(clamped),
        ),
        "cubic": compose(
            *(
                CubicSpline(times, part)(clamped)
                for part in (parts["translation"], turns, parts["scale"])
            )
        ),
    }
    for interpolation in INTERPOLATIONS:
        animation = Animation(
            name=interpolation,
            nodes=nodes,
            times=times,
            matrices=matrices,
            interpolation=interpolation,
        )
        sampled = g.np.array([animation.interpolate(time) for time in queries])
        assert animation.duration == times[-1], f"{interpolation} duration"
        assert all(
            g.np.allclose(animation.interpolate(time), matrix, atol=1e-12)
            for time, matrix in zip(times, matrices)
        ), f"{interpolation} keyframes exact"
        assert g.np.allclose(sampled, expected[interpolation]), f"{interpolation} samples"

    # editing the keyframes in place moves what is sampled
    middle = (times[2] + times[3]) / 2
    before = animation.interpolate(middle)
    animation.matrices[..., :3, 3] += [1.0, 2.0, 3.0]
    after = animation.interpolate(middle)
    assert g.np.allclose(
        after[:, :3, 3], before[:, :3, 3] + [1.0, 2.0, 3.0]
    ) and g.np.allclose(after[:, :3, :3], before[:, :3, :3]), "in place edit"
    # one keyframe holds forever, and a repeated time is still finite
    single = Animation(name="single", nodes=nodes, times=[2.0], matrices=matrices[:1])
    assert all(
        g.np.allclose(single.interpolate(time), matrices[0]) for time in queries
    ), "single keyframe"
    repeated = Animation(
        name="repeated", nodes=nodes[:1], times=[0.0, 0.0, 1.0], matrices=matrices[:3, :1]
    )
    assert all(g.np.isfinite(repeated.interpolate(time)).all() for time in queries), (
        "repeated time finite"
    )
    for name, kwargs in {
        "decreasing times": {"times": times[::-1], "matrices": matrices},
        "wrong shape": {"times": times, "matrices": matrices[:, :2]},
        "glTF name": {"times": times, "matrices": matrices, "interpolation": "LINEAR"},
        "misspelled": {"times": times, "matrices": matrices, "interpolation": "cubik"},
    }.items():
        with g.pytest.raises(ValueError):
            Animation(name=name, nodes=nodes, **kwargs)


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
    assert any(
        not g.np.allclose(truth[times[0]][node], truth[times[-1]][node])
        for node in truth[times[0]]
    ), f"{label} moves"
    for factor in [2.5, [2.5] * 3]:
        scaled = scene.scaled(factor)
        # not rebuilt flat, which would drop the nodes animations drive
        assert scaled.graph.transforms.parents == scene.graph.transforms.parents, label
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
        animation.matrices[...] = 99.0
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
    return Animation(
        name=name,
        nodes=nodes,
        times=times,
        matrices=matrices.reshape((len(times), len(nodes), 4, 4)),
    )


def test_scene():
    """Animations drive scene nodes under their parents and scale with the scene."""
    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="child",
        parent_node_name="parent",
        transform=tf.translation_matrix([5, 0, 0]),
    )
    # an animation only moves nodes the graph has, it never adds them
    for node in ("parent", "solo"):
        scene.graph.update(frame_to=node)
    animation = rigid_animation(
        "spin", ["child", "solo", "parent"], g.np.linspace(0, 1, 9), 3
    )
    scene.animations.append(animation)
    graph = scene.graph
    edge_data = graph.transforms.edge_data

    def unmoved(data):
        """Edge data other than the matrix an animation drives."""
        return {key: value for key, value in data.items() if key != "matrix"}

    edge = unmoved(edge_data[("parent", "child")])
    # every edge written one at a time by `SceneGraph.update`
    reference = scene.copy().graph
    assert g.np.isclose(scene.duration, 1.0), "duration"
    for time in [0.0, 0.3, 0.62, 1.0]:
        sampled = animation.interpolate(time)
        pose = dict(zip(animation.nodes, sampled))
        scene.animate(time)
        for node, matrix in pose.items():
            parent = reference.transforms.parents[node]
            data = reference.transforms.edge_data[(parent, node)]
            reference.update(node, parent, **{**data, "matrix": matrix})
        assert (
            reference.transforms.parents == graph.transforms.parents
            and {k: unmoved(v) for k, v in reference.transforms.edge_data.items()}
            == {k: unmoved(v) for k, v in edge_data.items()}
            and all(
                g.np.array_equal(reference[node][0], graph[node][0])
                for node in graph.nodes
            )
        ), f"same as update {time}"
        # an animation drives one edge so the world transform is that
        # edge composed under whatever the parent is doing
        assert g.np.allclose(graph.get("child", "parent")[0], pose["child"]), (
            f"edge {time}"
        )
        assert g.np.allclose(graph["child"][0], pose["parent"] @ pose["child"]), (
            f"world {time}"
        )
        assert g.np.allclose(graph["solo"][0], pose["solo"]), f"base frame {time}"
        assert graph.transforms.parents["child"] == "parent", f"parent kept {time}"
        # only the matrix changes: the edge keeps its geometry and metadata
        assert unmoved(edge_data[("parent", "child")]) == edge, f"edge data kept {time}"
        # sampling doesn't read the graph the animation just wrote
        assert g.np.array_equal(animation.interpolate(time), sampled), (
            f"graph independent {time}"
        )

    # past its end each animation holds, while a longer one beside it moves
    slow = rigid_animation("slow", ["slow"], g.np.linspace(0, 3, 7), 9)
    looped = scene.copy()
    looped.graph.update(frame_to="slow")
    looped.animations.append(slow)
    looped.animate(2.3)
    pose = dict(zip(animation.nodes, animation.interpolate(1.0)))
    assert g.np.allclose(
        looped.graph.get("child", "parent")[0], pose["child"]
    ) and g.np.allclose(looped.graph["slow"][0], slow.interpolate(2.3)[0]), "holds"
    assert g.trimesh.Scene().duration == 0.0, "empty duration"
    # a subscene keeps the nodes it has, but not its root
    sub = scene.subscene("parent").animations[0]
    assert sub.nodes == ["child"] and g.np.array_equal(
        sub.matrices[:, 0], animation.matrices[:, animation.nodes.index("child")]
    ), "subscene nodes"
    assert_operations(scene, g.np.linspace(0.0, 1.0, 9), "rigid")
    scene.units = "mm"
    assert "parent" in scene.convert_units("m").graph.nodes, "convert units"


def bent_tube():
    """An open tube on Z 0-2 whose knee at Z=1 bends while the hip slides on X."""
    tube = g.trimesh.creation.revolve(
        g.np.column_stack((g.np.full(21, 0.2), g.np.linspace(0.0, 2.0, 21))),
        sections=16,
    )
    # the knee takes over from the hip between Z of 0.8 and 1.2
    knee_weight = g.np.clip((tube.vertices[:, 2] - 0.8) / 0.4, 0.0, 1.0)
    tube.vertex_attributes["joints"] = g.np.tile([0, 1, 0, 0], (len(knee_weight), 1))
    tube.vertex_attributes["weights"] = g.np.column_stack(
        (1.0 - knee_weight, knee_weight, g.np.zeros((len(knee_weight), 2)))
    )

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


def test_skin():
    """Skinning matches the analytic bend, also after a reload and scene edits."""
    from scipy.spatial import cKDTree

    scene, knee_weight = bent_tube()
    rest = scene.geometry["tube"].vertices.copy()

    def tube_at(source, time):
        """The posed tube vertices of a scene at a time."""
        return posed_dump(source, time)["tube"]

    def same_points(a, b):
        """Distance between point sets, as vertex order may change on load."""
        return max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())

    def bent(time):
        """The analytic answer: the hip slides and the knee bends about Z=1."""
        time = g.np.clip(time, 0.0, 1.0)
        hip = tf.translation_matrix([0.5 * time, 0, 0])
        knee = hip @ tf.rotation_matrix(time * g.np.pi / 2, [1, 0, 0], [0, 0, 1])
        expected = (1 - knee_weight)[:, None] * tf.transform_points(rest, hip)
        return expected + knee_weight[:, None] * tf.transform_points(rest, knee)

    for time in [0.0, 0.5, 1.0]:
        assert g.np.allclose(tube_at(scene, time), bent(time), atol=1e-12), (
            f"analytic {time}"
        )
    # at the bind pose the joints put every vertex where it was
    assert g.np.allclose(tube_at(scene, 0.0), rest, atol=1e-12), "bind pose"
    # the skinned mesh's own node doesn't move it
    moved = scene.copy()
    moved.graph.update(frame_to="holder", matrix=transforms(2, seed=99)[0])
    assert g.np.allclose(tube_at(moved, 0.6), tube_at(scene, 0.6), atol=1e-12), (
        "node transform ignored"
    )

    # the file loaded back in bends the same at every time, held outside
    # of the animation, to the float32 the file stores
    glb = scene.export(file_type="glb")
    reloaded = g.roundtrip(glb, "glb")
    for time in [-1.0, 0.0, 0.13, 0.5, 0.77, 1.0, 5.0]:
        assert same_points(tube_at(reloaded, time), bent(time)) < 1e-5, f"reloaded {time}"

    assert_operations(scene, [0.0, 0.4, 0.7, 1.0], "skinned")
    joined = g.trimesh.scene.scene.append_scenes([scene, scene.copy()])
    assert len({joint for skin in joined.skins.values() for joint in skin.joints}) == 4, (
        "append renames joints"
    )
    # a subscene without the joints can't be skinned but still works
    holder = scene.subscene("holder")
    assert not holder.skins and len(holder.dump()) == 1, "subscene drops orphan skin"
    # bounds are of where the mesh actually is
    vertices = tube_at(scene, 1.0)
    scene.animate(1.0)
    assert g.np.allclose(
        scene.bounds, [vertices.min(axis=0), vertices.max(axis=0)], atol=1e-12
    ), "bounds are posed"


def assert_posed(scene, arrays, prefix=""):
    """
    Assert a skinned render covers and is as deep as the scene posed by `dump`.

    Ray depth can't be exact as a bent tube folds over itself in one node.

    Parameters
    ------------
    scene : trimesh.Scene
      Skinned scene that was rendered.
    arrays : RenderArrays
      With `depth` and `segmentation`.
    prefix : str
      Label of the asserts.
    """
    posed = g.trimesh.Scene(scene.dump(concatenate=True))
    posed.camera = scene.camera
    posed.camera_transform = scene.camera_transform
    reference = g.render(
        posed,
        resolution=scene.camera.resolution,
        include_depth=True,
        include_segmentation=True,
    )
    covered = arrays.segmentation >= 0
    # float32 depth is rounded to half an epsilon of its size: allow 4
    tolerance = 4.0 * g.np.finfo(g.np.float32).eps * reference.depth.max()
    g.assert_rays(scene, arrays, prefix, depth=False)
    assert g.np.array_equal(covered, reference.segmentation >= 0), prefix
    close = g.np.abs(arrays.depth - reference.depth) <= tolerance
    assert close[covered].all(), prefix


def test_skin_render():
    """The GPU skins exactly like the numpy skinning `dump` uses."""
    scene, _ = bent_tube()
    scene.animate(1.0)
    scene.camera.resolution = (200, 150)
    scene.camera_transform = scene.camera.look_at(
        scene.bounds, rotation=tf.euler_matrix(0.2, 1.2, 0.0)
    )
    arrays = g.render(
        scene, resolution=[200, 150], include_depth=True, include_segmentation=True
    )
    assert_posed(scene, arrays)

    # and it isn't what an unskinned tube would look like
    unskinned = scene.copy()
    unskinned.skins = {}
    static = g.ray_reference(unskinned).node >= 0
    assert not g.np.array_equal(arrays.segmentation >= 0, static)

    # one mesh drawn by two nodes with their own skins bending opposite ways
    twin = scene.copy()
    twin.graph.update(frame_to="hip_2", matrix=tf.translation_matrix([1.5, 0, 0]))
    twin.graph.update(
        frame_from="hip_2",
        frame_to="knee_2",
        matrix=tf.translation_matrix([0, 0, 1]) @ tf.rotation_matrix(-1.2, [1, 0, 0]),
    )
    twin.graph.update(frame_from="holder", frame_to="tube_2", geometry="tube")
    twin.skins["tube_2"] = Skin(
        joints=["hip_2", "knee_2"],
        inverse_bind_matrices=[tf.translation_matrix([-1.5, 0, z]) for z in [0, -1]],
    )
    twin.camera_transform = twin.camera.look_at(twin.bounds)
    arrays = g.render(
        twin, resolution=[200, 150], include_depth=True, include_segmentation=True
    )
    assert_posed(twin, arrays, "twins ")


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_skin()
