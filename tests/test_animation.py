try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.animation import Animation, Skin

tf = g.trimesh.transformations


def transforms(count, seed=0, rigid=False):
    """An array of `count` random transforms, mirrored and stretched unless `rigid`."""
    return g.np.array(list(g.random_transforms(count, 20.0, seed, rigid)))


def test_sample():
    """Sampling hits keyframes, clamps, keeps edges apart, slerps, steps and edits."""
    times = g.np.linspace(0.0, 4.0, 12)
    matrices = g.np.stack((transforms(12), transforms(12, seed=1)), axis=1)
    edges = [(None, "a"), ("a", "b")]
    query = g.np.linspace(-1.0, 5.0, 37)
    spun = tf.rotation_matrix(0.7, [0, 0, 1])
    degenerate = {
        "single keyframe": ([2.0], [spun]),
        "repeated time": ([0.0, 0.0, 1.0], [spun, spun, spun]),
        "vanishing scale": ([0.0, 1.0], [spun, g.np.diag([0.0, 0.0, 0.0, 1.0])]),
    }

    checks = {}
    for mode in ("linear", "step"):
        current = Animation(edges, times, matrices, interpolation=mode)
        alone = Animation(edges[1:], times, matrices[:, 1], interpolation=mode)
        # landing exactly on a keyframe reproduces it, which pins the
        # decomposition, quaternion order, and mirrors all at once
        checks[f"{mode} keyframes"] = g.np.allclose(current.at(times), matrices)
        checks[f"{mode} clamps before"] = g.np.allclose(current.at(-99.0), matrices[0])
        checks[f"{mode} clamps after"] = g.np.allclose(current.at(99.0), matrices[-1])
        checks[f"{mode} scalar shape"] = current.at(1.0).shape == (2, 4, 4)
        checks[f"{mode} array shape"] = current.at(times).shape == (12, 2, 4, 4)
        # every edge samples the same as an animation of just that edge
        checks[f"{mode} no cross-talk"] = g.np.allclose(
            current.at(query)[:, 1], alone.at(query)[:, 0]
        )
        for name, (keys, keyed) in degenerate.items():
            odd = Animation([(None, "a")], keys, keyed, interpolation=mode)
            checks[f"{name} {mode} finite"] = g.np.isfinite(odd.at(query)).all()
            checks[f"{name} {mode} keyframes"] = g.np.allclose(
                odd.at(odd.times)[:, 0], g.np.reshape(keyed, (-1, 4, 4))
            )
    checks["duration"] = g.np.isclose(current.duration, 4.0)

    # a rotation travels the arc between keyframes: the angle away from
    # the first keyframe grows exactly linearly with time
    ends = transforms(2, seed=7, rigid=True)
    fraction = g.np.linspace(0.0, 1.0, 21)
    sampled = Animation([(None, "a")], [0, 1], ends).at(fraction)[:, 0]
    trace = g.np.trace((g.np.linalg.inv(ends[0]) @ sampled)[:, :3, :3], axis1=1, axis2=2)
    angle = g.np.arccos(g.np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    checks["keyframes far apart"] = angle[-1] > 0.5
    checks["arc not chord"] = g.np.allclose(angle, fraction * angle[-1])

    # a step holds the keyframe at or before the time
    walk = g.np.tile(g.np.eye(4), (5, 1, 1))
    walk[:, 0, 3] = g.np.arange(5)
    step = Animation([(None, "a")], g.np.arange(5.0), walk, interpolation="step")
    checks["step before"] = g.np.isclose(step.at(2.99)[0, 0, 3], 2.0)
    checks["step on"] = g.np.isclose(step.at(3.0)[0, 0, 3], 3.0)

    # editing the matrices in place is seen by the next sample, which
    # requires the cached decomposition to be thrown away
    current = Animation([(None, "a")], times[:8], transforms(8))
    middle = 0.5 * (times[3] + times[4])
    before = current.at(middle)
    current.matrices[3, 0, :3, 3] = [9.0, 9.0, 9.0]
    copied = current.copy()
    copied.matrices[3, 0, :3, 3] = [0.0, 0.0, 0.0]
    checks["edit changes sample"] = not g.np.allclose(current.at(middle), before)
    checks["edit exact at keyframe"] = g.np.allclose(current.at(times[3])[0, :3, 3], 9)
    checks["copy shares nothing"] = g.np.allclose(current.matrices[3, 0, :3, 3], 9.0)
    checks["copy samples its own"] = g.np.allclose(copied.at(times[3])[0, :3, 3], 0)

    for name, ok in checks.items():
        assert ok, name

    # every way of constructing one wrong says so
    for kwargs in [
        {"times": [0.0, 1.0], "matrices": g.np.eye(4)},
        {"times": [1.0, 0.0], "matrices": g.np.tile(g.np.eye(4), (2, 1, 1))},
        {"times": [], "matrices": g.np.zeros((0, 4, 4))},
        {"times": [0.0], "matrices": g.np.eye(4), "interpolation": "cubic"},
    ]:
        with g.pytest.raises(ValueError):
            Animation(edges=[(None, "a")], **kwargs)


def posed_dump(scene, time):
    """The {node: world vertices} of a copy of `scene` animated to `time`."""
    scene = scene.copy()
    scene.animate(time)
    return {mesh.metadata["node"]: mesh.vertices for mesh in scene.dump()}


def test_scene():
    """Animations drive scene edges under their parents and scale with the scene."""
    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="child",
        parent_node_name="parent",
        transform=tf.translation_matrix([5, 0, 0]),
    )
    local = transforms(9, seed=3, rigid=True)
    animation = Animation(
        # `parent` hangs from the base frame by name, `solo` by None
        edges=[("parent", "child"), (None, "solo"), ("world", "parent")],
        times=g.np.linspace(0.0, 1.0, 9),
        matrices=g.np.stack((local, local, transforms(9, seed=6, rigid=True)), axis=1),
        name="spin",
    )
    scene.animations.append(animation)
    # an animation only moves nodes the graph has, it never adds them
    for node in ("parent", "solo"):
        scene.graph.update(frame_to=node)
    graph = scene.graph
    edge_data = graph.transforms.edge_data

    def unmoved(data):
        """Edge data other than the matrix an animation drives."""
        return {key: value for key, value in data.items() if key != "matrix"}

    edge = unmoved(edge_data[("parent", "child")])
    # every edge written one at a time by `SceneGraph.update`
    reference = scene.copy().graph
    checks = {"duration": g.np.isclose(scene.duration, 1.0)}
    for time in [0.0, 0.3, 0.62, 1.0]:
        scene.animate(time)
        sampled = animation.at(time)
        for (frame_from, frame_to), matrix in zip(animation.edges, sampled):
            data = reference.transforms.edge_data.get(
                (frame_from or reference.base_frame, frame_to), {}
            )
            reference.update(frame_to, frame_from, **{**data, "matrix": matrix})
        checks[f"same as update {time}"] = (
            reference.transforms.parents == graph.transforms.parents
            and {k: unmoved(v) for k, v in reference.transforms.edge_data.items()}
            == {k: unmoved(v) for k, v in edge_data.items()}
            and all(
                g.np.array_equal(reference[node][0], graph[node][0])
                for node in graph.nodes
            )
        )
        # an animation drives one edge so the world transform is that
        # edge composed under whatever the parent is doing
        checks[f"edge {time}"] = g.np.allclose(
            graph.get("child", "parent")[0], sampled[0]
        )
        checks[f"world {time}"] = g.np.allclose(
            graph["child"][0], sampled[2] @ sampled[0]
        )
        checks[f"base frame {time}"] = g.np.allclose(graph["solo"][0], sampled[1])
        checks[f"parent kept {time}"] = graph.transforms.parents["child"] == "parent"
        # only the matrix changes: the edge keeps its geometry and metadata
        checks[f"edge data kept {time}"] = unmoved(edge_data[("parent", "child")]) == edge

    # only the named animations move
    scene.animate(0.0)
    scene.animate(0.62, name="nothing-is-called-this")
    checks["name filter"] = g.np.allclose(
        graph.get("child", "parent")[0], animation.at(0.0)[0]
    )
    # looping repeats each animation past its own end, so a longer one
    # beside it keeps going rather than both looping together
    slow = Animation(
        edges=[(None, "slow")],
        times=g.np.linspace(0.0, 3.0, 7),
        matrices=transforms(7, seed=9, rigid=True)[:, None],
        name="slow",
    )
    looped = scene.copy()
    looped.graph.update(frame_to="slow")
    looped.animations.append(slow)
    looped.animate(2.3, loop=True)
    checks["loops each"] = g.np.allclose(
        looped.graph.get("child", "parent")[0], animation.at(0.3)[0]
    ) and g.np.allclose(looped.graph["slow"][0], slow.at(2.3)[0])
    looped.animate(2.3)
    checks["holds without looping"] = g.np.allclose(
        looped.graph.get("child", "parent")[0], animation.at(1.0)[0]
    )
    checks["empty duration"] = g.trimesh.Scene().duration == 0.0
    # a copied scene's animation can't reach back into the original
    scene.copy().animations[0].matrices[0, 0, 0, 3] = 99.0
    checks["copy independent"] = animation.matrices[0, 0, 0, 3] != 99.0

    factor = 3.0
    scaled = scene.scaled(factor)
    dense = g.np.linspace(0.0, 1.0, 31)
    before, after = animation.at(dense), scaled.animations[0].at(dense)
    # translations scale and rotations don't
    checks["translation scaled"] = g.np.allclose(
        after[..., :3, 3], before[..., :3, 3] * factor
    )
    checks["rotation kept"] = g.np.allclose(after[..., :3, :3], before[..., :3, :3])
    # scaling doesn't rebuild the graph flat, which would drop the
    # nodes the animation drives
    checks["nodes kept"] = set(scaled.graph.nodes) == set(graph.nodes)
    checks["parents kept"] = scaled.graph.transforms.parents == graph.transforms.parents
    checks["instancing kept"] = len(scaled.geometry) == len(scene.geometry)
    checks["motion is real"] = not g.np.allclose(
        posed_dump(scene, 0.0)["child"], posed_dump(scene, 1.0)["child"]
    )
    # every dumped vertex at every moment is the original times `factor`
    for time in g.np.linspace(0.0, 1.0, 9):
        truth = posed_dump(scene, time)
        checks[f"posed {time:.2f}"] = all(
            g.np.allclose(vertices, truth[node] * factor)
            for node, vertices in posed_dump(scaled, time).items()
        )
    scene.units = "mm"
    checks["convert units"] = "parent" in scene.convert_units("m").graph.nodes

    # a cubic spline needs two keyframes, so a lone one exports as a
    # held value rather than crashing the spline fit
    lone = g.trimesh.Scene()
    lone.add_geometry(g.trimesh.creation.box(), node_name="mover")
    pose = tf.translation_matrix([1.0, 2.0, 3.0]) @ tf.rotation_matrix(0.5, [0, 0, 1])
    lone.animations.append(Animation([(None, "mover")], [0.5], pose[None, None]))
    exported = lone.export(file_type="glb", animation_spline=True)
    header, _ = g.glb_chunks(exported)
    held = g.roundtrip(exported, "glb").animations[0]
    checks["lone keyframe not spline"] = all(
        sampler["interpolation"] != "CUBICSPLINE"
        for sampler in header["animations"][0]["samplers"]
    )
    checks["lone keyframe round trip"] = g.np.allclose(
        held.at([0.0, 0.5, 9.0])[:, 0], pose, atol=1e-6
    )

    for name, ok in checks.items():
        assert ok, name


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
    hip = g.np.tile(g.np.eye(4), (len(times), 1, 1))
    hip[:, 0, 3] = 0.5 * times
    knee = tf.translation_matrix([0, 0, 1]) @ tf.rotation_matrix(
        times * g.np.pi / 2, [1, 0, 0]
    )
    scene.animations.append(
        Animation([(None, "hip"), ("hip", "knee")], times, g.np.stack((hip, knee), 1))
    )
    return scene, knee_weight


def spec_skinned(glb, time):
    """World vertices of every skinned node in a GLB per the glTF spec and scipy."""
    from scipy.spatial.transform import Rotation, Slerp

    np = g.np
    header, blob = g.glb_chunks(glb)
    nodes = header["nodes"]
    accessors = g.accessor_values(header, blob)

    trs = {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}
    trs = {key: np.array([n.get(key, v) for n in nodes], float) for key, v in trs.items()}
    for animation in header.get("animations", []):
        for channel in animation["channels"]:
            sampler = animation["samplers"][channel["sampler"]]
            times = accessors[sampler["input"]][:, 0]
            values = accessors[sampler["output"]]
            # LINEAR clamps outside the keyframes and slerps rotations
            clamped = np.clip(time, times[0], times[-1])
            path, node = channel["target"]["path"], channel["target"]["node"]
            trs[path][node] = [np.interp(clamped, times, v) for v in values.T]
            if path == "rotation":
                trs[path][node] = Slerp(times, Rotation.from_quat(values))(
                    clamped
                ).as_quat()

    local = np.tile(np.eye(4), (len(nodes), 1, 1))
    local[:, :3, :3] = Rotation.from_quat(trs["rotation"]).as_matrix()
    local[:, :3, :3] *= trs["scale"][:, None, :]
    local[:, :3, 3] = trs["translation"]
    for index, node in enumerate(nodes):
        if "matrix" in node:
            local[index] = np.reshape(node["matrix"], (4, 4)).T
    parent = {c: i for i, node in enumerate(nodes) for c in node.get("children", [])}

    def world(index):
        """A node's matrix under all of its parents."""
        if index in parent:
            return world(parent[index]) @ local[index]
        return local[index]

    result = {}
    for node in nodes:
        if "skin" not in node:
            continue
        skin = header["skins"][node["skin"]]
        bind = (
            accessors[skin["inverseBindMatrices"]].reshape((-1, 4, 4)).transpose(0, 2, 1)
        )
        # the spec's joint matrix is under the inverse of the mesh node's
        # world transform, which the vertex shader then applies and cancels
        joint = np.array([world(j) @ b for j, b in zip(skin["joints"], bind)])
        attributes = header["meshes"][node["mesh"]]["primitives"][0]["attributes"]
        joints = accessors[attributes["JOINTS_0"]].astype(int)
        weights = accessors[attributes["WEIGHTS_0"]][:, :, None, None]
        position = np.column_stack(
            (accessors[attributes["POSITION"]], np.ones(len(joints)))
        )
        blended = (weights * joint[joints]).sum(axis=1)
        result[node["name"]] = (blended @ position[:, :, None])[:, :3, 0]
    return result


def test_skin():
    """Skinning matches the analytic bend, the spec, a reload, and scene edits."""
    from scipy.spatial import cKDTree

    scene, knee_weight = bent_tube()
    rest = scene.geometry["tube"].vertices.copy()

    def tube_at(source, time):
        """The posed tube vertices of a scene at a time."""
        return posed_dump(source, time)["tube"]

    def same_points(a, b):
        """Distance between point sets, as vertex order may change on load."""
        return max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())

    checks = {}
    for time in [0.0, 0.5, 1.0]:
        # the analytic answer: the hip slides and the knee bends about Z=1
        hip = tf.translation_matrix([0.5 * time, 0, 0])
        knee = hip @ tf.rotation_matrix(time * g.np.pi / 2, [1, 0, 0], [0, 0, 1])
        expected = (1 - knee_weight)[:, None] * tf.transform_points(rest, hip)
        expected += knee_weight[:, None] * tf.transform_points(rest, knee)
        checks[f"analytic {time}"] = g.np.allclose(
            tube_at(scene, time), expected, atol=1e-12
        )
    # at the bind pose the joints put every vertex where it was
    checks["bind pose"] = g.np.allclose(tube_at(scene, 0.0), rest, atol=1e-12)
    # the skinned mesh's own node doesn't move it
    moved = scene.copy()
    moved.graph.update(frame_to="holder", matrix=transforms(2, seed=99)[0])
    checks["node transform ignored"] = g.np.allclose(
        tube_at(moved, 0.6), tube_at(scene, 0.6), atol=1e-12
    )
    checks["really moves"] = g.np.abs(tube_at(scene, 1.0) - rest).max() > 0.5

    # an independent reading of the exported file agrees at every time,
    # as does the file loaded back in, even outside the animation
    glb = scene.export(file_type="glb")
    reloaded = g.roundtrip(glb, "glb")
    for time in [-1.0, 0.0, 0.13, 0.5, 0.77, 1.0, 5.0]:
        reference = spec_skinned(glb, time)["tube"]
        checks[f"spec {time}"] = same_points(tube_at(scene, time), reference) < 1e-5
        checks[f"reloaded {time}"] = (
            same_points(tube_at(reloaded, time), reference) < 1e-5
        )

    header = g.glb_chunks(glb)[0]
    accessors = header["accessors"]
    attributes = header["meshes"][0]["primitives"][0]["attributes"]
    kinds = {
        k: (accessors[i]["componentType"], accessors[i]["type"])
        for k, i in attributes.items()
    }
    checks["joints are uint16 VEC4"] = kinds["JOINTS_0"] == (5123, "VEC4")
    checks["weights are float VEC4"] = kinds["WEIGHTS_0"] == (5126, "VEC4")
    bind = accessors[header["skins"][0]["inverseBindMatrices"]]
    checks["inverse bind count"] = bind["count"] == 2
    checks["no custom copies"] = not any(key.startswith("_") for key in attributes)
    checks["one skin"] = len(header["skins"]) == 1
    again = g.glb_chunks(reloaded.export(file_type="glb"))[0]
    checks["export is stable"] = [len(skin["joints"]) for skin in again["skins"]] == [2]

    # everything which copies or combines scenes keeps skins working
    bind = scene.skins["tube"].inverse_bind_matrices
    scene.copy().skins["tube"].inverse_bind_matrices[1, 0, 3] = 99.0
    checks["copy shares nothing"] = bind[1, 0, 3] == 0.0
    for factor in [2.5, [2.5, 2.5, 2.5]]:
        scaled = scene.scaled(factor)
        for time in [0.0, 0.4, 1.0]:
            checks[f"scaled {factor} {time}"] = g.np.allclose(
                tube_at(scaled, time), 2.5 * tube_at(scene, time), atol=1e-9
            )
    joined = g.trimesh.scene.scene.append_scenes([scene, scene.copy()])
    joined_posed = posed_dump(joined, 0.7)
    checks["append keeps both"] = len(joined_posed) == 2 and all(
        g.np.allclose(vertices, tube_at(scene, 0.7), atol=1e-12)
        for vertices in joined_posed.values()
    )
    checks["append renames joints"] = (
        len({joint for skin in joined.skins.values() for joint in skin.joints}) == 4
    )
    # a subscene without the joints can't be skinned but still works
    holder = scene.subscene("holder")
    checks["subscene drops orphan skin"] = not holder.skins and len(holder.dump()) == 1
    # bounds are of where the mesh actually is
    vertices = tube_at(scene, 1.0)
    scene.animate(1.0)
    checks["bounds are posed"] = g.np.allclose(
        scene.bounds, [vertices.min(axis=0), vertices.max(axis=0)], atol=1e-12
    )

    for name, ok in checks.items():
        assert ok, name


def posed_checks(scene, arrays, prefix=""):
    """
    A render of a skinned scene is the render of the same scene posed on
    the CPU by `dump`: the same pixels covered and the same depth to within
    float32 rounding. Ray casts can't check depth exactly here as a bent
    tube folds over itself inside one node, which isn't a node edge.
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
    checks = g.ray_checks(scene, arrays, prefix)
    # replaced by the exact comparison with the CPU posed mesh below
    checks.pop(f"{prefix}depth is rays")
    checks[f"{prefix}coverage is cpu skin"] = g.np.array_equal(
        covered, reference.segmentation >= 0
    )
    checks[f"{prefix}depth is cpu skin"] = bool(
        (g.np.abs(arrays.depth - reference.depth)[covered] <= tolerance).all()
    )
    return checks


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
    checks = posed_checks(scene, arrays)

    # and it isn't what an unskinned tube would look like
    unskinned = scene.copy()
    unskinned.skins = {}
    static = g.ray_reference(unskinned).node >= 0
    checks["not the unskinned tube"] = g.iou(arrays.segmentation >= 0, static) < 0.8

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
    checks.update(posed_checks(twin, arrays, "twins "))

    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_skin()
