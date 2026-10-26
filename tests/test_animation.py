try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.animation import Animation, Interpolation, Skin

tf = g.trimesh.transformations


def transforms(count, seed=0, rigid=False):
    """An array of `count` random transforms, mirrored and stretched unless `rigid`."""
    return g.np.array(list(g.random_transforms(count, 20.0, seed, rigid)))


def test_sample():
    """
    Every node of an animation samples as the glTF spec samples its parts
    or as a spline through them, which may be in either hemisphere.
    """
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation

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
    matrices = g.np.tile(g.np.eye(4), (keys, count, 1, 1))
    matrices[..., :3, :3] = (
        Rotation.from_quat(parts["rotation"].reshape((-1, 4)))
        .as_matrix()
        .reshape((keys, count, 3, 3))
        * parts["scale"][..., None, :]
    )
    matrices[..., :3, 3] = parts["translation"]
    nodes = [f"node_{index}" for index in range(count)]
    # a spline through quaternions in the same hemisphere as the last
    flips = (parts["rotation"][1:] * parts["rotation"][:-1]).sum(axis=2) < 0
    turns = parts["rotation"].copy()
    turns[1:] *= g.np.cumprod(g.np.where(flips, -1, 1), axis=0)[..., None]
    splines = [
        CubicSpline(times, part) for part in (parts["translation"], turns, parts["scale"])
    ]

    checks = {}
    queries = [-1.0, *times, *(times[1:] + times[:-1]) / 2, *g.np.linspace(-0.5, 4.5, 23)]
    for interpolation in Interpolation:
        animation = Animation(
            name=interpolation,
            nodes=nodes,
            times=times,
            matrices=matrices,
            interpolation=interpolation,
        )
        checks[f"{interpolation} duration"] = animation.duration == times[-1]
        checks[f"{interpolation} keyframes exact"] = all(
            g.np.allclose(animation.at(time), matrix, atol=1e-12)
            for time, matrix in zip(times, matrices)
        )
        for time in queries:
            sampled = animation.at(time)
            checks[f"{interpolation} shape {time:.3f}"] = sampled.shape == (count, 4, 4)
            # the glTF spec's names for linear and step
            mode = interpolation.value.upper()
            clamped = g.np.clip(time, times[0], times[-1])
            for index, matrix in enumerate(sampled):
                translation, quaternion, scale = (
                    g.spec_sample(times, values[:, index], mode, path == "rotation", time)
                    for path, values in parts.items()
                )
                if interpolation == Interpolation.CUBIC:
                    translation, quaternion, scale = (
                        spline(clamped)[index] for spline in splines
                    )
                expected = g.np.eye(4)
                expected[:3, :3] = Rotation.from_quat(quaternion).as_matrix() * scale
                expected[:3, 3] = translation
                checks[f"{interpolation} {nodes[index]} {time:.3f}"] = g.np.allclose(
                    matrix, expected
                )

    # editing the keyframes in place moves what is sampled
    middle = (times[2] + times[3]) / 2
    before = animation.at(middle)
    animation.matrices[..., :3, 3] += [1.0, 2.0, 3.0]
    after = animation.at(middle)
    checks["in place edit"] = g.np.allclose(
        after[:, :3, 3], before[:, :3, 3] + [1.0, 2.0, 3.0]
    ) and g.np.allclose(after[:, :3, :3], before[:, :3, :3])
    # one keyframe holds forever, and a repeated time is still finite
    single = Animation(name="single", nodes=nodes, times=[2.0], matrices=matrices[:1])
    checks["single keyframe"] = all(
        g.np.allclose(single.at(time), matrices[0]) for time in queries
    )
    repeated = Animation(
        name="repeated", nodes=nodes[:1], times=[0.0, 0.0, 1.0], matrices=matrices[:3, :1]
    )
    checks["repeated time finite"] = all(
        g.np.isfinite(repeated.at(time)).all() for time in queries
    )
    for name, kwargs in {
        "decreasing times": {"times": times[::-1], "matrices": matrices},
        "wrong shape": {"times": times, "matrices": matrices[:, :2]},
        "glTF name": {"times": times, "matrices": matrices, "interpolation": "LINEAR"},
        "misspelled": {"times": times, "matrices": matrices, "interpolation": "cubik"},
    }.items():
        try:
            Animation(name=name, nodes=nodes, **kwargs)
            checks[f"raises {name}"] = False
        except ValueError:
            checks[f"raises {name}"] = True
    for name, ok in checks.items():
        assert ok, name


def posed_dump(scene, time):
    """The {node: world vertices} of a copy of `scene` animated to `time`."""
    scene = scene.copy()
    scene.animate(time)
    return {mesh.metadata["node"]: mesh.vertices for mesh in scene.dump()}


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
    checks = {"duration": g.np.isclose(scene.duration, 1.0)}
    for time in [0.0, 0.3, 0.62, 1.0]:
        sampled = animation.at(time)
        pose = dict(zip(animation.nodes, sampled))
        scene.animate(time)
        for node, matrix in pose.items():
            parent = reference.transforms.parents[node]
            data = reference.transforms.edge_data[(parent, node)]
            reference.update(node, parent, **{**data, "matrix": matrix})
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
            graph.get("child", "parent")[0], pose["child"]
        )
        checks[f"world {time}"] = g.np.allclose(
            graph["child"][0], pose["parent"] @ pose["child"]
        )
        checks[f"base frame {time}"] = g.np.allclose(graph["solo"][0], pose["solo"])
        checks[f"parent kept {time}"] = graph.transforms.parents["child"] == "parent"
        # only the matrix changes: the edge keeps its geometry and metadata
        checks[f"edge data kept {time}"] = unmoved(edge_data[("parent", "child")]) == edge
        # sampling doesn't read the graph the animation just wrote
        checks[f"graph independent {time}"] = g.np.array_equal(
            animation.at(time), sampled
        )

    # only the named animations move
    scene.animate(0.0)
    start = graph.get("child", "parent")[0]
    scene.animate(0.62, name="nothing-is-called-this")
    checks["name filter"] = g.np.array_equal(graph.get("child", "parent")[0], start)
    # looping repeats each animation past its own end, so a longer one
    # beside it keeps going rather than both looping together
    slow = rigid_animation("slow", ["slow"], g.np.linspace(0, 3, 7), 9)
    looped = scene.copy()
    looped.graph.update(frame_to="slow")
    looped.animations.append(slow)
    looped.animate(2.3, loop=True)
    pose = dict(zip(animation.nodes, animation.at(0.3)))
    checks["loops each"] = g.np.allclose(
        looped.graph.get("child", "parent")[0], pose["child"]
    ) and g.np.allclose(looped.graph["slow"][0], slow.at(2.3)[0])
    looped.animate(2.3)
    pose = dict(zip(animation.nodes, animation.at(1.0)))
    checks["holds without looping"] = g.np.allclose(
        looped.graph.get("child", "parent")[0], pose["child"]
    )
    checks["empty duration"] = g.trimesh.Scene().duration == 0.0
    # a copied scene's animation can't reach back into the original
    scene.copy().animations[0].matrices[0, 0, 0, 3] = 99.0
    checks["copy independent"] = animation.matrices[0, 0, 0, 3] != 99.0
    # a subscene keeps the nodes it has, but not its root
    sub = scene.subscene("parent").animations[0]
    checks["subscene nodes"] = sub.nodes == ["child"] and g.np.array_equal(
        sub.matrices[:, 0], animation.matrices[:, animation.nodes.index("child")]
    )

    factor = 3.0
    scaled = scene.scaled(factor)
    checks["motion is real"] = not g.np.allclose(
        posed_dump(scene, 0.0)["child"], posed_dump(scene, 1.0)["child"]
    )
    # scaling doesn't rebuild the graph flat, which would drop the
    # nodes the animation drives
    checks["nodes kept"] = set(scaled.graph.nodes) == set(graph.nodes)
    checks["parents kept"] = scaled.graph.transforms.parents == graph.transforms.parents
    checks["instancing kept"] = len(scaled.geometry) == len(scene.geometry)
    # every dumped vertex at every moment is the original times `factor`
    for time in g.np.linspace(0.0, 1.0, 9):
        truth = posed_dump(scene, time)
        checks[f"posed {time:.2f}"] = all(
            g.np.allclose(vertices, truth[node] * factor)
            for node, vertices in posed_dump(scaled, time).items()
        )
    scene.units = "mm"
    checks["convert units"] = "parent" in scene.convert_units("m").graph.nodes

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


def spec_skinned(glb, time):
    """World vertices of every skinned node in a GLB per the glTF spec and scipy."""
    np = g.np
    header, blob = g.glb_chunks(glb)
    nodes = header["nodes"]
    accessors = g.accessor_values(header, blob)
    local = g.spec_nodes(header, accessors, time)
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
