try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.animation import RigidAnimation, keyframes_from_matrix


def affine(count, seed=0, rigid=False):
    """
    Build a stack of transforms to keyframe, mirrored and non-uniformly
    scaled unless `rigid`, which no TQS shortcut can fake its way through.
    """
    random = g.np.random.default_rng(seed)
    matrices = g.trimesh.transformations.random_rotation_matrix(num=count, seed=seed)
    if not rigid:
        scale = random.uniform(0.2, 4.0, (count, 3))
        # a mirror can't be a unit quaternion so it has to land in the scale
        scale[::5, 0] *= -1.0
        matrices[:, :3, :3] *= scale.reshape((-1, 1, 3))
    matrices[:, :3, 3] = random.uniform(-10.0, 10.0, (count, 3))
    return matrices


def test_malformed():
    # every way of constructing one wrong should say so rather than
    # silently producing an animation which samples to garbage
    for kwargs, message in [
        ({}, "either"),
        ({"times": [0, 1], "matrices": [g.np.eye(4)]}, "must correspond"),
        ({"keyframes": g.np.zeros(0, dtype=g.trimesh.scene.animation.KEYFRAME)}, "one"),
        ({"times": [1, 0], "matrices": g.np.tile(g.np.eye(4), (2, 1, 1))}, "increasing"),
        ({"times": [0.0], "matrices": [g.np.eye(4)], "interpolation": "nope"}, "unsupp"),
    ]:
        with g.pytest.raises(ValueError, match=message):
            RigidAnimation(frame_to="a", **kwargs)

    # passing both spellings is ambiguous rather than one winning silently
    with g.pytest.raises(ValueError, match="not both"):
        RigidAnimation(
            frame_to="a",
            times=[0.0],
            matrices=[g.np.eye(4)],
            keyframes=keyframes_from_matrix([0.0]),
        )


def test_sample():
    times = g.np.linspace(0.0, 4.0, 12)
    matrices = affine(12)

    for mode in ("linear", "step", "cubic"):
        current = RigidAnimation(
            frame_to="a", times=times, matrices=matrices, interpolation=mode
        )
        # landing exactly on a keyframe has to reproduce it whatever the
        # blend is: this pins the TQS decomposition, the `wxyz` ordering,
        # and where a mirror was pushed, all at once and for all 3 modes
        assert g.np.allclose(current.matrices, matrices)
        assert g.np.allclose(current.at(times), matrices)
        # outside the range clamps rather than extrapolating
        assert g.np.allclose(current.at(-99.0), matrices[0])
        assert g.np.allclose(current.at(99.0), matrices[-1])
        # a scalar time is a single matrix, an array is a stack
        assert current.at(1.0).shape == (4, 4)
        assert current.at(times).shape == (12, 4, 4)
        assert g.np.isclose(current.duration, 4.0)

    # a rotation has to travel the *arc* between keyframes: sampled at
    # uniform times the angle away from the start keyframe grows exactly
    # linearly, where lerping the quaternion would run fast through the
    # middle. note being orthonormal is not enough to catch that, as
    # `quaternion_matrix` renormalizes and a lerp still looks rigid
    ends = affine(2, seed=7, rigid=True)
    current = RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=ends)

    fraction = g.np.linspace(0.0, 1.0, 21)
    # the rotation taking the first keyframe to each sample
    relative = g.np.linalg.inv(ends[0]) @ current.at(fraction)
    trace = g.np.trace(relative[:, :3, :3], axis1=1, axis2=2)
    angle = g.np.arccos(g.np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

    assert angle[-1] > 0.5, "keyframes too close to tell an arc from a chord"
    assert g.np.allclose(angle, fraction * angle[-1])
    assert g.np.allclose(
        relative[:, :3, :3] @ relative[:, :3, :3].transpose(0, 2, 1), g.np.eye(3)
    )


def test_step():
    times = g.np.arange(5, dtype=g.np.float64)
    walk = g.trimesh.transformations.translation_matrix([1, 0, 0]) * g.np.ones((5, 1, 1))
    walk[:, 0, 3] = times

    current = RigidAnimation(
        frame_to="a", times=times, matrices=walk, interpolation="step"
    )
    # a step holds the keyframe at-or-before the query, so landing exactly
    # on one takes that keyframe rather than the one before it
    assert g.np.isclose(current.at(2.99)[0, 3], 2.0)
    assert g.np.isclose(current.at(3.0)[0, 3], 3.0)
    assert g.np.isclose(current.at(-1.0)[0, 3], 0.0)
    assert g.np.isclose(current.at(99.0)[0, 3], 4.0)


def test_cubic():
    # a cubic Hermite reproduces any cubic polynomial *exactly* when handed
    # that polynomial's own derivative as the tangent. that pins all four
    # basis terms at once to machine precision, where comparing against a
    # linear blend would also pass for a spline which ignored its tangents
    times = g.np.linspace(0.0, 3.0, 7)
    coef = g.np.array(
        [[1.0, -2.0, 0.5], [0.7, 1.3, -0.4], [-0.2, 0.6, 0.9], [0.3, -0.1, 0.2]]
    )
    powers = g.np.arange(4).reshape((-1, 1))

    def poly(t):
        return g.np.tensordot(g.np.reshape(t, (-1, 1)) ** powers.T, coef, axes=1)

    def slope(t):
        return g.np.tensordot(
            g.np.reshape(t, (-1, 1)) ** powers[:3].T * [1, 2, 3], coef[1:], axes=1
        )

    keyframes = keyframes_from_matrix(times)
    keyframes["translation"] = poly(times)
    keyframes["translation_in"] = slope(times)
    keyframes["translation_out"] = slope(times)

    cubic = RigidAnimation(frame_to="a", keyframes=keyframes, interpolation="cubic")
    linear = RigidAnimation(frame_to="a", keyframes=keyframes, interpolation="linear")

    query = g.np.linspace(0.0, 3.0, 61)
    assert g.np.allclose(cubic.at(query)[:, :3, 3], poly(query))
    # and the curve is genuinely bent, i.e. this isn't vacuously true of
    # any interpolation which happens to hit the keyframes
    assert g.np.abs(cubic.at(query) - linear.at(query)).max() > 1e-2

    # per the spec a cubic blends the quaternion elementwise and
    # renormalizes, so what comes out is still a rotation
    spun = keyframes_from_matrix(
        times, g.trimesh.transformations.random_rotation_matrix(num=7, seed=2)
    )
    spun["quaternion_in"] = 0.5
    spun["quaternion_out"] = 0.5
    rotation = RigidAnimation(frame_to="a", keyframes=spun, interpolation="cubic").at(
        query
    )[:, :3, :3]
    assert g.np.allclose(rotation @ rotation.transpose(0, 2, 1), g.np.eye(3))


def test_cache():
    times = g.np.linspace(0.0, 4.0, 8)
    current = RigidAnimation(frame_to="a", times=times, matrices=affine(8))

    # warm both caches before touching anything underneath them
    assert current.matrices.shape == (8, 4, 4)
    assert current.at(times).shape == (8, 4, 4)

    # mutating the keyframes in-place has to be visible: a stale
    # `matrices` would silently play the animation it used to be
    current.keyframes["translation"][3] = [9.0, 9.0, 9.0]
    assert g.np.allclose(current.matrices[3][:3, 3], [9, 9, 9])
    assert g.np.allclose(current.at(times[3])[:3, 3], [9, 9, 9])

    # and so does replacing them wholesale through the setter
    current.keyframes = keyframes_from_matrix(times, affine(8, seed=1))
    assert g.np.allclose(current.at(times), current.matrices)


def test_degenerate():
    # a single keyframe is constant for all time and leaves no interval
    # to bracket a query with, which would otherwise divide by zero
    one = RigidAnimation(frame_to="a", times=[2.0], matrices=[g.np.eye(4) * 2])
    assert one.duration == 2.0
    assert one.at(99.0).shape == (4, 4)
    assert one.at(g.np.linspace(0, 3, 7)).shape == (7, 4, 4)
    assert g.np.allclose(one.at(-99.0), one.at(99.0))

    # duplicated keyframe times are a zero-length interval, same problem
    spun = g.trimesh.transformations.rotation_matrix(0.7, [0, 0, 1])
    for mode in ("linear", "step", "cubic"):
        current = RigidAnimation(
            frame_to="a",
            times=[0.0, 0.0, 1.0],
            matrices=g.np.array([spun, spun, spun]),
            interpolation=mode,
        )
        sampled = current.at(g.np.linspace(-1, 2, 13))
        assert g.np.isfinite(sampled).all(), mode
        assert g.np.allclose(sampled, spun), mode


def test_scene():
    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="child",
        parent_node_name="parent",
        transform=g.trimesh.transformations.translation_matrix([5, 0, 0]),
    )
    scene.graph.update(
        frame_to="parent",
        matrix=g.trimesh.transformations.rotation_matrix(0.7, [0, 1, 0]),
    )

    times = g.np.linspace(0.0, 1.0, 9)
    local = affine(9, seed=3, rigid=True)
    nested = RigidAnimation(
        frame_to="child", frame_from="parent", times=times, matrices=local, name="spin"
    )
    solo = RigidAnimation(frame_to="solo", times=times, matrices=local, name="spin")
    scene.animations.extend([nested, solo])

    assert g.np.isclose(scene.duration, 1.0)
    assert g.trimesh.Scene().duration == 0.0

    parent = scene.graph.get(frame_to="parent")[0]
    for time in [0.0, 0.3, 0.62, 1.0]:
        scene.animate(time)
        # the animation drives *one edge*, so the world transform is that
        # edge composed under whatever the parent is doing above it
        assert g.np.allclose(
            scene.graph.get(frame_to="child", frame_from="parent")[0], nested.at(time)
        )
        assert g.np.allclose(
            scene.graph.get(frame_to="child")[0], parent @ nested.at(time)
        )
        # driving an edge must not reparent the node onto the base frame
        assert scene.graph.transforms.parents["child"] == "parent"
        assert ("world", "child") not in scene.graph.transforms.edge_data
        # a `frame_from` of None means the base frame
        assert g.np.allclose(scene.graph.get(frame_to="solo")[0], solo.at(time))

    # only the named animations move
    scene.animate(0.0)
    scene.animate(0.62, name="nothing-is-called-this")
    assert g.np.allclose(
        scene.graph.get(frame_to="child", frame_from="parent")[0], nested.at(0.0)
    )


def test_copy():
    times = g.np.linspace(0.0, 1.0, 6)
    scene = g.trimesh.Scene(g.trimesh.creation.box())
    node = scene.graph.nodes_geometry[0]
    scene.animations.append(
        RigidAnimation(frame_to=node, times=times, matrices=affine(6, seed=4))
    )

    copied = scene.copy()
    assert len(copied.animations) == 1
    assert g.np.allclose(copied.animations[0].at(times), scene.animations[0].at(times))

    # the keyframe arrays must not be shared with the original, or editing
    # a copy would reach back and change the scene it came from
    copied.animations[0].keyframes["translation"][0] = [99.0, 0.0, 0.0]
    assert copied.animations[0].matrices[0, 0, 3] == 99.0
    assert scene.animations[0].matrices[0, 0, 3] != 99.0


def animated_assembly():
    """A nested scene whose animation drives an edge above the geometry."""
    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="child",
        parent_node_name="crank",
        transform=g.trimesh.transformations.translation_matrix([4.0, 0.0, 0.0]),
    )
    scene.graph.update(frame_to="crank", matrix=g.np.eye(4))

    times = g.np.linspace(0.0, 1.0, 5)
    scene.animations.append(
        RigidAnimation(
            frame_to="crank",
            frame_from="world",
            times=times,
            matrices=affine(5, seed=6, rigid=True),
            name="spin",
        )
    )
    return scene


def posed_dump(scene, time):
    """{node : world-space vertices} at one moment of the animation."""
    scene = scene.copy()
    scene.animate(time)
    return {m.metadata["node"]: m.vertices for m in scene.dump()}


def test_scaled():
    times = g.np.linspace(0.0, 1.0, 7)
    scene = g.trimesh.Scene(g.trimesh.creation.box())
    node = scene.graph.nodes_geometry[0]
    scene.animations.append(
        RigidAnimation(frame_to=node, times=times, matrices=affine(7, seed=5))
    )

    factor = 3.0
    scaled = scene.scaled(factor)

    # sample rather than compare keyframes so this holds for any blend:
    # the translation of the animated transform scales and the rotation
    # block does not, which a whole-matrix multiply would break
    dense = g.np.linspace(0.0, 1.0, 31)
    before = scene.animations[0].at(dense)
    after = scaled.animations[0].at(dense)
    assert g.np.allclose(after[:, :3, 3], before[:, :3, 3] * factor)
    assert g.np.allclose(after[:, :3, :3], before[:, :3, :3])


def test_scaled_assembly():
    scene = animated_assembly()
    factor = 3.0
    scaled = scene.scaled(factor)

    # scaling must not rebuild the graph flat, which reparents every
    # geometry onto the base frame and drops any node carrying none —
    # including the edge an animation drives
    assert set(scaled.graph.nodes) == set(scene.graph.nodes)
    assert scaled.graph.transforms.parents == scene.graph.transforms.parents
    assert all(a.frame_to in scaled.graph.nodes for a in scaled.animations)
    # instancing survives, i.e. geometry wasn't copied per-node
    assert len(scaled.geometry) == len(scene.geometry)

    # the predicate: at every moment of the animation every dumped vertex
    # is the original moved by exactly `factor`. that pins position,
    # orientation and size together, where a bounding box would only pin
    # size, and it catches a flattened graph — which leaves the geometry
    # sitting still instead of moving
    for time in g.np.linspace(0.0, 1.0, 9):
        truth = posed_dump(scene, time)
        assert all(
            g.np.allclose(vertices, truth[node] * factor)
            for node, vertices in posed_dump(scaled, time).items()
        )
    # and the motion is real, so the check above isn't vacuous
    assert not g.np.allclose(
        posed_dump(scene, 0.0)["child"], posed_dump(scene, 1.0)["child"]
    )

    # `convert_units` scales through the same path
    scene.units = "mm"
    assert "crank" in scene.convert_units("m").graph.nodes


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
