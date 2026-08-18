try:
    from . import generic as g
except BaseException:
    import generic as g


def random_transforms(count, seed=0, mirror=True):
    """
    Stack general affine transforms without a loop.

    Includes non-uniform scale and optionally mirrors, which is what
    separates a real decomposition from one which only handles rotations.

    Parameters
    ------------
    count : int
      How many transforms to generate.
    seed : int
      Which random values to use.
    mirror : bool
      Flip the handedness of a few of them. Note that a segment between
      a mirrored and an unmirrored keyframe has no well defined
      interpolation in any TQS scheme, as the scale has to pass through
      zero to change sign, so tests about the path *between* keyframes
      want this off while tests about the keyframes themselves want it on.

    Returns
    ----------
    matrices : (count, 4, 4) float
      Homogeneous transformation matrices.
    """
    random = g.np.random.default_rng(seed)
    matrices = g.trimesh.transformations.random_rotation_matrix(num=count, seed=seed)
    scale = random.uniform(0.2, 4.0, (count, 3))
    if mirror:
        # a unit quaternion can't represent a reflection
        scale[::5, 0] *= -1.0

    matrices[:, :3, :3] *= scale.reshape((-1, 1, 3))
    matrices[:, :3, 3] = random.uniform(-10.0, 10.0, (count, 3))

    return matrices


def test_layout():
    """
    The keyframe dtype has to stay packed into contiguous runs.

    The sampler blends all ten channels at once through a flat view of
    the structured array, which silently misaligns rather than raising
    if a field is ever reordered, resized, or padded.
    """
    from trimesh.scene import animation

    fields = animation.KEYFRAME.fields
    # a byte of padding anywhere and the flat view stops lining up
    assert animation.KEYFRAME.itemsize == 31 * 8
    # {field name : which float64 column it starts at}
    column = {name: fields[name][1] // 8 for name in animation.KEYFRAME.names}

    assert column["time"] == 0
    # values and both tangents are each a contiguous run of ten
    assert column["translation"] == animation._VALUE.start
    assert column["translation_in"] == animation._TANGENT_IN.start
    assert column["translation_out"] == animation._TANGENT_OUT.start
    for start in (animation._VALUE, animation._TANGENT_IN, animation._TANGENT_OUT):
        assert start.stop - start.start == 10

    # and every run has the same internal layout, which is what lets one
    # set of within-group slices address the values and both tangents
    for group in ("", "_in", "_out"):
        base = column[f"translation{group}"]
        for field, within in (
            ("translation", animation._TRANSLATION),
            ("quaternion", animation._QUATERNION),
            ("scale", animation._SCALE),
        ):
            assert column[f"{field}{group}"] - base == within.start


def test_identity():
    """
    Comparison has to stay identity-based.

    A generated `__eq__` would compare the numpy fields elementwise
    and raise on the ambiguous truth value, which breaks anything as
    ordinary as `animation in scene.animations`.
    """
    from trimesh.scene.animation import RigidAnimation

    eye = g.np.tile(g.np.eye(4), (2, 1, 1))
    current = RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=eye)
    other = RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=eye)

    assert current == current
    assert current != other
    assert current in [current, other]
    assert len({current, other}) == 2


def test_keyframes_exact():
    """
    Sampling at a keyframe time must return that keyframe exactly.

    This is the predicate the whole storage choice rests on: it holds
    for arbitrary affine keyframes only because the transform is stored
    decomposed rather than being decomposed on every sample.
    """
    from trimesh.scene.animation import RigidAnimation

    matrices = random_transforms(37)
    times = g.np.linspace(0.0, 5.0, len(matrices))

    for mode in ("linear", "step", "cubic"):
        current = RigidAnimation(
            frame_to="a", times=times, matrices=matrices, interpolation=mode
        )
        # the stored transform must be the one which was passed in
        assert g.np.allclose(current.matrices, matrices)
        # and sampling at every keyframe time must return it
        assert g.np.allclose(current.at(times), matrices)


def test_cache():
    """
    Mutating keyframes must invalidate everything derived from them.

    A `cached_property` silently fails this, and fails it *partially*:
    fields which happen to be views track while the rest go stale.
    """
    from trimesh.scene.animation import RigidAnimation

    matrices = random_transforms(9, seed=1)
    times = g.np.linspace(0.0, 2.0, len(matrices))
    current = RigidAnimation(frame_to="a", times=times, matrices=matrices)

    # populate every derived value
    assert g.np.allclose(current.at(times), matrices)

    # an in-place edit of any field has to be picked up
    current.keyframes["translation"][3] = [9.0, 9.0, 9.0]
    assert g.np.allclose(current.matrices[3][:3, 3], [9, 9, 9])
    assert g.np.allclose(current.at(times[3])[:3, 3], [9, 9, 9])

    # including a rotation, which is what goes stale under a cached_property.
    # these keyframes carry scale too so compare the rotation factor rather
    # than the whole block, which is rotation and scale together
    spun = g.trimesh.transformations.quaternion_from_matrix(g.spin_z([1.1])[0])
    current.keyframes["quaternion"][4] = spun
    for block in (current.matrices[4], current.at(times[4])):
        assert g.np.allclose(g.trimesh.transformations.tqs_from_matrix(block)[1], spun)

    # and so does replacing the array wholesale
    replaced = current.keyframes.copy()
    replaced["scale"][5] = [2.0, 2.0, 2.0]
    current.keyframes = replaced
    assert g.np.allclose(current.at(times[5]), current.matrices[5])
    assert g.np.isclose(g.np.linalg.det(current.matrices[5][:3, :3]), 8.0)


def test_cubic():
    """
    A cubic spline must actually follow its tangents.
    """
    from trimesh.scene.animation import RigidAnimation, keyframes_from_matrix

    random = g.np.random.default_rng(4)
    matrices = random_transforms(11, seed=2)
    times = g.np.linspace(0.0, 3.0, len(matrices))

    keyframes = keyframes_from_matrix(times, matrices)
    keyframes["translation_in"] = random.uniform(-3.0, 3.0, (len(times), 3))
    keyframes["translation_out"] = random.uniform(-3.0, 3.0, (len(times), 3))
    keyframes["quaternion_in"] = random.uniform(-0.4, 0.4, (len(times), 4))
    keyframes["quaternion_out"] = random.uniform(-0.4, 0.4, (len(times), 4))

    cubic = RigidAnimation(frame_to="a", keyframes=keyframes, interpolation="cubic")
    linear = RigidAnimation(
        frame_to="a", keyframes=keyframes.copy(), interpolation="linear"
    )

    # the Hermite basis is exactly the bracketing keyframes at either end
    assert g.np.allclose(cubic.at(times), matrices)

    # with non-zero tangents the path between keyframes has to leave the
    # straight line, or the tangents are being ignored somewhere
    query = g.np.linspace(0.0, 3.0, 251)
    assert g.np.abs(cubic.at(query) - linear.at(query)).max() > 1e-2

    # an elementwise blend of unit quaternions isn't one, so the sampler
    # has to renormalize: if it doesn't the scale drifts with the rotation
    sampled = cubic.at(query)
    _t, quaternion, _s = g.trimesh.transformations.tqs_from_matrix(sampled)
    assert g.np.allclose(g.np.linalg.norm(quaternion, axis=1), 1.0)

    # and outside the keyframes it clamps rather than flying off
    assert g.np.allclose(cubic.at(-99.0), matrices[0])
    assert g.np.allclose(cubic.at(99.0), matrices[-1])


def test_resample():
    """
    Resampling should preserve the motion, not just the keyframes.
    """
    from trimesh.scene.animation import RigidAnimation, keyframes_from_matrix

    times = g.np.linspace(0.0, 4.0, 9)
    # no mirrors: this is a test about the path between keyframes
    matrices = random_transforms(len(times), seed=7, mirror=False)
    current = RigidAnimation(
        frame_to="child", frame_from="parent", times=times, matrices=matrices, name="w"
    )

    dense = g.np.linspace(0.0, 4.0, 33)
    resampled = current.resample(dense)

    # the whole path has to agree, not only at the new keyframes: sampling
    # at `dense` alone would pass even if interpolation were dropped
    query = g.np.linspace(0.0, 4.0, 411)
    assert g.np.allclose(resampled.at(query), current.at(query))
    # and the result is exact at its own keyframes
    assert g.np.allclose(resampled.at(dense), resampled.matrices)
    assert g.np.allclose(resampled.times, dense)

    # the edge and name come along, since it is the same motion
    assert resampled.frame_to == "child"
    assert resampled.frame_from == "parent"
    assert resampled.name == "w"

    # a step stays stepped, or resampling would smooth it out. `dense` is a
    # superset of `times` so the held values land exactly on the same edges
    original = RigidAnimation(
        frame_to="a", times=times, matrices=matrices, interpolation="step"
    )
    stepped = original.resample(dense)
    assert stepped.interpolation == "step"
    assert g.np.allclose(stepped.at(query), original.at(query))

    # but a spline can't stay one: its tangents only describe the curve
    # through the original keyframes, so they have to be dropped
    cubic = RigidAnimation(
        frame_to="a",
        keyframes=keyframes_from_matrix(times, matrices),
        interpolation="cubic",
    ).resample(dense)
    assert cubic.interpolation == "linear"
    assert not cubic.keyframes["translation_in"].any()
    assert not cubic.keyframes["quaternion_out"].any()


def test_sample():
    """
    Sampling should reproduce keyframes and stay rigid between them.
    """
    from trimesh.scene.animation import RigidAnimation

    times = g.np.linspace(0.0, 4.0, 41)
    matrices = g.spin_z(times * g.np.pi * 0.5)
    current = RigidAnimation(frame_to="a", times=times, matrices=matrices)

    assert len(current) == len(times)
    assert g.np.isclose(current.duration, 4.0)

    # sampling exactly at a keyframe must return that keyframe
    assert g.np.allclose(current.at(times), matrices)
    # a single time returns a single matrix
    assert current.at(1.0).shape == (4, 4)

    # interpolated samples of a rigid animation must stay rigid: a
    # naive elementwise blend of matrices would fail both of these
    between = current.at(g.np.linspace(0.0, 4.0, 397))
    rotation = between[:, :3, :3]
    assert g.np.allclose(rotation @ rotation.transpose(0, 2, 1), g.np.eye(3))
    assert g.np.allclose(g.np.linalg.det(rotation), 1.0)
    assert g.np.allclose(between[:, 3, :], [0, 0, 0, 1])

    # times outside the keyframes clamp rather than extrapolating
    assert g.np.allclose(current.at(-99.0), matrices[0])
    assert g.np.allclose(current.at(99.0), matrices[-1])


def test_step():
    """
    A stepped animation should hold the keyframe at-or-before each time.
    """
    from trimesh.scene.animation import RigidAnimation

    times = g.np.arange(5, dtype=g.np.float64)
    matrices = g.np.tile(g.np.eye(4), (5, 1, 1))
    matrices[:, 0, 3] = g.np.arange(5)

    current = RigidAnimation(
        frame_to="a", times=times, matrices=matrices, interpolation="step"
    )

    # landing exactly on a keyframe picks that keyframe, not the previous
    assert g.np.allclose(current.at(times), matrices)
    # and it holds until the next one
    assert g.np.allclose(current.at(2.99)[0, 3], 2.0)
    assert g.np.allclose(current.at(3.0)[0, 3], 3.0)
    assert g.np.allclose(current.at(-1.0)[0, 3], 0.0)
    assert g.np.allclose(current.at(99.0)[0, 3], 4.0)


def test_degenerate():
    """
    Animations which barely have a timeline should still sample.
    """
    from trimesh.scene.animation import RigidAnimation

    single = g.np.tile(g.np.eye(4), (1, 1, 1))
    single[0, :3, 3] = [1, 2, 3]

    # a single keyframe is constant for all time and leaves no
    # interval to bracket a query with
    for mode in ("linear", "step"):
        current = RigidAnimation(
            frame_to="a", times=[0.0], matrices=single, interpolation=mode
        )
        assert current.duration == 0.0
        assert g.np.allclose(current.at(99.0), single[0])
        assert current.at(99.0).shape == (4, 4)
        assert current.at(g.np.linspace(0, 3, 7)).shape == (7, 4, 4)

    # repeated keyframe times are a zero length interval which
    # would divide by zero when working out where a query falls
    repeated = RigidAnimation(
        frame_to="a",
        times=[0.0, 1.0, 1.0, 2.0],
        matrices=g.np.tile(g.np.eye(4), (4, 1, 1)),
    )
    sampled = repeated.at(g.np.linspace(0.0, 2.0, 25))
    assert g.np.isfinite(sampled).all()
    assert g.np.allclose(sampled, g.np.eye(4))

    # identical adjacent rotations are the degenerate slerp arc
    held = RigidAnimation(
        frame_to="a", times=[0.0, 1.0], matrices=g.np.tile(g.spin_z([0.7]), (2, 1, 1))
    )
    between = held.at(g.np.linspace(0.0, 1.0, 11))
    assert g.np.isfinite(between).all()
    assert g.np.allclose(between, g.spin_z([0.7])[0])


def test_apply():
    """
    Applying an animation should write the local transform.
    """
    from trimesh.scene.animation import RigidAnimation

    tf = g.trimesh.transformations

    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="child",
        parent_node_name="parent",
        transform=tf.translation_matrix([5, 0, 0]),
    )
    scene.graph.update(
        frame_to="parent", frame_from="world", matrix=tf.translation_matrix([0, 7, 0])
    )

    times = g.np.linspace(0.0, 1.0, 9)
    current = RigidAnimation(
        frame_to="child", frame_from="parent", times=times, matrices=g.spin_z(times)
    )
    scene.animations.append(current)

    graph = scene.graph
    # every edge matrix before anything has been applied, which the scene
    # has to be restorable to exactly
    rest = {k: v["matrix"].copy() for k, v in graph.transforms.edge_data.items()}

    for time in [0.0, 0.3, 0.62, 1.0]:
        scene.animate(time)
        # this writes the transform across the animation's own edge, which
        # on a nested graph is not the transform from the base frame
        local = graph.get(frame_to="child", frame_from="parent")[0]
        assert g.np.allclose(local, current.at(time))

        # the parent transform still composes on top of it
        world = graph.get(frame_to="child")[0]
        assert g.np.allclose(world, g.np.dot(graph.get(frame_to="parent")[0], local))

        # and the node has to stay exactly where it was in the graph:
        # a wrong `frame_from` reparents it and leaves a second edge
        # behind, with `parents` and `edge_data` then disagreeing
        assert graph.transforms.parents["child"] == "parent"
        assert ("world", "child") not in graph.transforms.edge_data

    # `None` puts every edge it touched back bit-for-bit, which no partial
    # restore or stale cache can fake
    scene.animate(None)
    after = {k: v["matrix"] for k, v in graph.transforms.edge_data.items()}
    assert set(after) == set(rest)
    assert all((after[k] == rest[k]).all() for k in rest)

    # and it stays the rest pose no matter how many times it's asked
    scene.animate(None)
    assert all((graph.transforms.edge_data[k]["matrix"] == rest[k]).all() for k in rest)

    # a name filter only drives the animations which carry it
    scene.animate(0.62, name="nothing-is-called-this")
    assert all((graph.transforms.edge_data[k]["matrix"] == rest[k]).all() for k in rest)

    # a scene with no animations has a duration of zero rather than raising
    assert scene.duration == current.duration
    assert g.trimesh.Scene().duration == 0.0


def test_apply_flat():
    """
    An animation on a base frame child needs no `frame_from`.
    """
    from trimesh.scene.animation import RigidAnimation

    scene = g.trimesh.Scene()
    scene.add_geometry(g.trimesh.creation.box(), node_name="solo")

    times = g.np.linspace(0.0, 1.0, 5)
    current = RigidAnimation(frame_to="solo", times=times, matrices=g.spin_z(times))
    scene.animations.append(current)
    assert current.frame_from is None

    # `None` has to mean the base frame, matching `SceneGraph.update`
    rest = scene.graph.get(frame_to="solo")[0].copy()
    scene.animate(0.4)
    assert g.np.allclose(scene.graph.get(frame_to="solo")[0], current.at(0.4))
    assert scene.graph.transforms.parents["solo"] == scene.graph.base_frame

    # and it has to resolve to the *same* edge the rest pose was taken
    # from, or the restore would write a second edge rather than undo one
    scene.animate(None)
    assert (scene.graph.get(frame_to="solo")[0] == rest).all()
    assert scene.graph.transforms.parents["solo"] == scene.graph.base_frame


def test_malformed():
    """
    Nonsense keyframes should raise rather than silently misbehave.
    """
    from trimesh.scene.animation import RigidAnimation

    eye = g.np.tile(g.np.eye(4), (3, 1, 1))

    # mismatched times and matrices
    try:
        RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=eye)
        raise AssertionError("accepted mismatched lengths!")
    except ValueError:
        pass

    # no keyframes at all
    try:
        RigidAnimation(frame_to="a", times=[], matrices=g.np.zeros((0, 4, 4)))
        raise AssertionError("accepted an empty animation!")
    except ValueError:
        pass

    # times which go backwards
    try:
        RigidAnimation(frame_to="a", times=[0.0, 2.0, 1.0], matrices=eye)
        raise AssertionError("accepted decreasing times!")
    except ValueError:
        pass


def test_scene_copy():
    """
    Animations should survive a scene copy without sharing arrays.
    """
    from trimesh.scene.animation import RigidAnimation

    scene = g.trimesh.Scene(g.trimesh.creation.box())
    node = scene.graph.nodes_geometry[0]
    times = g.np.linspace(0.0, 1.0, 5)

    scene.animations.append(
        RigidAnimation(frame_to=node, times=times, matrices=g.spin_z(times), name="spin")
    )

    copied = scene.copy()
    assert len(copied.animations) == 1
    assert copied.animations[0].name == "spin"
    assert g.np.allclose(copied.animations[0].at(0.5), scene.animations[0].at(0.5))

    # mutating the copy must not reach back into the original
    copied.animations[0].keyframes["time"][0] = 99.0
    copied.animations[0].keyframes["translation"][0] = [99.0, 0.0, 0.0]
    assert scene.animations[0].times[0] == 0.0
    assert scene.animations[0].matrices[0, 0, 3] == 0.0
    # and the copy has to actually see its own edit, i.e. the deepcopy
    # brought the cache along without bringing stale values along
    assert copied.animations[0].matrices[0, 0, 3] == 99.0

    # a fresh scene starts with no animations rather than a shared list
    assert len(g.trimesh.Scene().animations) == 0
    assert g.trimesh.Scene().animations is not g.trimesh.Scene().animations


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
