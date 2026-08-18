try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene import animation
from trimesh.scene.animation import RigidAnimation, keyframes_from_matrix
from trimesh.transformations import rotation_matrix

# the axis every spin in here is about
Z = [0.0, 0.0, 1.0]


def test_animation():
    """
    Keyframe storage, sampling, and every interpolation mode.
    """
    tf = g.trimesh.transformations

    # ------------------------------------------------------------ layout
    # the sampler blends all ten channels at once through a flat view of
    # the structured array, which silently misaligns rather than raising
    # if a field is ever reordered, resized, or padded
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

    # -------------------------------------------------- malformed inputs
    eye = g.np.tile(g.np.eye(4), (3, 1, 1))
    with g.pytest.raises(ValueError, match="must correspond"):
        RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=eye)
    with g.pytest.raises(ValueError, match="at least one keyframe"):
        RigidAnimation(frame_to="a", times=[], matrices=g.np.zeros((0, 4, 4)))
    with g.pytest.raises(ValueError, match="must be increasing"):
        RigidAnimation(frame_to="a", times=[0.0, 2.0, 1.0], matrices=eye)
    with g.pytest.raises(ValueError, match="unsupported interpolation"):
        RigidAnimation(
            frame_to="a", times=[0.0, 1.0, 2.0], matrices=eye, interpolation="bezier"
        )

    # comparison has to stay identity-based: a generated `__eq__` would
    # compare the numpy fields elementwise and raise on the ambiguous
    # truth value, breaking anything as ordinary as `a in scene.animations`
    current = RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=eye[:2])
    other = RigidAnimation(frame_to="a", times=[0.0, 1.0], matrices=eye[:2])
    assert current == current
    assert current != other
    assert current in [current, other]
    assert len({current, other}) == 2

    # ------------------------------------------------- arbitrary affines
    # non-uniform scale throughout and a mirror every fifth, which a unit
    # quaternion can't represent so it has to land in the scale instead
    random = g.np.random.default_rng(0)
    affine = tf.random_rotation_matrix(num=37, seed=0)
    scale = random.uniform(0.2, 4.0, (37, 3))
    scale[::5, 0] *= -1.0
    affine[:, :3, :3] *= scale.reshape((-1, 1, 3))
    affine[:, :3, 3] = random.uniform(-10.0, 10.0, (37, 3))
    times = g.np.linspace(0.0, 5.0, len(affine))

    # the predicate the whole storage choice rests on: it holds for
    # arbitrary affine keyframes only because the transform is stored
    # decomposed rather than being decomposed on every sample
    for mode in ("linear", "step", "cubic"):
        current = RigidAnimation(
            frame_to="a", times=times, matrices=affine, interpolation=mode
        )
        # the stored transform must be the one which was passed in
        assert g.np.allclose(current.matrices, affine)
        # and sampling at every keyframe time must return it
        assert g.np.allclose(current.at(times), affine)

    # -------------------------------------------------------- cache
    # mutating keyframes must invalidate everything derived from them. a
    # `cached_property` silently fails this, and fails it *partially*:
    # fields which happen to be views track while the rest go stale
    current = RigidAnimation(frame_to="a", times=times[:9], matrices=affine[:9])
    # populate every derived value
    assert g.np.allclose(current.at(times[:9]), affine[:9])

    # an in-place edit of any field has to be picked up
    current.keyframes["translation"][3] = [9.0, 9.0, 9.0]
    assert g.np.allclose(current.matrices[3][:3, 3], [9, 9, 9])
    assert g.np.allclose(current.at(times[3])[:3, 3], [9, 9, 9])

    # including a rotation, which is what goes stale under a cached_property.
    # these keyframes carry scale too so compare the rotation factor rather
    # than the whole block, which is rotation and scale together
    spun = tf.quaternion_from_matrix(rotation_matrix(1.1, Z))
    current.keyframes["quaternion"][4] = spun
    for block in (current.matrices[4], current.at(times[4])):
        assert g.np.allclose(tf.tqs_from_matrix(block)[1], spun)

    # and so does replacing the array wholesale
    replaced = current.keyframes.copy()
    replaced["scale"][5] = [2.0, 2.0, 2.0]
    current.keyframes = replaced
    assert g.np.allclose(current.at(times[5]), current.matrices[5])
    assert g.np.isclose(g.np.linalg.det(current.matrices[5][:3, :3]), 8.0)

    # --------------------------------------------------------- cubic
    # a spline must actually follow its tangents
    keyframes = keyframes_from_matrix(times[:11], affine[:11])
    keyframes["translation_in"] = random.uniform(-3.0, 3.0, (11, 3))
    keyframes["translation_out"] = random.uniform(-3.0, 3.0, (11, 3))
    keyframes["quaternion_in"] = random.uniform(-0.4, 0.4, (11, 4))
    keyframes["quaternion_out"] = random.uniform(-0.4, 0.4, (11, 4))

    cubic = RigidAnimation(frame_to="a", keyframes=keyframes, interpolation="cubic")
    linear = RigidAnimation(
        frame_to="a", keyframes=keyframes.copy(), interpolation="linear"
    )

    # the Hermite basis is exactly the bracketing keyframes at either end
    assert g.np.allclose(cubic.at(times[:11]), affine[:11])

    # with non-zero tangents the path between keyframes has to leave the
    # straight line, or the tangents are being ignored somewhere
    query = g.np.linspace(times[0], times[10], 251)
    assert g.np.abs(cubic.at(query) - linear.at(query)).max() > 1e-2

    # an elementwise blend of unit quaternions isn't one, so the sampler
    # has to renormalize: if it doesn't the scale drifts with the rotation
    _t, quaternion, _s = tf.tqs_from_matrix(cubic.at(query))
    assert g.np.allclose(g.np.linalg.norm(quaternion, axis=1), 1.0)

    # and outside the keyframes it clamps rather than flying off
    assert g.np.allclose(cubic.at(-99.0), affine[0])
    assert g.np.allclose(cubic.at(99.0), affine[10])

    # ---------------------------------------------------- rigid sampling
    times = g.np.linspace(0.0, 4.0, 41)
    matrices = rotation_matrix(times * g.np.pi * 0.5, Z)
    current = RigidAnimation(frame_to="a", times=times, matrices=matrices)

    assert len(current) == len(times)
    assert g.np.isclose(current.duration, 4.0)
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

    # ---------------------------------------------------------- step
    # a stepped animation holds the keyframe at-or-before each time
    held = g.np.tile(g.np.eye(4), (5, 1, 1))
    held[:, 0, 3] = g.np.arange(5)
    current = RigidAnimation(
        frame_to="a",
        times=g.np.arange(5, dtype=g.np.float64),
        matrices=held,
        interpolation="step",
    )
    # landing exactly on a keyframe picks that keyframe, not the previous
    assert g.np.allclose(current.at(g.np.arange(5, dtype=g.np.float64)), held)
    # and it holds until the next one
    assert g.np.allclose(current.at(2.99)[0, 3], 2.0)
    assert g.np.allclose(current.at(3.0)[0, 3], 3.0)
    assert g.np.allclose(current.at(-1.0)[0, 3], 0.0)
    assert g.np.allclose(current.at(99.0)[0, 3], 4.0)

    # ----------------------------------------------------- degenerate
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
    still = RigidAnimation(
        frame_to="a",
        times=[0.0, 1.0],
        matrices=g.np.tile(rotation_matrix(0.7, Z), (2, 1, 1)),
    )
    between = still.at(g.np.linspace(0.0, 1.0, 11))
    assert g.np.isfinite(between).all()
    assert g.np.allclose(between, rotation_matrix(0.7, Z))

    # ------------------------------------------------------- resample
    # no mirrors here: a segment between a mirrored and an unmirrored
    # keyframe has no well defined interpolation in any TQS scheme, as
    # the scale has to pass through zero to change sign
    times = g.np.linspace(0.0, 4.0, 9)
    smooth = tf.random_rotation_matrix(num=9, seed=7)
    smooth[:, :3, :3] *= random.uniform(0.2, 4.0, (9, 1, 3))
    smooth[:, :3, 3] = random.uniform(-10.0, 10.0, (9, 3))
    current = RigidAnimation(
        frame_to="child", frame_from="parent", times=times, matrices=smooth, name="w"
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
        frame_to="a", times=times, matrices=smooth, interpolation="step"
    )
    stepped = original.resample(dense)
    assert stepped.interpolation == "step"
    assert g.np.allclose(stepped.at(query), original.at(query))

    # but a spline can't stay one: its tangents only describe the curve
    # through the original keyframes, so they have to be dropped
    cubic = RigidAnimation(
        frame_to="a",
        keyframes=keyframes_from_matrix(times, smooth),
        interpolation="cubic",
    ).resample(dense)
    assert cubic.interpolation == "linear"
    assert not cubic.keyframes["translation_in"].any()
    assert not cubic.keyframes["quaternion_out"].any()


def test_animation_scene():
    """
    Applying animations to a scene graph, and copying them.
    """
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
    # a second node hanging directly off the base frame, which an
    # animation targets without naming a `frame_from` at all
    scene.add_geometry(g.trimesh.creation.box(), node_name="solo")

    times = g.np.linspace(0.0, 1.0, 9)
    nested = RigidAnimation(
        frame_to="child",
        frame_from="parent",
        times=times,
        matrices=rotation_matrix(times, Z),
    )
    flat = RigidAnimation(
        frame_to="solo", times=times, matrices=rotation_matrix(times * 2.0, Z)
    )
    scene.animations.extend([nested, flat])
    assert flat.frame_from is None

    graph = scene.graph
    # every edge matrix before anything has been applied, which the scene
    # has to be restorable to exactly
    rest = {k: v["matrix"].copy() for k, v in graph.transforms.edge_data.items()}

    for time in [0.0, 0.3, 0.62, 1.0]:
        scene.animate(time)
        # this writes the transform across the animation's own edge, which
        # on a nested graph is not the transform from the base frame
        local = graph.get(frame_to="child", frame_from="parent")[0]
        assert g.np.allclose(local, nested.at(time))

        # the parent transform still composes on top of it
        world = graph.get(frame_to="child")[0]
        assert g.np.allclose(world, g.np.dot(graph.get(frame_to="parent")[0], local))

        # and the node has to stay exactly where it was in the graph:
        # a wrong `frame_from` reparents it and leaves a second edge
        # behind, with `parents` and `edge_data` then disagreeing
        assert graph.transforms.parents["child"] == "parent"
        assert ("world", "child") not in graph.transforms.edge_data

        # a `frame_from` of None has to mean the base frame, matching
        # `SceneGraph.update`, and resolve to the *same* edge the rest
        # pose came from or the restore writes a second edge
        assert g.np.allclose(graph.get(frame_to="solo")[0], flat.at(time))
        assert graph.transforms.parents["solo"] == graph.base_frame

    # `None` puts every edge it touched back bit-for-bit, which no partial
    # restore or stale cache can fake
    scene.animate(None)
    after = {k: v["matrix"] for k, v in graph.transforms.edge_data.items()}
    assert set(after) == set(rest)
    assert all((after[k] == rest[k]).all() for k in rest)
    assert graph.transforms.parents["solo"] == graph.base_frame

    # and it stays the rest pose no matter how many times it's asked
    scene.animate(None)
    assert all((graph.transforms.edge_data[k]["matrix"] == rest[k]).all() for k in rest)

    # a name filter only drives the animations which carry it
    scene.animate(0.62, name="nothing-is-called-this")
    assert all((graph.transforms.edge_data[k]["matrix"] == rest[k]).all() for k in rest)

    # a scene with no animations has a duration of zero rather than raising
    assert scene.duration == nested.duration
    assert g.trimesh.Scene().duration == 0.0

    # --------------------------------------------------------- copying
    nested.name = "spin"
    copied = scene.copy()
    assert len(copied.animations) == len(scene.animations)
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
