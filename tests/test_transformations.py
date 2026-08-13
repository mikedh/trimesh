try:
    from . import generic as g
except BaseException:
    import generic as g


def test_flips_winding():
    tf = g.trimesh.transformations
    # a transform flips winding exactly when its determinant is negative
    assert not tf.flips_winding(g.np.eye(4))
    assert not tf.flips_winding(tf.rotation_matrix(1.0, [0, 1, 0]))
    # single-axis reflections flip
    assert tf.flips_winding(g.np.diag([-1, 1, 1, 1]))
    # two reflections compose to a rotation
    assert not tf.flips_winding(g.np.diag([-1, -1, 1, 1]))
    # mirroring all three axes flips — determinant sign, not element signs
    assert tf.flips_winding(g.np.diag([-1, -1, -1, 1]))
    # anisotropic positive scale never flips
    assert not tf.flips_winding(g.np.diag([2.0, 0.5, 3.0, 1]))
    # bare (3, 3) rotations are accepted too
    assert tf.flips_winding(g.np.diag([-1.0, 1.0, 1.0]))
    assert not tf.flips_winding(g.np.eye(3))
    # anything else errors early
    try:
        tf.flips_winding(g.np.zeros((2, 3)))
        raise AssertionError("should have raised on (2, 3)")
    except ValueError:
        pass

    # winding checks must not consume the global numpy RNG —
    # seeded pipelines depend on bit-exact reproducibility
    matrix = tf.random_rotation_matrix(seed=0)
    state = g.np.random.get_state()
    tf.flips_winding(matrix)
    after = g.np.random.get_state()
    assert state[0] == after[0]
    assert g.np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]


class TransformTest(g.unittest.TestCase):
    def test_doctest(self):
        """
        Run doctests on transformations, which checks docstrings
        for interactive sessions and then verifies they execute
        correctly.

        This is how the upstream transformations unit tests,
        but it depends on numpy string formatting and is very
        flaky.
        """
        import doctest
        import random

        import trimesh

        # make sure formatting is the same as their docstrings
        g.np.set_printoptions(suppress=True, precision=5)

        # monkey patch import transformations with random for the examples
        trimesh.transformations.random = random

        # search for interactive sessions in docstrings and verify they work
        # they are super unreliable and depend on janky string formatting
        # the examples call the `random_*` functions which draw from the
        # global RNG so seed in a scope which puts the state back
        with g.RandomSeed(0):
            results = doctest.testmod(
                trimesh.transformations, verbose=False, raise_on_error=False
            )

        if results.failed > 0:
            raise ValueError(str(results))
        g.log.debug(str(results))

    def test_downstream(self):
        """
        Run tests on functions that were added by us to the
        original transformations.py
        """
        tr = g.trimesh.transformations

        assert not tr.is_rigid(g.np.ones((4, 4)))

        planar = tr.planar_matrix(offset=[10, -10], theta=0.0)
        assert g.np.allclose(planar[:2, 2], [10, -10])

        planar = tr.planar_matrix(offset=[0, -0], theta=g.np.pi)
        assert g.np.allclose(planar[:2, 2], [0, 0])

        planar = tr.planar_matrix(offset=[0, 0], theta=0.0)
        assert g.np.allclose(planar, g.np.eye(3))

        as_3D = tr.planar_matrix_to_3D(g.np.eye(3))
        assert g.np.allclose(as_3D, g.np.eye(4))

        spherical = tr.spherical_matrix(theta=0.0, phi=0.0)
        assert g.np.allclose(spherical, g.np.eye(4))

        points = g.np.arange(60, dtype=g.np.float64).reshape((-1, 3))
        assert g.np.allclose(tr.transform_points(points, g.np.eye(4)), points)

        points = g.np.arange(60, dtype=g.np.float64).reshape((-1, 2))
        assert g.np.allclose(tr.transform_points(points, g.np.eye(3)), points)

    def test_around(self):
        # draw inside the block: `g.random` pinned `offset` and `theta` to
        # one value so all 100 rounds used the same rotation
        with g.RandomSeed() as r:
            # check transform_around on 2D points
            points = r.random((100, 2))
            for i, p in enumerate(points):
                offset = r.random(2)
                matrix = g.trimesh.transformations.planar_matrix(
                    theta=r.random() + 0.1, offset=offset, point=p
                )

                # apply the matrix
                check = g.trimesh.transform_points(points, matrix)
                compare = g.np.isclose(check, points + offset)
                # the point we rotated around shouldn't move
                assert compare[i].all()
                # all other points should move
                assert compare.all(axis=1).sum() == 1

        # check transform_around on 3D points
        points = g.random((100, 3))
        for (i, p), matrix in zip(
            enumerate(points), g.random_transforms(len(points), translate=0.0)
        ):
            matrix = g.trimesh.transformations.transform_around(matrix, p)

            # apply the matrix
            check = g.trimesh.transform_points(points, matrix)
            compare = g.np.isclose(check, points)
            # the point we rotated around shouldn't move
            assert compare[i].all()
            # all other points should move
            assert compare.all(axis=1).sum() == 1

    def test_rotation(self):
        """
        test
        """
        rotation_matrix = g.trimesh.transformations.rotation_matrix

        R = rotation_matrix(g.np.pi / 2, [0, 0, 1], [1, 0, 0])
        assert g.np.allclose(g.np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])

        # draw from a stream: `g.random` returns the same values for the same
        # shape, which put `point` exactly on the axis through `direc`
        with g.RandomSeed() as r:
            angle = (r.random() - 0.5) * (2 * g.np.pi)
            direc = r.random(3) - 0.5
            point = r.random(3) - 0.5
        R0 = rotation_matrix(angle, direc, point)
        R1 = rotation_matrix(angle - 2 * g.np.pi, direc, point)
        assert g.trimesh.transformations.is_same_transform(R0, R1)

        R0 = rotation_matrix(angle, direc, point)
        R1 = rotation_matrix(-angle, -direc, point)
        assert g.trimesh.transformations.is_same_transform(R0, R1)

        I = g.np.identity(4, g.np.float64)  # NOQA
        assert g.np.allclose(I, rotation_matrix(g.np.pi * 2, direc))

        assert g.np.allclose(2, g.np.trace(rotation_matrix(g.np.pi / 2, direc, point)))

        # test symbolic
        if g.sp is not None:
            angle = g.sp.Symbol("angle")
            Rs = rotation_matrix(angle, [0, 0, 1], [1, 0, 0])

            R = g.np.array(Rs.subs(angle, g.np.pi / 2.0).evalf()).astype(g.np.float64)

            assert g.np.allclose(g.np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])

    def test_tiny(self):
        """
        Test transformations with models containing
        very small triangles.
        """
        for validate in [False, True]:
            m = g.get_mesh("ADIS16480.STL", validate=validate)
            m.apply_scale(0.001)
            m._cache.clear()
            g.np.nonzero(g.np.linalg.norm(m.face_normals, axis=1) < 1e-3)
            m.apply_transform(
                g.trimesh.transformations.rotation_matrix(g.np.pi / 4, [0, 0, 1])
            )

    def test_quat(self):
        """
        Do some simple checks on our quaternion math.
        """
        # shortcuts to long function names
        tf = g.trimesh.transformations
        is_rigid = tf.is_rigid
        multiply = tf.quaternion_multiply
        to_matrix = tf.quaternion_matrix
        from_matrix = tf.quaternion_from_matrix
        random_matrix = tf.random_rotation_matrix
        random_quat = tf.random_quaternion

        # get some arbitrary rotation matrices
        a = tf.rotation_matrix(0.2, g.trimesh.unitize([1, 2, 3]))
        b = tf.rotation_matrix(0.3, g.trimesh.unitize([1, -2, 0]))

        # convert arbitrary rotations to quaternions
        qa = from_matrix(a)
        qb = from_matrix(b)
        # matrix multiply the original matrices
        mm = g.np.dot(a, b)
        # quaternion multiply then convert back to matrix
        qm = to_matrix(multiply(qa, qb))
        # results should be the same
        assert g.np.allclose(mm, qm, atol=1e-5)
        # all random matrices should be rigid transforms
        assert all(is_rigid(T) for T in random_matrix(num=100, seed=0))
        # random quaternions should all be unit vector
        assert g.np.allclose(
            g.np.linalg.norm(random_quat(num=100, seed=0), axis=1), 1.0, atol=1e-6
        )

    def test_quat_batched(self):
        """
        The quaternion helpers should accept a single value or a stack.
        """
        tf = g.trimesh.transformations

        matrices = tf.random_rotation_matrix(num=100, seed=0)
        assert matrices.shape == (100, 4, 4)

        quat = tf.quaternion_from_matrix(matrices)
        assert quat.shape == (100, 4)

        # the single strongest predicate available here: converting to
        # quaternions and back must be an exact inverse. this is invariant
        # to the sign convention so it can't be cheated, and simultaneously
        # catches `wxyz` vs `xyzw` ordering and row vs column major storage
        assert g.np.allclose(tf.quaternion_matrix(quat), matrices)

        # the batched result must agree elementwise with the scalar one
        assert g.np.allclose(
            quat, [tf.quaternion_from_matrix(m) for m in matrices], atol=1e-8
        )
        # exported quaternions are always unit length
        assert g.np.allclose(g.np.linalg.norm(quat, axis=1), 1.0)

        # a single matrix still returns a single quaternion
        assert tf.quaternion_from_matrix(matrices[0]).shape == (4,)
        assert tf.quaternion_matrix(quat[0]).shape == (4, 4)
        # but a length-1 stack must stay a stack rather than being squeezed
        assert tf.quaternion_from_matrix(matrices[:1]).shape == (1, 4)
        assert tf.quaternion_matrix(quat[:1]).shape == (1, 4, 4)

    def test_tqs(self):
        """
        Decomposing to translation-quaternion-scale must be exact.

        Round tripping is the strongest predicate available: it is
        invariant to the quaternion sign convention so it can't be
        cheated, and it catches `wxyz` ordering, row vs column major
        storage, and where a mirror was pushed all at once.
        """
        tf = g.trimesh.transformations

        count = 100
        random = g.np.random.default_rng(0)
        matrices = tf.random_rotation_matrix(num=count, seed=0)

        scale = random.uniform(0.1, 5.0, (count, 3))
        # mirror a third of them: a unit quaternion can't hold a reflection
        # so it has to end up in the scale instead
        scale[::3, 0] *= -1.0
        matrices[:, :3, :3] *= scale.reshape((-1, 1, 3))
        matrices[:, :3, 3] = random.uniform(-10.0, 10.0, (count, 3))

        translation, quaternion, recovered = tf.tqs_from_matrix(matrices)
        assert translation.shape == (count, 3)
        assert quaternion.shape == (count, 4)
        assert recovered.shape == (count, 3)

        # non-uniform scale and mirrors both have to come back exactly
        assert g.np.allclose(tf.tqs_matrix(translation, quaternion, recovered), matrices)
        assert g.np.allclose(g.np.abs(recovered), g.np.abs(scale))
        # the rotation factor is always a unit quaternion
        assert g.np.allclose(g.np.linalg.norm(quaternion, axis=1), 1.0)
        # and the mirrored ones are the ones which came back negative
        assert g.np.array_equal(
            (recovered < 0).any(axis=1), g.np.linalg.det(matrices[:, :3, :3]) < 0
        )

        # a degenerate zero-scale axis must not divide by zero
        flat = g.np.tile(g.np.eye(4), (3, 1, 1))
        flat[1, :3, :3] *= [0.0, 1.0, 1.0]
        flat[2, :3, :3] *= 0.0
        assert g.np.allclose(tf.tqs_matrix(*tf.tqs_from_matrix(flat)), flat)

        # a single matrix returns single values rather than stacks
        single = tf.tqs_from_matrix(matrices[0])
        assert [s.shape for s in single] == [(3,), (4,), (3,)]
        assert g.np.allclose(tf.tqs_matrix(*single), matrices[0])
        # but a length-1 stack must stay a stack rather than being squeezed
        assert [s.shape for s in tf.tqs_from_matrix(matrices[:1])] == [
            (1, 3),
            (1, 4),
            (1, 3),
        ]

    def test_slerp_batched(self):
        """
        Spherical interpolation should accept a single value or a stack.
        """
        tf = g.trimesh.transformations

        q0 = tf.random_quaternion(num=50, seed=0)
        q1 = tf.random_quaternion(num=50, seed=1)
        fraction = g.np.linspace(0.0, 1.0, 50)

        batched = tf.quaternion_slerp(q0, q1, fraction)
        assert batched.shape == (50, 4)
        # must match calling the scalar version one at a time
        assert g.np.allclose(
            batched,
            [tf.quaternion_slerp(a, b, f) for a, b, f in zip(q0, q1, fraction)],
        )
        # interpolation stays on the unit sphere the whole way
        assert g.np.allclose(g.np.linalg.norm(batched, axis=1), 1.0)

        # the endpoints are returned exactly as passed
        assert g.np.allclose(tf.quaternion_slerp(q0, q1, 0.0), q0)
        assert g.np.allclose(tf.quaternion_slerp(q0, q1, 1.0), q1)

        # a single fraction broadcasts against a stack of quaternions
        assert tf.quaternion_slerp(q0, q1, 0.5).shape == (50, 4)
        # and a single pair of quaternions broadcasts against many fractions
        assert tf.quaternion_slerp(q0[0], q1[0], fraction).shape == (50, 4)
        # everything single returns a single quaternion
        assert tf.quaternion_slerp(q0[0], q1[0], 0.5).shape == (4,)

        # identical quaternions are a degenerate arc which would otherwise
        # divide by `sin(0)`, and are extremely common in keyframed
        # animation where a part simply isn't moving
        held = tf.quaternion_slerp(q0, q0.copy(), fraction)
        assert g.np.isfinite(held).all()
        assert g.np.allclose(held, q0)

        # arguments must not be mutated in place
        before = q1.copy()
        tf.quaternion_slerp(q0, q1, fraction)
        assert g.np.allclose(q1, before)

        # slerp travels at a constant rate: the midpoint is exactly
        # half the total arc away from each endpoint
        mid = tf.quaternion_slerp(q0, q1, 0.5)
        dot = g.np.abs(g.trimesh.util.diagonal_dot(q0, q1))
        assert g.np.allclose(
            g.np.arccos(g.np.clip(dot, -1, 1)),
            2.0
            * g.np.arccos(
                g.np.clip(g.np.abs(g.trimesh.util.diagonal_dot(q0, mid)), -1, 1)
            ),
        )

    def test_angle(self):
        assert g.np.isclose(
            g.trimesh.transformations.angle_between_vectors(g.np.ones(3), g.np.ones(3)),
            0.0,
        )

    def test_symbolic_rotation(self):
        # you can pass `sympy.Symbol` to `trimesh.transformation.rotation_matrix`
        try:
            import sympy as sp
        except BaseException:
            return
        tf = g.trimesh.transformations

        a = sp.Symbol("a")
        vector = [1, 1, 1]
        m = tf.rotation_matrix(a, vector)
        for v in [0.0, 1.1, 1.234, g.np.pi]:
            # evaluate the symbolic matrix with a value
            s = g.np.array(m.subs({a: v}).evalf(), dtype=g.np.float64)
            # call rotation matrix with a scalar
            n = tf.rotation_matrix(v, vector)

            # they should be the same matrix
            assert g.np.allclose(s, n)

    def test_symbolic_euler(self):
        # some of the functions have been modified to support `sympy.Symbol`
        # values which is useful for calculating final rotations symbolically
        try:
            import sympy as sp
        except BaseException:
            return

        euler = g.trimesh.transformations.euler_matrix

        ra, rb, rc = sp.symbols("ra rb rc")
        m = euler(ra, rb, rc)
        for rot in g.random((100, 3)):
            # get the euler matrix evaluated from the symbolic matrix
            s = g.np.array(
                m.subs({ra: rot[0], rb: rot[1], rc: rot[2]}).evalf(), dtype=g.np.float64
            )
            # get it from a numeric scalar
            n = euler(*rot)

            # they should be the same matrix
            assert g.np.allclose(s, n)

    def test_symbolic_translate(self):
        # some of the functions have been modified to support `sympy.Symbol`
        # values which is useful for calculating final rotations symbolically
        try:
            import sympy as sp
        except BaseException:
            return

        translate = g.trimesh.transformations.translation_matrix

        x, y, z = sp.symbols("x y z")

        m = translate([x, y, z])

        for T in g.random((100, 3)):
            # get the euler matrix evaluated from the symbolic matrix
            s = g.np.array(
                m.subs({x: T[0], y: T[1], z: T[2]}).evalf(), dtype=g.np.float64
            )
            # get it from a numeric scalar
            n = translate(T)

            # they should be the same matrix
            assert g.np.allclose(s, n)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
