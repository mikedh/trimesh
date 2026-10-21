try:
    from . import generic as g
except BaseException:
    import generic as g


# linear parts of transforms: non-uniform, mirror, shear, rank-2,
# zero, huge uniform scale, identity, and a rotation
LINEAR = [
    g.np.diag([2.0, 3.0, 4.0]),
    g.np.diag([-2.0, 3.0, 4.0]),
    [[1, 2, 0], [0, 1, 1], [0, 0, 1]],
    g.np.diag([2.0, 3.0, 0.0]),
    g.np.zeros((3, 3)),
    g.np.eye(3) * 1e14,
    g.np.eye(3),
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
]


@g.pytest.mark.parametrize("linear", LINEAR)
@g.pytest.mark.parametrize("cached", [False, True])
def test_affine_surface_normals(linear, cached):
    mesh = g.Trimesh([[0, 0, 0], [1, 0, 1], [0, 1, 1]], [[0, 1, 2]], process=False)
    if cached:
        assert g.np.allclose(mesh.face_normals, [-1, -1, 1] / g.np.sqrt(3))
        assert g.np.allclose(mesh.vertex_normals, mesh.face_normals)
    matrix = g.np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = [5, -2, 7]
    mesh.apply_transform(matrix)
    edges = mesh.triangles[:, 1:] - mesh.triangles[:, :1]
    normal = g.np.cross(edges[:, 0], edges[:, 1])
    length = g.np.linalg.norm(normal)
    if length:
        normal /= length
    assert g.np.allclose(mesh.face_normals, normal)
    assert g.np.allclose(mesh.vertex_normals, normal)


@g.pytest.mark.parametrize("linear", LINEAR)
def test_affine_authored_vertex_normals(linear):
    mesh = g.trimesh.creation.icosphere(subdivisions=1)
    # authored normals unrelated to the faces with some zeros
    normals = g.trimesh.unitize(
        g.np.random.default_rng(0).normal(size=mesh.vertices.shape)
    )
    normals[::7] = 0.0
    mesh.vertex_normals = normals
    # two tangent directions perpendicular to each authored normal
    tangent = g.trimesh.unitize(g.np.cross(normals, [0.36, 0.48, 0.8]))
    bitangent = g.np.cross(normals, tangent)

    matrix = g.np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = [5, -2, 7]
    mesh.apply_transform(matrix)
    new = mesh.vertex_normals

    # dot each new normal against transformed directions
    def dot(vectors):
        moved = g.np.dot(vectors, matrix[:3, :3].T)
        return (new * g.trimesh.unitize(moved, threshold=0)).sum(axis=1)

    length = g.np.linalg.norm(new, axis=1)
    checks = {
        "zero normals stay zero": g.np.allclose(new[::7], 0.0),
        "nonzero normals are unit": g.np.allclose(length[length > 0], 1.0),
        "perpendicular to tangent": g.np.abs(dot(tangent)).max() < 1e-8,
        "perpendicular to bitangent": g.np.abs(dot(bitangent)).max() < 1e-8,
        "same side as transformed normal": (dot(normals) > -1e-8).all(),
    }
    for name, ok in checks.items():
        assert ok, name


@g.pytest.mark.parametrize("linear", [L for L in LINEAR if g.np.linalg.det(L) != 0.0])
@g.pytest.mark.parametrize("cached", [False, True])
def test_affine_preserves_convexity(linear, cached):
    mesh = g.trimesh.creation.icosphere(subdivisions=1)
    if cached:
        assert mesh.face_normals.shape == mesh.faces.shape
        assert mesh.vertex_normals.shape == mesh.vertices.shape
    matrix = g.np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = [5, -2, 7]
    mesh.apply_transform(matrix)

    edges = mesh.triangles[:, 1:] - mesh.triangles[:, :1]
    truth = g.trimesh.unitize(g.np.cross(edges[:, 0], edges[:, 1]))
    fresh = g.Trimesh(mesh.vertices, mesh.faces, process=False)
    checks = {
        "convex": mesh.is_convex,
        # mirrors must keep consistent winding and a positive volume
        "volume": mesh.is_volume,
        "face normals match edge cross products": g.np.allclose(mesh.face_normals, truth),
        "vertex normals in recomputed hemisphere": (
            (mesh.vertex_normals * fresh.vertex_normals).sum(axis=1) > 0
        ).all(),
    }
    for name, ok in checks.items():
        assert ok, name


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


def test_is_rigid():
    tr = g.trimesh.transformations
    # any linear map is `R1 @ diag(s) @ R2` so truth comes from `s` alone
    # rigid if every `|s| == 1`
    scales = [[1, 1, 1], [-1, 1, 1], [2, 2, 2], [-2, 2, -2], [1e5] * 3, [1e-3] * 3]
    scales += [[2, 3, 4], [1, 1, 1 + 1e-4], [1, 1, 0], [0, 0, 0]]
    rotations = tr.random_rotation_matrix(num=len(scales) * 2)
    for s, R1, R2 in zip(scales, rotations[::2], rotations[1::2]):
        matrix = R1 @ g.np.diag(s + [1]) @ R2
        matrix[:3, 3] = g.np.random.random(3) * 100
        rigid = bool((g.np.abs(s) == 1).all())
        assert tr.is_rigid(matrix) == rigid
        # an equal error in every entry of the bottom row is never rigid
        matrix[3] += 1e-3
        assert not tr.is_rigid(matrix)

    # `M @ M.T == I + 0.5` has an equal error in every entry and is not rigid
    values, vectors = g.np.linalg.eigh(g.np.eye(3) + 0.5)
    matrix = g.np.eye(4)
    matrix[:3, :3] = vectors @ g.np.diag(g.np.sqrt(values)) @ vectors.T
    assert not tr.is_rigid(matrix)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
