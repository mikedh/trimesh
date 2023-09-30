try:
    from . import generic as g
except BaseException:
    import generic as g

tol_norm = 1e-10


class AlignTests(g.unittest.TestCase):
    def test_align(self):
        """
        Test aligning two 3D vectors
        """

        # function we're testing
        align = g.trimesh.geometry.align_vectors
        is_rigid = g.trimesh.transformations.is_rigid

        # start with some edge cases and make sure the transform works
        target = g.np.array([0, 0, -1], dtype=g.np.float64)
        vectors = g.np.vstack(
            (
                g.trimesh.unitize(g.random((1000, 3)) - 0.5),
                g.random((1000, 3)) - 0.5,
                [-target, target],
                g.trimesh.util.generate_basis(target),
                [
                    [7.12106798e-07, -7.43194705e-08, 1.00000000e00],
                    [0, 0, -1],
                    [1e-4, 1e-4, -1],
                ],
            )
        )

        # collect errors
        norms = []
        unitized = g.trimesh.unitize(vectors)
        for unit_dest, dest in zip(unitized[-10:], vectors[-10:]):
            for unit, vector in zip(unitized, vectors):
                T, a = align(vector, dest, return_angle=True)
                assert is_rigid(T)
                assert g.np.isclose(g.np.linalg.det(T), 1.0)
                # rotate vector with transform
                check = g.np.dot(T[:3, :3], unit)
                # compare to target vector
                norm = g.np.linalg.norm(check - unit_dest)
                norms.append(norm)
                assert norm < tol_norm

        norms = g.np.array(norms)
        g.log.debug(
            "vector error after transform:\n"
            + "err.ptp: {}\nerr.std: {}\nerr.mean: {}\nerr.median: {}".format(
                norms.ptp(), norms.std(), norms.mean(), g.np.median(norms)
            )
        )

        # these vectors should be perpendicular and zero
        angles = [
            align(i, target, return_angle=True)[1]
            for i in g.trimesh.util.generate_basis(target)
        ]
        assert g.np.allclose(angles, [g.np.pi / 2, g.np.pi / 2, 0.0])

    def test_range(self):
        # function we're testing
        align = g.trimesh.geometry.align_vectors
        is_rigid = g.trimesh.transformations.is_rigid

        # generate angles from 0 to 180 degrees
        angles = g.np.linspace(0.0, g.np.pi / 1e7, 10000)
        # generate on- plane vectors
        vectors = g.np.column_stack(
            (g.np.cos(angles), g.np.sin(angles), g.np.zeros(len(angles)))
        )

        # rotate them arbitrarily off the plane just for funsies
        vectors = g.trimesh.transform_points(vectors, g.transforms[20])

        for angle, vector in zip(angles, vectors):
            g.trimesh.util.generate_basis(vector)
            # check alignment to first vector
            # which was created with zero angle
            T, a = align(vector, vectors[0], return_angle=True)
            assert is_rigid(T)
            # check to make sure returned angle corresponds with truth

            assert g.np.isclose(a, angle, atol=1e-6)

            # check to make sure returned transform is correct
            check = g.np.dot(T[:3, :3], vector)
            norm = g.np.linalg.norm(check - vectors[0])

            assert norm < tol_norm

    def test_rigid(self):
        # check issues with near-reversed vectors not returning rigid
        align = g.trimesh.geometry.align_vectors
        T = align([0, 0, -1], [-1e-17, 1e-17, 1])
        assert g.np.isclose(g.np.linalg.det(T), 1.0)

        T = align([0, 0, -1], [-1e-4, 1e-4, 1])
        assert g.np.isclose(g.np.linalg.det(T), 1.0)

        vector_1 = g.np.array([7.12106798e-07, -7.43194705e-08, 1.00000000e00])
        vector_2 = g.np.array([0, 0, -1])
        T, angle = align(vector_1, vector_2, return_angle=True)
        assert g.np.isclose(g.np.linalg.det(T), 1.0)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
