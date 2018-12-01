try:
    from . import generic as g
except BaseException:
    import generic as g


class SphericalTests(g.unittest.TestCase):

    def test_spherical(self):
        """
        Convert vectors to spherical coordinates
        """
        # random unit vectors
        v = g.trimesh.unitize(g.np.random.random((1000, 3)) - .5)
        # (n, 2) angles in radians
        spherical = g.trimesh.util.vector_to_spherical(v)
        # back to unit vectors
        v2 = g.trimesh.util.spherical_to_vector(spherical)

        assert g.np.allclose(v, v2)


class HemisphereTests(g.unittest.TestCase):

    def test_hemisphere(self):
        for dimension in [2, 3]:
            # random unit vectors
            v = g.trimesh.unitize(
                g.np.random.random((10000, dimension)) - .5)

            # add some on- axis points
            v[:dimension] = g.np.eye(dimension)

            # stack vector and negative vector into one soup
            v = g.np.column_stack((v, -v)).reshape((-1, dimension))

            resigned = g.trimesh.util.vector_hemisphere(v)

            # after resigning, negative vectors should equal positive
            check = (abs(g.np.diff(resigned.reshape((-1, 2, dimension)),
                                   axis=1).sum(axis=2)) < g.trimesh.constants.tol.zero).all()
            assert check

            a, s = g.trimesh.util.vector_hemisphere(v, return_sign=True)
            assert g.np.allclose(v, a * s.reshape((-1, 1)))


class AlignTests(g.unittest.TestCase):

    def test_align(self):
        """
        Test aligning two 3D vectors
        """

        # function we're testing
        align = g.trimesh.geometry.align_vectors

        # start with some edge cases and make sure the transform works
        target = g.np.array([0, 0, 1], dtype=g.np.float64)
        vectors = g.np.vstack((g.trimesh.unitize(g.np.random.random((1000, 3)) - .5),
                               [-target, target],
                               g.trimesh.util.generate_basis(target)))
        for vector in vectors:
            T, a = align(vector, target, return_angle=True)
            result = g.np.dot(T, g.np.append(vector, 1))[:3]
            aligned = g.np.linalg.norm(result - target) < 1e8
            assert aligned

        # these vectors should be perpendicular and zero
        angles = [align(i, target, return_angle=True)[1]
                  for i in g.trimesh.util.generate_basis(target)]
        assert g.np.allclose(angles, [g.np.pi / 2, g.np.pi / 2, 0.0])

        # generate angles from 0 to 180 degrees
        angles = g.np.linspace(0.0, g.np.pi, 1000)
        # generate on- plane vectors
        vectors = g.np.column_stack((g.np.cos(angles),
                                     g.np.sin(angles),
                                     g.np.zeros(len(angles))))

        # rotate them arbitrarily off the plane just for funsies
        vectors = g.trimesh.transform_points(vectors,
                                             g.transforms[20])

        for angle, vector in zip(angles, vectors):
            # check alignment to first vector
            # which was created with zero angle
            T, a = align(vector, vectors[0], return_angle=True)

            # check to make sure returned angle corresponds with truth
            assert g.np.isclose(a, angle)

            # check to make sure returned transform is correct
            check = g.np.dot(T, g.np.append(vector, 1))[:3]
            aligned = g.np.linalg.norm(check - vector[0]) < 1e8
            assert aligned


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
