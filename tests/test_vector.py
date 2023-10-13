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
        v = g.trimesh.unitize(g.random((1000, 3)) - 0.5)
        # (n, 2) angles in radians
        spherical = g.trimesh.util.vector_to_spherical(v)
        # back to unit vectors
        v2 = g.trimesh.util.spherical_to_vector(spherical)

        assert g.np.allclose(v, v2)


class HemisphereTests(g.unittest.TestCase):
    def test_hemisphere(self):
        for dimension in [2, 3]:
            # random unit vectors
            v = g.trimesh.unitize(g.random((10000, dimension)) - 0.5)

            # add some on- axis points
            v[:dimension] = g.np.eye(dimension)

            # stack vector and negative vector into one soup
            v = g.np.column_stack((v, -v)).reshape((-1, dimension))

            resigned = g.trimesh.util.vector_hemisphere(v)

            # after resigning, negative vectors should equal positive
            check = (
                abs(g.np.diff(resigned.reshape((-1, 2, dimension)), axis=1).sum(axis=2))
                < g.trimesh.constants.tol.zero
            ).all()
            assert check

            a, s = g.trimesh.util.vector_hemisphere(v, return_sign=True)
            assert g.np.allclose(v, a * s.reshape((-1, 1)))


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
