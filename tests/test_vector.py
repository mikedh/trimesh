try:
    from . import generic as g
except BaseException:
    import generic as g


class SphericalTests(g.unittest.TestCase):

    def test_spherical(self):
        v = g.trimesh.unitize(g.np.random.random((1000, 3)) - .5)
        spherical = g.trimesh.util.vector_to_spherical(v)
        v2 = g.trimesh.util.spherical_to_vector(spherical)
        self.assertTrue((g.np.abs(v - v2) < g.trimesh.constants.tol.merge).all())


class HemisphereTests(g.unittest.TestCase):

    def test_hemisphere(self):
        for dimension in [2, 3]:
            v = g.trimesh.unitize(g.np.random.random((10000, dimension)) - .5)

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


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
