try:
    from . import generic as g
except BaseException:
    import generic as g


class IntegralMeanCurvatureTest(g.unittest.TestCase):
    def test_IMCsphere(self):
        # how close do we need to be - relative tolerance
        tol = 1e-3
        for radius in [0.1, 1.0, 3.1459, 29.20]:
            m = g.trimesh.creation.icosphere(subdivisions=4, radius=radius)
            IMC = m.integral_mean_curvature
            ref = 4 * g.np.pi * radius
            assert g.np.isclose(IMC, ref, rtol=tol)

    def test_IMCcapsule(self):
        # how close do we need to be - relative tolerance
        tol = 1e-3
        radius = 1.2
        for aspect_ratio in [0.0, 0.5, 1.0, 4.0, 100]:
            L = aspect_ratio * radius
            m = g.trimesh.creation.capsule(height=L, radius=radius, count=[64, 64])
            IMC = m.integral_mean_curvature
            ref = g.np.pi * (L + 4 * radius)
            assert g.np.isclose(IMC, ref, rtol=tol)

    def test_IMCbox(self):
        # how close do we need to be - relative tolerance
        tol = 1e-3
        n_tests = 4
        extents = 1 - g.random((n_tests, 3))
        for extent in extents:
            m = g.trimesh.creation.box(extents=extent)
            IMC = m.integral_mean_curvature
            # only the right angle (=pi/2) edges contribute,
            # edges that lie on the faces of the box do not
            ref = 4 * extent.sum() * g.np.pi / 2 * 0.5
            assert g.np.isclose(IMC, ref, rtol=tol)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
