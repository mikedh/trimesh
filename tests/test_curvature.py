try:
    from . import generic as g
except BaseException:
    import generic as g


class CurvatureTest(g.unittest.TestCase):
    def test_gaussian_curvature(self):
        for radius in g.np.linspace(0.25, 2.0, 10):
            m = g.trimesh.creation.icosphere(radius=radius)

            gauss = g.trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh=m, points=m.vertices, radius=radius * 2.0
            ) / (4 * g.np.pi)
            assert g.np.allclose(gauss, 1.0, atol=0.01)

        # a torus should have approximately half its vertices with positive
        # curvature, and half with negative
        t = g.get_mesh("torus.STL")
        gauss = g.trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh=t, points=t.vertices, radius=1.0
        )
        ratio = float((gauss < 0.0).sum()) / float(len(gauss))
        assert g.np.isclose(ratio, 0.5, atol=0.2)

    def test_mean_curvature(self):
        m = g.trimesh.creation.icosphere()
        mean = g.trimesh.curvature.discrete_mean_curvature_measure(m, m.vertices, 2.0) / (
            4 * g.np.pi
        )
        # Check mean curvature for unit sphere is 1.0
        assert g.np.allclose(mean, 1.0, atol=0.01)

    def test_vertex_defect(self):
        # a subdivided box will only have corners and planar regions
        # so all vertex defects should be 0 or 90 degrees
        m = g.trimesh.primitives.Box().subdivide()
        assert g.np.logical_or(
            g.np.isclose(m.vertex_defects, 0.0),
            g.np.isclose(m.vertex_defects, g.np.pi / 2.0),
        ).all()

        assert len(m.vertex_defects) == len(m.vertices)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
