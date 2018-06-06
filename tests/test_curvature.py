import generic as g

class CurvatureTest(g.unittest.TestCase):

    def test_gaussian_curvature(self):
        m = g.trimesh.creation.icosphere()

        gauss = g.trimesh.curvature.discrete_gaussian_curvature_measure(m, m.vertices, 2.0)/(4*g.np.pi)
        
        # Check gaussian curvature for unit sphere is 1.0
        assert (g.np.abs(gauss - 1.0) < 0.01).all()

    def test_mean_curvature(self):
        m = g.trimesh.creation.icosphere()

        mean = g.trimesh.curvature.discrete_mean_curvature_measure(m, m.vertices, 2.0)/(4*g.np.pi)
        
        # Check mean curvature for unit sphere is 1.0
        assert (g.np.abs(mean - 1.0) < 0.01).all()

        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
