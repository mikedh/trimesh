import generic as g

class RegistrationTest(g.unittest.TestCase):

    def test_procrustes(self):
        
        X = g.np.random.normal(size=30).reshape(-1,3)
        Y = g.trimesh.transformations.transform_points(
                X, g.trimesh.transformations.random_rotation_matrix())
        Y = g.np.random.uniform(1,100)*Y + 100*g.np.random.normal(size=3)
        
        matrix, transformed, cost = g.trimesh.registration.procrustes(X, Y)
        assert(cost < 0.01)

    def test_icp(self):
        m = g.trimesh.creation.box()
        X = m.sample(10)
        X = X + [0.1, 0.1, 0.1]
        
        matrix, transformed, cost = g.trimesh.registration.icp(X, m, scale=False)
        
        assert(cost < 0.01)

        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
