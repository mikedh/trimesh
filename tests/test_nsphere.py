import generic as g

class NSphereTest(g.unittest.TestCase):
        
    def test_minball(self):
        for m in g.get_meshes(5):
            s = m.bounding_sphere
            self.assertTrue(s.volume > (m.volume-g.tol.merge))
            
        s_degenerate = g.trimesh.primitives.Sphere().bounding_sphere

        for d in [2,3]:
            for i in range(5):
                points = g.np.random.random((100,d))
                C,R = g.trimesh.nsphere.minimum_nsphere(points)
                R_check = ((points - C)**2).sum(axis=1).max() ** .5

                self.assertTrue(len(C) == d)
                self.assertTrue(R > 0.0)
                self.assertTrue(abs(R - R_check) < g.tol.merge)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
