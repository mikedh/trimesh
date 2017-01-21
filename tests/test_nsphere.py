import generic as g


class NSphereTest(g.unittest.TestCase):

    def test_minball(self):

        # get some assorted mesh geometries to test general performance
        # and a perfect sphere mesh to test the degenerate case
        for m in g.np.append(g.get_meshes(5),
                             g.trimesh.primitives.Sphere()):

            s = m.bounding_sphere
            R_check = ((m.vertices - s.primitive.center)
                       ** 2).sum(axis=1).max() ** .5

            self.assertTrue(len(s.primitive.center) == 3)
            self.assertTrue(s.primitive.radius > 0.0)
            self.assertTrue(abs(s.primitive.radius - R_check) < g.tol.fit)
            self.assertTrue(s.volume > (m.volume - g.tol.fit))

        # check minimum n-sphere for sets of points in 2,3, and 4 dimensions
        for d in [2, 3, 4]:
            for i in range(5):
                points = g.np.random.random((100, d))
                C, R = g.trimesh.nsphere.minimum_nsphere(points)
                R_check = ((points - C)**2).sum(axis=1).max() ** .5
                self.assertTrue(len(C) == d)
                self.assertTrue(R > 0.0)
                self.assertTrue(abs(R - R_check) < g.tol.merge)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
