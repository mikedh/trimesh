import generic as g

class TrianglesTest(g.unittest.TestCase):
    def test_barycentric(self):
        for m in g.get_meshes(4):
            # a simple test which gets the barycentric coordinate at each of the three
            # vertices, checks to make sure the barycentric is [1,0,0] for the vertex
            # and then converts back to cartesian and makes sure the original points
            #  are the same as the conversion and back
            for method in ['cross', 'cramer']:
                for i in range(3):
                    barycentric = g.trimesh.triangles.points_to_barycentric(m.triangles, 
                                                                            m.triangles[:,i],
                                                                            method=method)
                    self.assertTrue((g.np.abs(barycentric - g.np.roll([1.0, 0, 0], i)) < 1e-8).all())

                    points = g.trimesh.triangles.barycentric_to_points(m.triangles,
                                                                       barycentric)
                    self.assertTrue((g.np.abs(points - m.triangles[:,i]) < 1e-8).all())

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
