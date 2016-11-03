import generic as g

class NearestTest(g.unittest.TestCase):

    def test_naive(self):
        '''
        Test the naive nearest point function
        '''

        # generate a unit sphere mesh
        sphere = g.trimesh.primitives.Sphere(subdivisions = 4)
        
        # randomly sample surface of a unit sphere, then expand to radius 2.0
        points = g.trimesh.sample.sample_surface_sphere(100) * 2

        # use the triangles from the unit sphere
        triangles = sphere.triangles

        # do the check
        closest, distance, tid = g.trimesh.nearest.closest_point_naive(triangles, points)

        # the distance from a sphere of radius 1.0 to a sphere of radius 2.0
        # should be pretty darn close to 1.0
        self.assertTrue((g.np.abs(distance - 1.0) < .01).all())

        # the vector for the closest point should be the same as the vector 
        # to the query point
        vector = g.trimesh.util.diagonal_dot(closest, points/2.0)
        self.assertTrue((g.np.abs(vector - 1.0) < .01).all())

    def test_helper(self):
        # just make sure the plumbing returns something
        for mesh in g.get_meshes(2):
            points = (g.np.random.random((100,3)) - .5) * 100

            a = mesh.nearest.on_surface(points)
            self.assertTrue(a is not None)

            b = mesh.nearest.vertex(points)
            self.assertTrue(b is not None)

            


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
