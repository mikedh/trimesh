import generic as g


class PointsTest(g.unittest.TestCase):

    def test_pointcloud(self):
        '''
        Test pointcloud object
        '''
        shape = (100, 3)
        cloud = g.trimesh.points.PointCloud(g.np.random.random(shape))

        self.assertTrue(cloud.vertices.shape == shape)
        self.assertTrue(cloud.extents.shape == (3,))
        self.assertTrue(cloud.bounds.shape == (2, 3))


    def test_vertexonly(self):
        """
        Test to make sure we can instantiate a mesh with just vertices
        for some reason
        """
        
        v = g.np.random.random((1000,3))
        v[g.np.floor(g.np.random.random(90)*len(v)).astype(int)] = v[0]

        mesh = g.trimesh.Trimesh(v)

        assert len(mesh.vertices) < 950
        assert len(mesh.vertices) > 900
        

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
