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


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
