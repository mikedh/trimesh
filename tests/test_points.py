try:
    from . import generic as g
except BaseException:
    import generic as g


class PointsTest(g.unittest.TestCase):

    def test_pointcloud(self):
        """
        Test PointCloud object
        """
        shape = (100, 3)
        # random points
        points = g.np.random.random(shape)
        # make sure randomness never gives duplicates by offsetting
        points += g.np.arange(shape[0]).reshape((-1, 1))

        # make some duplicate vertices
        points[:10] = points[0]

        # create a pointcloud object
        cloud = g.trimesh.points.PointCloud(points)

        initial_md5 = cloud.md5()

        assert cloud.convex_hull.volume > 0.0

        # check shapes of data
        assert cloud.vertices.shape == shape
        assert cloud.shape == shape
        assert cloud.extents.shape == (3,)
        assert cloud.bounds.shape == (2, 3)

        assert cloud.md5() == initial_md5

        # set some random colors
        cloud.colors = g.np.random.random((shape[0], 4))
        # remove the duplicates we created
        cloud.merge_vertices()

        # new shape post- merge
        new_shape = (shape[0] - 9, shape[1])

        # make sure vertices and colors are new shape
        assert cloud.vertices.shape == new_shape
        assert len(cloud.colors) == new_shape[0]
        assert cloud.md5() != initial_md5

        # AABB volume should be same as points
        assert g.np.isclose(cloud.bounding_box.volume,
                            g.np.product(points.ptp(axis=0)))

        # will populate all bounding primitives
        assert cloud.bounding_primitive.volume > 0.0
        # ... except AABB (it uses OBB)
        assert cloud.bounding_box.volume > 0.0

        # check getitem and setitem
        cloud[0] = [10, 10, 10]
        assert g.np.allclose(cloud[0], [10, 10, 10])
        # cloud should have copied
        assert not g.np.allclose(points[0], [10, 10, 10])

    def test_empty(self):
        p = g.trimesh.PointCloud(None)
        assert p.is_empty

        p.vertices = [[0, 1, 2]]
        assert not p.is_empty

    def test_vertex_only(self):
        """
        Test to make sure we can instantiate a mesh with just
        vertices and no faces for some unknowable reason
        """

        v = g.np.random.random((1000, 3))
        v[g.np.floor(g.np.random.random(90) * len(v)).astype(int)] = v[0]

        mesh = g.trimesh.Trimesh(v)

        assert len(mesh.vertices) < 950
        assert len(mesh.vertices) > 900

    def test_plane(self):
        # make sure plane fitting works for 2D points in space
        for i in range(10):
            # create a random rotation
            matrix = g.trimesh.transformations.random_rotation_matrix()
            # create some random points in spacd
            p = g.np.random.random((1000, 3))
            # make them all lie on the XY plane so we know
            # the correct normal to check against
            p[:, 2] = 0
            # transform them into random frame
            p = g.trimesh.transform_points(p, matrix)
            # we made the Z values zero before transforming
            # so the true normal should be Z then rotated
            truth = g.trimesh.transform_points([[0, 0, 1]],
                                               matrix,
                                               translate=False)[0]
            # run the plane fit
            C, N = g.trimesh.points.plane_fit(p)

            # sign of normal is arbitrary on fit so check both
            assert g.np.allclose(truth, N) or g.np.allclose(truth, -N)

    def test_kmeans(self,
                    cluster_count=5,
                    points_per_cluster=100):
        """
        Test K-means clustering
        """
        clustered = []
        for i in range(cluster_count):
            # use repeatable random- ish coordinatez
            clustered.append(
                g.random((points_per_cluster, 3)) + (i * 10.0))
        clustered = g.np.vstack(clustered)

        # run k- means clustering on our nicely separated data
        centroids, klabel = g.trimesh.points.k_means(points=clustered,
                                                     k=cluster_count)

        # reshape to make sure all groups have the same index
        variance = klabel.reshape(
            (cluster_count, points_per_cluster)).ptp(
            axis=1)

        assert len(centroids) == cluster_count
        assert (variance == 0).all()

    def test_tsp(self):
        """
        Test our solution for visiting every point in order.
        """
        for dimension in [2, 3]:
            for count in [2, 10, 100]:
                for i in range(10):
                    points = g.np.random.random((count, dimension))

                    # find a path that visits every point quickly
                    idx, dist = g.trimesh.points.tsp(points, start=0)

                    # indexes should visit every point exactly once
                    assert set(idx) == set(range(len(points)))
                    assert len(idx) == len(points)
                    assert len(dist) == len(points) - 1

                    # shouldn't be any negative indexes
                    assert (idx >= 0).all()

                    # make sure distances returned are correct
                    dist_check = g.np.linalg.norm(
                        g.np.diff(points[idx], axis=0), axis=1)
                    assert g.np.allclose(dist_check, dist)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
