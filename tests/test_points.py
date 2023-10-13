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
        points = g.random(shape)
        # make sure randomness never gives duplicates by offsetting
        points += g.np.arange(shape[0]).reshape((-1, 1))

        # make some duplicate vertices
        points[:10] = points[0]

        # create a pointcloud object
        cloud = g.trimesh.points.PointCloud(points)

        initial_hash = hash(cloud)

        assert cloud.convex_hull.volume > 0.0

        # check shapes of data
        assert cloud.vertices.shape == shape
        assert cloud.shape == shape
        assert cloud.extents.shape == (3,)
        assert cloud.bounds.shape == (2, 3)

        assert hash(cloud) == initial_hash

        # set some random colors
        cloud.colors = g.random((shape[0], 4))
        # remove the duplicates we created
        cloud.merge_vertices()

        # new shape post- merge
        new_shape = (shape[0] - 9, shape[1])

        # make sure vertices and colors are new shape
        assert cloud.vertices.shape == new_shape
        assert len(cloud.colors) == new_shape[0]
        assert hash(cloud) != initial_hash

        # AABB volume should be same as points
        assert g.np.isclose(cloud.bounding_box.volume, g.np.prod(points.ptp(axis=0)))

        # will populate all bounding primitives
        assert cloud.bounding_primitive.volume > 0.0
        # ... except AABB (it uses OBB)
        assert cloud.bounding_box.volume > 0.0

        # check getitem and setitem
        cloud[0] = [10, 10, 10]
        assert g.np.allclose(cloud[0], [10, 10, 10])
        # cloud should have copied
        assert not g.np.allclose(points[0], [10, 10, 10])

        # check to see if copy works
        assert g.np.allclose(cloud.vertices, cloud.copy().vertices)

    def test_empty(self):
        p = g.trimesh.PointCloud(None)
        assert p.is_empty
        assert p.copy().is_empty

        p.vertices = [[0, 1, 2]]
        assert not p.is_empty

    def test_vertex_only(self):
        """
        Test to make sure we can instantiate a mesh with just
        vertices and no faces for some unknowable reason
        """

        v = g.random((1000, 3))
        v[g.np.floor(g.random(90) * len(v)).astype(int)] = v[0]

        mesh = g.trimesh.Trimesh(v)

        assert len(mesh.vertices) < 950
        assert len(mesh.vertices) > 900

    def test_plane(self):
        # make sure plane fitting works for 2D points in space
        for _i in range(10):
            # create a random rotation
            matrix = g.trimesh.transformations.random_rotation_matrix()
            # create some random points in spacd
            p = g.random((1000, 3))
            # make them all lie on the XY plane so we know
            # the correct normal to check against
            p[:, 2] = 0
            # transform them into random frame
            p = g.trimesh.transform_points(p, matrix)
            # we made the Z values zero before transforming
            # so the true normal should be Z then rotated
            truth = g.trimesh.transform_points([[0, 0, 1]], matrix, translate=False)[0]
            # run the plane fit
            C, N = g.trimesh.points.plane_fit(p)
            # sign of normal is arbitrary on fit so check both
            assert g.np.allclose(truth, N) or g.np.allclose(truth, -N)
        # make sure plane fit works with multiple point sets at once
        nb_points_sets = 20
        for _i in range(10):
            # create a random rotation
            matrices = [
                g.trimesh.transformations.random_rotation_matrix()
                for _ in range(nb_points_sets)
            ]
            # create some random points in spacd
            p = g.random((nb_points_sets, 1000, 3))
            # make them all lie on the XY plane so we know
            # the correct normal to check against
            p[..., 2] = 0
            # transform them into random frame
            for j, matrix in enumerate(matrices):
                p[j, ...] = g.trimesh.transform_points(p[j, ...], matrix)
            # p = g.trimesh.transform_points(p, matrix)
            # we made the Z values zero before transforming
            # so the true normal should be Z then rotated
            truths = g.np.zeros((len(p), 3))
            for j, matrix in enumerate(matrices):
                truths[j, :] = g.trimesh.transform_points(
                    [[0, 0, 1]], matrix, translate=False
                )[0]
            # run the plane fit
            C, N = g.trimesh.points.plane_fit(p)

            # sign of normal is arbitrary on fit so check both
            cosines = g.np.einsum("ij,ij->i", N, truths)
            assert g.np.allclose(g.np.abs(cosines), g.np.ones_like(cosines))

    def test_kmeans(self, cluster_count=5, points_per_cluster=100):
        """
        Test K-means clustering
        """
        clustered = []

        # get random-ish points in a -0.5:+0.5 interval
        points = g.random((points_per_cluster * cluster_count, 3)) - 0.5
        for index, group in enumerate(g.np.array_split(points, cluster_count)):
            # move the cluster along the XYZ vector
            clustered.append((group / 1000) + (index * 100))
        clustered = g.np.vstack(clustered)

        # run k- means clustering on our nicely separated data
        centroids, klabel = g.trimesh.points.k_means(points=clustered, k=cluster_count)

        # reshape to make sure all groups have the same index
        variance = klabel.reshape((cluster_count, points_per_cluster)).ptp(axis=1)

        assert len(centroids) == cluster_count
        assert (variance == 0).all()

    def test_tsp(self):
        """
        Test our solution for visiting every point in order.
        """
        for dimension in [2, 3]:
            for count in [2, 10, 100]:
                for _i in range(10):
                    points = g.random((count, dimension))

                    # find a path that visits every point quickly
                    idx, dist = g.trimesh.points.tsp(points, start=0)

                    # indexes should visit every point exactly once
                    assert set(idx) == set(range(len(points)))
                    assert len(idx) == len(points)
                    assert len(dist) == len(points) - 1

                    # shouldn't be any negative indexes
                    assert (idx >= 0).all()

                    # make sure distances returned are correct
                    dist_check = g.np.linalg.norm(g.np.diff(points[idx], axis=0), axis=1)
                    assert g.np.allclose(dist_check, dist)

    def test_xyz(self):
        """
        Test XYZ file loading
        """
        # test a small file from cloudcompare
        p = g.get_mesh("points_cloudcompare.xyz")
        assert p.vertices.shape == (101, 3)
        assert p.colors.shape == (101, 4)

        # test a small file from agisoft
        p = g.get_mesh("points_agisoft.xyz")
        assert p.vertices.shape == (100, 3)
        assert p.colors.shape == (100, 4)

        # test exports
        e = p.export(file_type="xyz")
        p = g.trimesh.load(g.trimesh.util.wrap_as_stream(e), file_type="xyz")
        assert p.vertices.shape == (100, 3)
        assert p.colors.shape == (100, 4)

    def test_obb(self):
        p = g.get_mesh("points_agisoft.xyz")
        original = p.bounds.copy()
        matrix = p.apply_obb()
        assert matrix.shape == (4, 4)
        assert not g.np.allclose(p.bounds, original)

    def test_ply(self):
        p = g.get_mesh("points_agisoft.xyz")
        assert isinstance(p, g.trimesh.PointCloud)
        assert len(p.vertices) > 0

        # initial color CRC
        initial = hash(p.visual)
        # set to random colors
        p.colors = g.random((len(p.vertices), 4))
        # visual CRC should have changed
        assert hash(p.visual) != initial

        # test exporting a pointcloud to a PLY file
        r = g.wrapload(p.export(file_type="ply"), file_type="ply")
        assert r.vertices.shape == p.vertices.shape
        # make sure colors survived the round trip
        assert g.np.allclose(r.colors, p.colors)

    def test_glb(self):
        p = g.get_mesh("points_agisoft.xyz")
        assert isinstance(p, g.trimesh.PointCloud)
        assert len(p.vertices) > 0
        # test exporting a pointcloud to a GLTF
        # TODO : WE SHOULD IMPLEMENT THE IMPORTER TOO
        r = p.export(file_type="gltf")
        assert len(g.json.loads(r["model.gltf"].decode("utf-8"))["meshes"]) == 1

    def test_remove_close(self):
        # create 100 unique points
        p = g.np.arange(300).reshape((100, 3))
        # should return the original 100 points
        culled, mask = g.trimesh.points.remove_close(g.np.vstack((p, p)), radius=0.1)
        assert culled.shape == (100, 3)
        assert mask.shape == (200,)

    def test_add_operator(self):
        points_1 = g.random((10, 3))
        points_2 = g.random((20, 3))
        colors_1 = [[123, 123, 123, 255]] * len(points_1)
        colors_2 = [[255, 0, 123, 255]] * len(points_2)

        # Test: Both cloud's colors are preserved
        cloud_1 = g.trimesh.points.PointCloud(points_1, colors=colors_1)
        cloud_2 = g.trimesh.points.PointCloud(points_2, colors=colors_2)

        cloud_sum = cloud_1 + cloud_2
        assert g.np.allclose(
            cloud_sum.colors, g.np.vstack((cloud_1.colors, cloud_2.colors))
        )

        # Next test: Only second cloud has colors
        cloud_1 = g.trimesh.points.PointCloud(points_1)
        cloud_2 = g.trimesh.points.PointCloud(points_2, colors=colors_2)

        cloud_sum = cloud_1 + cloud_2
        assert g.np.allclose(cloud_sum.colors[len(cloud_1.vertices) :], cloud_2.colors)

        # Next test: Only first cloud has colors
        cloud_1 = g.trimesh.points.PointCloud(points_1, colors=colors_1)
        cloud_2 = g.trimesh.points.PointCloud(points_2)

        cloud_sum = cloud_1 + cloud_2
        assert g.np.allclose(cloud_sum.colors[: len(cloud_1.vertices)], cloud_1.colors)

    def test_radial_sort(self):
        theta = g.np.linspace(0.0, g.np.pi * 2.0, 1000)
        points = g.np.column_stack(
            (g.np.cos(theta), g.np.sin(theta), g.np.zeros(len(theta)))
        )
        points *= g.random(len(theta)).reshape((-1, 1))

        # apply a random order to the points
        order = g.np.random.permutation(g.np.arange(len(points)))

        # get the sorted version of these points
        # when we pass them the randomly ordered points
        sort = g.trimesh.points.radial_sort(
            points[order], origin=[0, 0, 0], normal=[0, 0, 1]
        )
        # should have re-established original order
        assert g.np.allclose(points, sort)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
