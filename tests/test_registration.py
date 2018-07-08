try:
    from . import generic as g
except BaseException:
    import generic as g


class RegistrationTest(g.unittest.TestCase):

    def test_procrustes(self):
        # create random points in space
        points_a = (g.np.random.random((1000, 3)) - .5) * 1000
        # create a random transform
        matrix = g.trimesh.transformations.random_rotation_matrix()
        # add a translation component to transform
        matrix[:3, 3] = g.np.random.random(3) * 100
        # apply transform to points A
        points_b = g.trimesh.transform_points(points_a, matrix)

        # run the solver
        (matrixN,
         transformed,
         cost) = g.trimesh.registration.procrustes(points_a, points_b)
        # the points should be identical
        assert(cost < 0.01)
        # it should have found the matrix we used
        assert g.np.allclose(matrixN, matrix)

    def test_icp_mesh(self):
        # see if ICP alignment works with meshes
        m = g.trimesh.creation.box()
        X = m.sample(10)
        X = X + [0.1, 0.1, 0.1]
        matrix, transformed, cost = g.trimesh.registration.icp(
            X, m, scale=False)
        assert(cost < 0.01)

    def test_icp_points(self):
        # see if ICP alignment works with point clouds
        # create random points in space
        points_a = (g.np.random.random((1000, 3)) - .5) * 1000
        # create a random transform
        #matrix = g.trimesh.transformations.random_rotation_matrix()
        # create a small transform
        # ICP will not work at all with large transforms
        matrix = g.trimesh.transformations.rotation_matrix(
            g.np.radians(1.0),
            [0, 0, 1])

        # take a few randomly chosen points and make
        # sure the order is permutated
        index = g.np.random.choice(g.np.arange(len(points_a)), 20)
        # transform and apply index
        points_b = g.trimesh.transform_points(points_a[index], matrix)
        # tun the solver
        matrixN, transformed, cost = g.trimesh.registration.icp(points_b,
                                                                points_a)
        assert cost < 1e-3
        assert g.np.allclose(matrix,
                             g.np.linalg.inv(matrixN))
        assert g.np.allclose(transformed, points_a[index])


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
