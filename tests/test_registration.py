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

    def test_mesh(self):
        noise = .05
        extents = [6, 12, 3]

        # create the mesh as a simple box
        mesh = g.trimesh.creation.box(extents=extents)

        # subdivide until we have more faces than we want
        for i in range(3):
            mesh = mesh.subdivide()
        # apply tesselation and random noise
        mesh = mesh.permutate.noise(noise)
        # randomly rotation with translation
        transform = g.trimesh.transformations.random_rotation_matrix()
        transform[:3, 3] = (g.np.random.random(3) - .5) * 1000

        mesh.apply_transform(transform)

        scan = mesh
        # create a "true" mesh
        truth = g.trimesh.creation.box(extents=extents)

        for a, b in [[truth, scan], [scan, truth]]:
            a_to_b, cost = a.register(b)

            a_check = a.copy()
            a_check.apply_transform(a_to_b)

            assert g.np.linalg.norm(
                a_check.centroid -
                b.centroid) < (
                noise *
                2)

            # find the distance from the truth mesh to each scan vertex
            distance = a_check.nearest.on_surface(b.vertices)[1]
            assert distance.max() < (noise * 2)

        # try our registration with points
        points = g.trimesh.transform_points(
            scan.sample(100),
            matrix=g.trimesh.transformations.random_rotation_matrix())
        truth_to_points, cost = truth.register(points)
        truth.apply_transform(truth_to_points)
        distance = truth.nearest.on_surface(points)[1]

        assert distance.mean() < noise


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
