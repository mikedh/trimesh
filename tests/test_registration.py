try:
    from . import generic as g
except BaseException:
    import generic as g


class RegistrationTest(g.unittest.TestCase):

    def test_procrustes(self):

        # every combination of possible boolean options
        # a_flip and a_scale are apply-to-test-data
        opt = g.itertools.combinations([True, False] * 5, 5)
        for reflection, translation, scale, a_flip, a_scale in opt:
            # create random points in space
            points_a = (g.np.random.random((1000, 3)) - .5) * 1000
            # create a random transform
            matrix = g.trimesh.transformations.random_rotation_matrix()
            # add a translation component to transform
            matrix[:3, 3] = g.np.random.random(3) * 100
            # apply a flip (reflection) to test data
            if a_flip:
                matrix = g.np.dot(
                    matrix,
                    g.trimesh.transformations.reflection_matrix(
                        point=[0, 0, 0], normal=[0, 1, 0]))
            # apply scale to test data
            if a_scale:
                matrix = g.np.dot(
                    matrix,
                    g.trimesh.transformations.scale_matrix(0.1235234))
            # apply transform to points A
            points_b = g.trimesh.transform_points(points_a, matrix)

            # run the solver
            (matrixN,
             transformed,
             cost) = g.trimesh.registration.procrustes(
                 points_a, points_b,
                 reflection=reflection,
                 translation=translation,
                 scale=scale)

            # the points should be identical if the function
            # was allowed to translate in space

            identical = (translation and
                         (not a_flip or reflection) and
                         (not a_scale or scale))

            if identical:
                assert(cost < 0.001)
                # it should have found the matrix we used
                assert g.np.allclose(matrixN, matrix)

            # if reflection is not allowed, the determinant
            # should always be close to 1.0 for the rotation
            det = g.np.linalg.det(matrixN[:3, :3])
            # reflection not allowed
            if not reflection:
                # determinant should always be positive
                assert det > 1e-8
                if not scale:
                    # no reflection or scale means determinant
                    # of 1.0
                    assert g.np.isclose(det, 1.0)

            # scaling not allowed
            if not scale:
                # allowed to be -1.0 or 1.0
                assert g.np.isclose(g.np.abs(det), 1.0)

            # points have a flip applied
            # procrustes is allowed to use reflection
            # and there is no scaling in the matrix
            if a_flip and reflection and not scale:
                assert g.np.isclose(det, -1.0)

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
        # matrix = g.trimesh.transformations.random_rotation_matrix()
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
        # apply tessellation and random noise
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

    def test_nricp(self):

        tooth = g.get_mesh('busted.STL', process=False)
        ball = g.get_mesh('ballA.off', process=False)

        # Center the meshes together
        tooth.vertices = ball.scale * (tooth.vertices - tooth.centroid[None, :])\
            / tooth.scale + ball.centroid

        # Make copies to check nothing is modified
        tooth_copy = tooth.copy()
        ball_copy = ball.copy()

        # First step set : wl = 0
        steps = [
            [0.02, 0, 0.5, 10],
            [0.007, 0.0, 0.5, 10],
            [0.002, 0.0, 0.0, 10],
        ]
        ball_landmarks = g.np.array([0], dtype=g.np.int32)
        tooth_landmarks = g.np.array([0], dtype=g.np.int32)
        result_no_ldm, records_no_ldm = g.trimesh.registration.nricp(
            ball, tooth, source_landmarks=ball_landmarks,
            target_landmarks=tooth_landmarks, steps=steps, return_records=True)

        assert len(records_no_ldm) >= 3
        assert g.np.allclose(tooth.vertices, tooth_copy.vertices)
        assert g.np.allclose(ball.vertices, ball_copy.vertices)
        assert records_no_ldm[-1][1].mean() < 0.00011
        assert not g.np.allclose(ball.vertices, result_no_ldm.vertices)

        # Second step set : wl > 0
        steps = [
            [0.02, 5, 0.5, 10],
            [0.007, 5, 0.5, 10],
            [0.002, 5, 0.0, 10],
        ]
        # Also test with target shape as a point cloud
        tooth_pc = g.trimesh.PointCloud(tooth.vertices)
        result_ldm, records_ldm = g.trimesh.registration.nricp(
            ball, tooth_pc, source_landmarks=ball_landmarks,
            target_landmarks=tooth_landmarks, steps=steps, return_records=True,
            use_faces=False)

        distance_no_ldm = g.np.linalg.norm(
                result_no_ldm.vertices[ball_landmarks[0]] -
                tooth.vertices[tooth_landmarks[0]],
                axis=-1)
        distance_ldm = g.np.linalg.norm(
                result_ldm.vertices[ball_landmarks[0]] -
                tooth.vertices[tooth_landmarks[0]],
                axis=-1)

        assert distance_no_ldm > 11.0
        assert distance_ldm < 0.03
        assert len(records_ldm) >= 3
        assert records_no_ldm[-1][1].mean() < records_ldm[-1][1].mean()
        assert not g.np.allclose(ball.vertices, result_ldm.vertices)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
