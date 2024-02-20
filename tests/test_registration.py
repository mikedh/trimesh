try:
    from . import generic as g
except BaseException:
    import generic as g


class RegistrationTest(g.unittest.TestCase):
    def test_procrustes(self):
        # every combination of possible boolean options
        # a_flip and a_scale are apply-to-test-data
        opt = list(g.itertools.combinations([True, False] * 6, 6))
        # generate deterministic but random-ish transforms
        matrices = iter(g.random_transforms(count=len(opt)))

        for reflection, translation, scale, a_flip, a_scale, weight in opt:
            # create random points in space
            points_a = (g.random((1000, 3)) - 0.5) * 1000
            # get a random transform from the iterator
            matrix = next(matrices)
            # apply a flip (reflection) to test data
            if a_flip:
                matrix = g.np.dot(
                    matrix,
                    g.trimesh.transformations.reflection_matrix(
                        point=[0, 0, 0], normal=[0, 1, 0]
                    ),
                )
            # apply scale to test data
            if a_scale:
                matrix = g.np.dot(
                    matrix, g.trimesh.transformations.scale_matrix(0.1235234)
                )
            # apply transform to points A
            points_b = g.trimesh.transform_points(points_a, matrix)

            # weight points or not
            if weight:
                weights = (g.random(len(points_a)) + 9) / 10
            else:
                weights = None

            # run the solver
            (matrixN, transformed, cost) = g.trimesh.registration.procrustes(
                points_a,
                points_b,
                reflection=reflection,
                translation=translation,
                scale=scale,
                weights=weights,
            )
            # if we're not weighting the results
            # should be identical with None vs all-ones
            (matrixN_C, transformed_C, cost_C) = g.trimesh.registration.procrustes(
                points_a,
                points_b,
                reflection=reflection,
                translation=translation,
                scale=scale,
                weights=g.np.ones(len(points_a)),
            )
            if weight:
                # weights should have changed the matrix
                assert not g.np.allclose(matrixN, matrixN_C)
            else:
                # no weights so everything should be identical
                assert g.np.allclose(matrixN, matrixN_C)
                assert g.np.allclose(transformed_C, transformed)
                assert g.np.isclose(cost, cost_C)

            # the points should be identical if the function
            # was allowed to translate in space
            # and there were no weights, scaling, or reflection
            identical = (
                translation
                and (not weight)
                and (not a_flip or reflection)
                and (not a_scale or scale)
            )

            if identical:
                assert cost < 0.001
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
        matrix, transformed, cost = g.trimesh.registration.icp(X, m, scale=False)
        assert cost < 0.01

    def test_icp_points(self):
        # see if ICP alignment works with point clouds
        # create random points in space
        points_a = (g.random((1000, 3)) - 0.5) * 1000
        # create a random transform
        # matrix = g.trimesh.transformations.random_rotation_matrix()
        # create a small transform
        # ICP will not work at all with large transforms
        matrix = g.trimesh.transformations.rotation_matrix(g.np.radians(1.0), [0, 0, 1])

        # take a few randomly chosen points and make
        # sure the order is permutated
        index = g.np.random.choice(g.np.arange(len(points_a)), 20)
        # transform and apply index
        points_b = g.trimesh.transform_points(points_a[index], matrix)
        # tun the solver
        matrixN, transformed, cost = g.trimesh.registration.icp(points_b, points_a)
        assert cost < 1e-3
        assert g.np.allclose(matrix, g.np.linalg.inv(matrixN))
        assert g.np.allclose(transformed, points_a[index])

    def test_mesh(self):
        noise = 0.05
        extents = [6, 12, 3]

        # create the mesh as a simple box
        mesh = g.trimesh.creation.box(extents=extents)

        # subdivide until we have more faces than we want
        for _i in range(3):
            mesh = mesh.subdivide()
        # apply tessellation and random noise
        mesh = mesh.permutate.noise(noise)
        # randomly rotation with translation
        transform = g.trimesh.transformations.random_rotation_matrix()
        transform[:3, 3] = (g.random(3) - 0.5) * 1000

        mesh.apply_transform(transform)

        scan = mesh
        # create a "true" mesh
        truth = g.trimesh.creation.box(extents=extents)

        for a, b in [[truth, scan], [scan, truth]]:
            a_to_b, cost = a.register(b)

            a_check = a.copy()
            a_check.apply_transform(a_to_b)

            assert g.np.linalg.norm(a_check.centroid - b.centroid) < (noise * 2)

            # find the distance from the truth mesh to each scan vertex
            distance = a_check.nearest.on_surface(b.vertices)[1]
            assert distance.max() < (noise * 2)

        # try our registration with points
        points = g.trimesh.transform_points(
            scan.sample(100), matrix=g.trimesh.transformations.random_rotation_matrix()
        )
        truth_to_points, cost = truth.register(points)
        truth.apply_transform(truth_to_points)
        distance = truth.nearest.on_surface(points)[1]

        assert distance.mean() < noise

    def test_nricp(self):
        # Get two meshes that have a comparable shape
        source = g.get_mesh("reference.obj", process=False)
        target = g.get_mesh("target.obj", process=False)

        # Vertex indices of landmarks source / target
        landmarks_vertex_indices = g.np.array(
            [
                [177, 1633],
                [181, 1561],
                [614, 1556],
                [610, 1629],
                [114, 315],
                [398, 413],
                [812, 412],
                [227, 99],
                [241, 87],
                [674, 86],
                [660, 98],
                [362, 574],
                [779, 573],
            ]
        )

        source_markers_vertices = source.vertices[landmarks_vertex_indices[:, 0]]
        target_markers_vertices = target.vertices[landmarks_vertex_indices[:, 1]]

        T = g.trimesh.registration.procrustes(
            source_markers_vertices, target_markers_vertices
        )[0]
        source.vertices = g.trimesh.transformations.transform_points(source.vertices, T)

        # Just for the sake of using barycentric coordinates...
        use_barycentric_coordinates = True
        if use_barycentric_coordinates:
            source_markers_vertices = source.vertices[landmarks_vertex_indices[:, 0]]
            source_markers_tids = g.trimesh.proximity.closest_point(
                source, source_markers_vertices
            )[2]
            source_markers_barys = g.trimesh.triangles.points_to_barycentric(
                source.triangles[source_markers_tids], source_markers_vertices
            )
            source_landmarks = (source_markers_tids, source_markers_barys)
        else:
            source_landmarks = landmarks_vertex_indices[:, 0]
        # Make copies to check nothing is modified
        source_copy = source.copy()
        target_copy = target.copy()

        # Parameters for nricp_amberg
        wl = 3
        max_iter = 10
        wn = 0.5
        steps_amberg = [
            # ws, wl, wn, max_iter
            [0.02, wl, wn, max_iter],
            [0.007, wl, wn, max_iter],
            [0.002, wl, wn, max_iter],
        ]

        # Parameters for nricp_sumner
        wl = 1000
        wi = 0.0001
        ws = 1
        wn = 1
        steps_sumner = [
            # wc, wi, ws, wl, wn
            [0.1, wi, ws, wl, wn],
            [0.1, wi, ws, wl, wn],
            [1, wi, ws, wl, wn],
            [5, wi, ws, wl, wn],
            [10, wi, ws, wl, wn],
            [50, wi, ws, wl, wn],
            [100, wi, ws, wl, wn],
            [1000, wi, ws, wl, wn],
            [10000, wi, ws, wl, wn],
        ]
        # Amberg et. al 2007
        records_amberg_no_ldm = g.trimesh.registration.nricp_amberg(
            source,
            target,
            distance_threshold=0.05,
            steps=steps_amberg,
            return_records=True,
        )
        records_amberg_ldm = g.trimesh.registration.nricp_amberg(
            source,
            target,
            source_landmarks=source_landmarks,
            target_positions=target_markers_vertices,
            steps=steps_amberg,
            return_records=True,
            distance_threshold=0.05,
        )
        try:
            g.trimesh.registration.nricp_amberg(source, target)
        except KeyError:
            raise AssertionError()  # related to #1724

        # Sumner and Popovic 2004
        records_sumner_no_ldm = g.trimesh.registration.nricp_sumner(
            source,
            target,
            distance_threshold=0.05,
            steps=steps_sumner,
            return_records=True,
        )
        records_sumner_ldm = g.trimesh.registration.nricp_sumner(
            source,
            target,
            source_landmarks=source_landmarks,
            target_positions=target_markers_vertices,
            steps=steps_sumner,
            return_records=True,
            distance_threshold=0.05,
        )
        try:
            g.trimesh.registration.nricp_sumner(source, target)
        except KeyError:
            raise AssertionError()  # related to #1724

        d_amberg_no_ldm = g.trimesh.proximity.closest_point(
            target, records_amberg_no_ldm[-1]
        )[1]
        d_amberg_ldm = g.trimesh.proximity.closest_point(target, records_amberg_ldm[-1])[
            1
        ]
        d_sumner_no_ldm = g.trimesh.proximity.closest_point(
            target, records_sumner_no_ldm[-1]
        )[1]
        d_sumner_ldm = g.trimesh.proximity.closest_point(target, records_sumner_ldm[-1])[
            1
        ]

        # Meshes should remain untouched
        assert g.np.allclose(source.vertices, source_copy.vertices)
        assert g.np.allclose(target.vertices, target_copy.vertices)

        # Basically a lot of check to check if the return values change somehow
        assert d_amberg_no_ldm.min() < 1e-7
        assert d_amberg_no_ldm.max() > 0.05
        assert d_amberg_no_ldm.mean() < 1e-3

        assert d_amberg_ldm.min() > 1e-7
        assert d_amberg_ldm.max() < 0.06
        assert d_amberg_ldm.mean() < 1e-3

        assert d_sumner_no_ldm.min() < 1e-8
        assert d_sumner_no_ldm.max() < 0.03
        assert d_sumner_no_ldm.mean() < 1e-3

        assert d_sumner_ldm.min() < 1e-8
        assert d_sumner_ldm.max() > 0.05
        assert d_sumner_ldm.mean() < 1e-3

        dl_amberg_no_ldm = g.np.linalg.norm(
            records_amberg_no_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )
        dl_amberg_ldm = g.np.linalg.norm(
            records_amberg_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )
        dl_sumner_no_ldm = g.np.linalg.norm(
            records_sumner_no_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )
        dl_sumner_ldm = g.np.linalg.norm(
            records_sumner_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )

        assert dl_amberg_no_ldm.min() > 0.01
        assert dl_amberg_no_ldm.max() > 0.1
        assert dl_amberg_no_ldm.mean() > 0.05

        assert dl_amberg_ldm.min() < 0.0002
        assert dl_amberg_ldm.max() < 0.001
        assert dl_amberg_ldm.mean() < 0.0004

        assert dl_sumner_no_ldm.min() > 0.01
        assert dl_sumner_no_ldm.max() > 0.1
        assert dl_sumner_no_ldm.mean() > 0.05

        assert dl_sumner_ldm.min() < 0.0007
        assert dl_sumner_ldm.max() < 0.006
        assert dl_sumner_ldm.mean() < 0.004

    def test_query_from_points(self):
        points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        query_point = [[0, 0.5, 0]]
        qres = g.trimesh.registration._from_points(
            points, query_point, return_normals=False
        )
        assert qres["vertex_indices"][0] == 1
        assert g.np.all(qres["nearest"][0] == [0, 1, 0])
        assert "normals" not in qres
        qres = g.trimesh.registration._from_points(
            points, query_point, return_normals=True
        )
        normal = g.np.ones(3)
        normal = normal / g.np.linalg.norm(normal)
        assert g.np.allclose(qres["normals"][0], normal) or g.np.allclose(
            qres["normals"][0], -normal
        )

    def test_query_from_mesh(self):
        points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        faces = [[0, 1, 2]]
        mesh = g.trimesh.Trimesh(vertices=points, faces=faces)
        query_point = [[0, 0.5, 0]]
        qres = g.trimesh.registration._from_mesh(
            mesh,
            query_point,
            return_barycentric_coordinates=False,
            return_normals=False,
            return_interpolated_normals=False,
        )

        assert "normals" not in qres
        assert "barycentric_coordinates" not in qres
        assert "interpolated_normals" not in qres

        qres = g.trimesh.registration._from_mesh(
            mesh,
            query_point,
            return_barycentric_coordinates=False,
            return_normals=False,
            return_interpolated_normals=True,
        )

        assert "normals" not in qres
        assert "barycentric_coordinates" in qres
        assert "interpolated_normals" in qres


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
