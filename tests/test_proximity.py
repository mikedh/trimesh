try:
    from . import generic as g
except BaseException:
    import generic as g


class NearestTest(g.unittest.TestCase):
    def test_naive(self):
        """
        Test the naive nearest point function
        """

        # generate a unit sphere mesh
        sphere = g.trimesh.primitives.Sphere(subdivisions=4)

        # randomly sample surface of a unit sphere, then expand to radius 2.0
        points = g.trimesh.sample.sample_surface_sphere(100) * 2

        # use the triangles from the unit sphere
        triangles = sphere.triangles  # NOQA

        # do the check
        closest, distance, tid = g.trimesh.proximity.closest_point_naive(sphere, points)

        # the distance from a sphere of radius 1.0 to a sphere of radius 2.0
        # should be pretty darn close to 1.0
        assert (g.np.abs(distance - 1.0) < 0.01).all()

        # the vector for the closest point should be the same as the vector
        # to the query point
        vector = g.trimesh.util.diagonal_dot(closest, points / 2.0)
        assert (g.np.abs(vector - 1.0) < 0.01).all()

    def test_helper(self):
        # just make sure the plumbing returns something
        for mesh in g.get_meshes(2):
            points = (g.random((100, 3)) - 0.5) * 100

            a = mesh.nearest.on_surface(points)
            assert a is not None

            b = mesh.nearest.vertex(points)
            assert b is not None

    def test_nearest_naive(self):
        funs = [
            g.trimesh.proximity.closest_point_naive,
            g.trimesh.proximity.closest_point,
        ]

        data_points = g.deque()
        data_dist = g.deque()

        tic = [g.time.time()]
        for i in funs:
            p, d = self.check_nearest_point_function(i)
            data_points.append(p)
            data_dist.append(d)
            tic.append(g.time.time())

        assert g.np.ptp(data_points, axis=0).max() < g.tol.merge
        assert g.np.ptp(data_dist, axis=0).max() < g.tol.merge

        log_msg = "\n".join(
            f"{i}: {j}s" for i, j in zip([i.__name__ for i in funs], g.np.diff(tic))
        )
        g.log.info("Compared the following nearest point functions:\n" + log_msg)

    def check_nearest_point_function(self, fun):
        # def plot_tri(tri, color='g'):
        #     plottable = g.np.vstack((tri, tri[0]))
        #     plt.plot(plottable[:, 0], plottable[:, 1], color=color)

        def points_on_circle(count):
            theta = g.np.linspace(0, g.np.pi * 2, count + 1)[:count]
            s = g.np.column_stack((theta, [g.np.pi / 2] * count))
            t = g.trimesh.util.spherical_to_vector(s)
            return t

        # generate some pseudorandom triangles
        # use our random to avoid spurious failures
        triangles = g.random((100, 3, 3)) - 0.5
        # put them on- plane
        triangles[:, :, 2] = 0.0

        # make one of the triangles equilaterial
        triangles[-1] = points_on_circle(3)

        # a circle of points surrounding the triangle
        query = points_on_circle(63) * 2
        # set the points up in space
        query[:, 2] = 10
        # a circle of points inside-ish the triangle
        query = g.np.vstack((query, query * 0.1))

        # loop through each triangle
        for triangle in triangles:
            # create a mesh with one triangle
            mesh = g.Trimesh(**g.trimesh.triangles.to_kwargs([triangle]))

            result, result_distance, result_tid = fun(mesh, query)

            polygon = g.Polygon(triangle[:, 0:2])
            polygon_buffer = polygon.buffer(1e-5)

            # all of the points returned should be on the triangle we're
            # querying
            assert all(polygon_buffer.intersects(g.Point(i)) for i in result[:, 0:2])

            # see what distance shapely thinks the nearest point
            # is for the 2D triangle and the query points
            distance_shapely = g.np.array(
                [polygon.distance(g.Point(i)) for i in query[:, :2]]
            )

            # see what distance our function returned for the nearest point
            distance_ours = ((query[:, :2] - result[:, :2]) ** 2).sum(axis=1) ** 0.5

            # how far was our distance from the one shapely gave
            distance_test = g.np.abs(distance_shapely - distance_ours)  # NOQA

            # we should have calculated the same distance as shapely
            assert g.np.allclose(distance_ours, distance_shapely)

        # now check to make sure closest point doesn't depend on
        # the frame, IE the results should be the same after
        # any rigid transform
        # chop query off to same length as triangles
        assert len(query) > len(triangles)
        query = query[: len(triangles)]
        # run the closest point query as a corresponding query
        close = g.trimesh.triangles.closest_point(triangles=triangles, points=query)
        # distance between closest point and query point
        # this should stay the same regardless of frame
        distance = g.np.linalg.norm(close - query, axis=1)
        for T in g.transforms:
            # transform the query points
            points = g.trimesh.transform_points(query, T)
            # transform the triangles we're checking
            tri = g.trimesh.transform_points(triangles.reshape((-1, 3)), T).reshape(
                (-1, 3, 3)
            )
            # run the closest point check
            check = g.trimesh.triangles.closest_point(triangles=tri, points=points)
            check_distance = g.np.linalg.norm(check - points, axis=1)
            # should be the same in any frame
            assert g.np.allclose(check_distance, distance)

        return result, result_distance

    def test_coplanar_signed_distance(self):
        mesh = g.trimesh.primitives.Box()

        # should be well outside the box but coplanar with a face
        # so the signed distance should be negative
        distance = mesh.nearest.signed_distance([mesh.bounds[0] + [100, 0, 0]])

        assert distance[0] < 0.0

        # constructed so origin is inside but also coplanar with
        # the nearest face
        mesh = g.get_mesh("origin_inside.STL")

        # origin should be inside, so distance should be positive
        distance = mesh.nearest.signed_distance([[0, 0, 0]])

        assert distance[0] > 0.0

    def test_noncoplanar_signed_distance(self):
        mesh = g.trimesh.primitives.Box()

        # should be well outside the box and not coplanar with a face
        # so the signed distance should be negative
        distance = mesh.nearest.signed_distance([mesh.bounds[0] + [100, 100, 100]])

        assert distance[0] < 0.0

    def test_edge_case(self):
        mesh = g.get_mesh("20mm-xyz-cube.stl")
        assert (mesh.nearest.signed_distance([[-51, 4.7, -20.6]]) < 0.0).all()

    def test_acute_edge_case(self):
        # acute tetrahedron with a sharp edge
        vertices = [[-1, 0.5, 0], [1, 0.5, 0], [0, -1, -0.5], [0, -1, 0.5]]
        faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]
        mesh = g.trimesh.Trimesh(vertices, faces)

        # a set of points on a line outside of the tetrahedron
        # their closest surface point is [0, 0.5, 0] on the sharp edge
        # for a point exactly in the middle a closest face is still ambiguous
        # -> take an even number of points
        n = 20
        n += n % 2
        pts = g.np.transpose([g.np.zeros(n), g.np.ones(n), g.np.linspace(-1, 1, n)])

        # the faces facing the points should differ for first and second half of the set
        # check their indices for inequality
        faceIdxsA, faceIdxsB = g.np.split(mesh.nearest.on_surface(pts)[-1], 2)
        assert (
            g.np.all(faceIdxsA == faceIdxsA[0])
            and g.np.all(faceIdxsB == faceIdxsB[0])
            and faceIdxsA[0] != faceIdxsB[0]
        )

    def test_candidates(self):
        mesh = g.trimesh.creation.random_soup(2000)
        points = g.random((2000, 3))
        g.trimesh.proximity.nearby_faces(mesh=mesh, points=points)

    def test_returns_correct_point_in_ambiguous_cases(self):
        mesh = g.trimesh.Trimesh(
            vertices=[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            faces=[[0, 1, 2], [0, 1, 3]],
            process=False,
        )
        # Query point is closer to face 0 but lies in face 1 normal direction
        closest, _, closest_face_index = g.trimesh.proximity.closest_point(
            mesh, [[-0.25 - 1e-9, 0.0, -0.25]]
        )
        g.np.testing.assert_almost_equal(closest[0], [0.0, 0.0, -0.25])
        self.assertEqual(closest_face_index[0], 1)

    def test_unreferenced_vertex(self):
        # check to see that meshes with unreferenced vertices
        # return correct values and ignore the unreferenced points
        query_point = [-1.0, -1.0, -1.0]
        mesh = g.trimesh.Trimesh(
            vertices=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-0.5, -0.5, -0.5],
            ],
            faces=[[0, 1, 2]],
            process=False,
        )

        proximity_query = g.trimesh.proximity.ProximityQuery(mesh)
        q = proximity_query.on_surface([query_point])
        assert len(q) == 3
        assert all(len(i) == 1 for i in q)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
