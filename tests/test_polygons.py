try:
    from . import generic as g
except BaseException:
    import generic as g


class PolygonTests(g.unittest.TestCase):
    def test_edges(self):
        """
        Test edges_to_polygon
        """
        m = g.get_mesh("featuretype.STL")

        # get a polygon for the second largest facet
        index = m.facets_area.argsort()[-2]
        normal = m.facets_normal[index]
        origin = m._cache["facets_origin"][index]
        T = g.trimesh.geometry.plane_transform(origin, normal)
        vertices = g.trimesh.transform_points(m.vertices, T)[:, :2]

        # find boundary edges for the facet
        edges = m.edges_sorted.reshape((-1, 6))[m.facets[index]].reshape((-1, 2))
        group = g.trimesh.grouping.group_rows(edges, require_count=1)

        # run the polygon conversion
        polygon = g.trimesh.path.polygons.edges_to_polygons(
            edges=edges[group], vertices=vertices
        )

        assert len(polygon) == 1
        assert g.np.isclose(polygon[0].area, m.facets_area[index])

        # try transforming the polygon around
        M = g.np.eye(3)
        M[0][2] = 10.0
        P2 = g.trimesh.path.polygons.transform_polygon(polygon[0], M)
        distance = (
            g.np.array(P2.centroid.coords)[0] - g.np.array(polygon[0].centroid.coords)[0]
        )
        assert g.np.allclose(distance, [10.0, 0])

    def test_random_polygon(self):
        """
        Test creation of random polygons
        """
        p = g.trimesh.path.polygons.random_polygon()
        assert p.area > 0.0
        assert p.is_valid

    def test_sample(self):
        """
        Test random sampling of polygons
        """

        p = g.Point([0, 0]).buffer(1.0)
        count = 100

        s = g.trimesh.path.polygons.sample(p, count=count)
        assert len(s) <= count
        assert s.shape[1] == 2

        radius = (s**2).sum(axis=1).max()
        assert radius < (1.0 + 1e-8)

        # test Path2D sample wiring
        path = g.trimesh.load_path(p)
        s = path.sample(count=count)
        assert len(s) <= count
        assert s.shape[1] == 2
        radius = (s**2).sum(axis=1).max()
        assert radius < (1.0 + 1e-8)

        # try getting OBB of samples
        T, extents = g.trimesh.path.polygons.polygon_obb(s)
        # OBB of samples should be less than diameter of circle
        diameter = g.np.reshape(p.bounds, (2, 2)).ptp(axis=0).max()
        assert (extents <= diameter).all()

        # test sampling with multiple bodies
        for i in range(3):
            assert g.np.isclose(path.area, p.area * (i + 1))
            path = path + g.trimesh.load_path(g.Point([(i + 2) * 2, 0]).buffer(1.0))
            s = path.sample(count=count)
            assert s.shape[1] == 2

    def test_project(self):
        m = g.trimesh.creation.icosphere(subdivisions=4)

        p = [g.trimesh.path.polygons.projected(m, normal=n) for n in g.random((100, 3))]

        # sphere projection should never have interiors
        assert all(len(i.interiors) == 0 for i in p)
        # sphere projected area should always be close to pi
        assert g.np.allclose([i.area for i in p], g.np.pi, atol=0.05)

    def test_project_backface(self):
        m = g.trimesh.Trimesh(
            vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], faces=[[0, 1, 2]]
        )

        # check ignore_sign argument
        front = m.projected(m.face_normals[0], ignore_sign=False)
        assert len(front.entities) == 1
        front = m.projected(m.face_normals[0], ignore_sign=True)
        assert len(front.entities) == 1

        back = m.projected(-m.face_normals[0], ignore_sign=False)
        assert len(back.entities) == 0
        back = m.projected(-m.face_normals[0], ignore_sign=True)
        assert len(back.entities) == 1

    def test_project_multi(self):
        mesh = g.trimesh.creation.box() + g.trimesh.creation.box().apply_translation(
            [3, 0, 0]
        )
        proj = mesh.projected(normal=[0, 0, 1])

        assert mesh.body_count == 2
        assert len(proj.root) == 2
        assert g.np.isclose(proj.area, 2.0)

    def test_second_moment(self):
        def rectangle(extents):
            # return the boundary of an origin-centered
            # rectangle as a numpy array
            a = g.np.abs(g.np.array(extents) / 2.0)
            lower, upper = -a, a
            return g.np.array([lower, [upper[0], lower[1]], upper, [lower[0], upper[1]]])

        def poly(bh, bhi=None):
            # return a rectangle centered at the origin
            # as a shapely Polygon
            shell = rectangle(bh)
            if bhi is not None:
                holes = [rectangle(bhi)]
            else:
                holes = []
            return Polygon(shell=shell, holes=holes)

        def poly_corner(bh):
            # return a rectangle with one corner
            # at the origin and the rest in positive space
            shell = rectangle(bh)
            shell += shell.min(axis=0)
            return Polygon(shell=shell)

        def poly_doublecorner(bh):
            # returns two equal sized rectangles as one polygon
            # Same as poly_corner(), but the rectangle gets
            # mirrored by 180 deg at the origin
            # This puts the centroid in the origin with Ixy != 0
            shell_1 = rectangle(bh)
            shell_1 += shell_1.min(axis=0)
            shell_2 = -shell_1
            shell = g.np.concatenate(
                (shell_1[2:, :], shell_1[:2, :], shell_2[2:, :], shell_2[:2, :]), axis=0
            )
            return Polygon(shell=shell)

        def truth(bh, bhi=None):
            # return the analytical second moment of area
            # for a rectangle with centroid at the origin
            # and width-height of `bh` and an interior
            # rectangle width-height of `bhi`
            if bhi is None:
                bhi = g.np.zeros(2)
            b, h = bh
            bi, hi = bhi
            return (
                g.np.array(
                    [b * h**3 - bi * hi**3, h * b**3 - hi * bi**3, 0.0],
                    dtype=g.np.float64,
                )
                / 12
            )

        def truth_corner(bh):
            # check a rectangle with one corner
            # at the origin and the rest in positive space
            b, h = bh
            return g.np.array(
                [b * h**3 / 3.0, h * b**3 / 3.0, 0.5 * b**2 * 0.5 * h**2],
                dtype=g.np.float64,
            )

        from shapely.geometry import Polygon

        from trimesh.path.polygons import second_moments, transform_polygon

        heights = g.np.array([[0.01, 0.01], [1, 1], [10, 2], [3, 21]])
        for bh in heights:
            # check the second moment of a rectangle
            # as polygon is already centered, centered doesn't have any effect
            O_moments, O_principal_moments, O_alpha, O_transform = second_moments(
                poly(bh), return_centered=True
            )
            # check against wikipedia
            t = truth(bh)
            # for a centered rectangle, the principal axis are already aligned
            # with the frame axis
            assert g.np.allclose(O_moments, t)
            assert g.np.any(g.np.isclose(O_moments, O_principal_moments[0]))
            assert g.np.isclose(O_moments[2], 0)  # Ixy = 0
            assert g.np.isclose(O_alpha, 0)
            assert g.np.allclose(O_transform, g.np.eye(3))

            # now check a rectangle with the corner, so Ixy != 0

            # First we test with centering. The results should be same as
            # with the initially centered rectangles
            C_moments, C_principal_moments, C_alpha, C_transform = second_moments(
                poly_corner(bh), return_centered=True
            )
            assert g.np.allclose(O_moments, C_moments)
            assert g.np.allclose(O_principal_moments, C_principal_moments)
            assert g.np.isclose(O_alpha, C_alpha)
            assert g.np.allclose(O_transform[:, :2], C_transform[:, :2])

            # Now without centering
            moments = second_moments(poly_corner(bh), return_centered=False)
            t = truth_corner(bh)
            assert g.np.allclose(moments, t)

            # Now we will get the transform for a double rectangle. Then we will apply
            # the transform and test if Ixy == 0, alpha == 0 etc.
            C_moments, C_principal_moments, C_alpha, C_transform = second_moments(
                poly_doublecorner(bh), return_centered=True
            )
            # apply the outputted transform to the polygon
            T_polygon = transform_polygon(poly_doublecorner(bh), C_transform)
            # call the function on the transformed polygon
            T_moments, T_principal_moments, T_alpha, T_transform = second_moments(
                T_polygon, return_centered=True
            )
            assert g.np.any(g.np.isclose(T_moments, C_principal_moments[0]))
            assert g.np.allclose(C_principal_moments, T_principal_moments)
            assert g.np.isclose(T_alpha, 0, atol=1e-7)
            assert g.np.allclose(T_transform, g.np.eye(3))

            # check polygons with interior
            for bhi in heights:
                # only check if interior is smaller than exterior
                if not (bhi < bh).all():
                    continue

                # check a rectangle with interiors
                c = second_moments(poly(bh, bhi), return_centered=False)
                t = truth(bh, bhi)
                assert g.np.allclose(c, t)

    def test_native_centroid(self):
        # check our native implementation of the shoelace algorithm
        # against the results from shapely which we use for native
        # checks of counter-clockwise 2D coordinate loops

        # get some polygons without interiors
        polygons = [g.Polygon(i) for i in g.data["nestable"]]
        # same data as (n, 2) float arrays
        coords = [g.np.array(i) for i in g.data["nestable"]]

        for p, c in zip(polygons, coords):
            # area will be signed with respect to counter-clockwise
            ccw, area, centroid = g.trimesh.util.is_ccw(c, return_all=True)
            assert g.np.allclose(centroid, g.np.array(p.centroid.coords)[0])
            assert g.np.isclose(abs(area), p.area)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
