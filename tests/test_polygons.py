try:
    from . import generic as g
except BaseException:
    import generic as g


class PolygonTests(g.unittest.TestCase):

    def test_edges(self):
        """
        Test edges_to_polygon
        """
        m = g.get_mesh('featuretype.STL')

        # get a polygon for the second largest facet
        index = m.facets_area.argsort()[-2]
        normal = m.facets_normal[index]
        origin = m._cache['facets_origin'][index]
        T = g.trimesh.geometry.plane_transform(origin, normal)
        vertices = g.trimesh.transform_points(m.vertices, T)[:, :2]

        # find boundary edges for the facet
        edges = m.edges_sorted.reshape(
            (-1, 6))[m.facets[index]].reshape((-1, 2))
        group = g.trimesh.grouping.group_rows(edges, require_count=1)

        # run the polygon conversion
        polygon = g.trimesh.path.polygons.edges_to_polygons(
            edges=edges[group],
            vertices=vertices)

        assert len(polygon) == 1
        assert g.np.isclose(polygon[0].area,
                            m.facets_area[index])

        # try transforming the polygon around
        M = g.np.eye(3)
        M[0][2] = 10.0
        P2 = g.trimesh.path.polygons.transform_polygon(polygon[0], M)
        distance = g.np.array(P2.centroid) - g.np.array(polygon[0].centroid)
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

        radius = (s ** 2).sum(axis=1).max()
        assert radius < (1.0 + 1e-8)

        # test Path2D sample wiring
        path = g.trimesh.load_path(p)
        s = path.sample(count=count)
        assert len(s) <= count
        assert s.shape[1] == 2
        radius = (s ** 2).sum(axis=1).max()
        assert radius < (1.0 + 1e-8)

        # try getting OBB of samples
        T, extents = g.trimesh.path.polygons.polygon_obb(s)
        # OBB of samples should be less than diameter of circle
        diameter = g.np.reshape(p.bounds, (2, 2)).ptp(axis=0).max()
        assert (extents <= diameter).all()

        # test sampling with multiple bodies
        for i in range(3):
            assert g.np.isclose(path.area, p.area * (i + 1))
            path = path + g.trimesh.load_path(
                g.Point([(i + 2) * 2, 0]).buffer(1.0))
            s = path.sample(count=count)
            assert s.shape[1] == 2

    def test_project(self):
        m = g.trimesh.creation.icosphere(subdivisions=4)

        p = [g.trimesh.path.polygons.projected(m, normal=n)
             for n in g.np.random.random((100, 3))]

        # sphere projection should never have interiors
        assert all(len(i.interiors) == 0 for i in p)
        # sphere projected area should always be close to pi
        assert g.np.allclose(
            [i.area for i in p], g.np.pi, atol=0.05)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
