try:
    from . import generic as g
except BaseException:
    import generic as g


class VectorTests(g.unittest.TestCase):

    def test_discrete(self):
        for d in g.get_2D():
            # store md5 before requesting passive functions
            md5 = d.md5()

            # make sure various methods return
            # basically the same bounds
            atol = d.scale / 1000
            for dis, pa, pl in zip(d.discrete,
                                   d.paths,
                                   d.polygons_closed):
                # bounds of discrete version of path
                bd = g.np.array([g.np.min(dis, axis=0),
                                 g.np.max(dis, axis=0)])
                # bounds of polygon version of path
                bl = g.np.reshape(pl.bounds, (2, 2))
                # try bounds of included entities from path
                pad = g.np.vstack([d.entities[i].discrete(d.vertices)
                                   for i in pa])
                bp = g.np.array([g.np.min(pad, axis=0),
                                 g.np.max(pad, axis=0)])

                assert g.np.allclose(bd, bl, atol=atol)
                assert g.np.allclose(bl, bp, atol=atol)

            # run some checks
            g.check_path2D(d)

            # copying shouldn't touch original file
            copied = d.copy()

            # these operations shouldn't have mutated anything!
            assert d.md5() == md5

            # copy should have saved the metadata
            assert set(copied.metadata.keys()) == set(d.metadata.keys())

            # file_name should be populated, and if we have a DXF file
            # the layer field should be populated with layer names
            if d.metadata['file_name'][-3:] == 'dxf':
                assert len(d.layers) == len(d.entities)

            for path in d.paths:
                verts = d.discretize_path(path)
                dists = g.np.sum((g.np.diff(verts, axis=0))**2, axis=1)**.5

                if not g.np.all(dists > g.tol_path.zero):
                    raise ValueError('{} had zero distance in discrete!',
                                     d.metadata['file_name'])

                circuit_dist = g.trimesh.util.euclidean(verts[0], verts[-1])
                circuit_test = circuit_dist < g.tol_path.merge
                if not circuit_test:
                    g.log.error('On file %s First and last vertex distance %f',
                                d.metadata['file_name'],
                                circuit_dist)
                assert circuit_test

                is_ccw = g.trimesh.path.util.is_ccw(verts)
                if not is_ccw:
                    g.log.error('discrete %s not ccw!',
                                d.metadata['file_name'])

            for i in range(len(d.paths)):
                assert d.polygons_closed[i].is_valid
                assert d.polygons_closed[i].area > g.tol_path.zero
            export_dict = d.export(file_type='dict')
            to_dict = d.to_dict()
            assert isinstance(to_dict, dict)
            assert isinstance(export_dict, dict)
            assert len(to_dict) == len(export_dict)

            export_svg = d.export(file_type='svg')  # NOQA
            simple = d.simplify()  # NOQA
            split = d.split()
            g.log.info('Split %s into %d bodies, checking identifiers',
                       d.metadata['file_name'],
                       len(split))
            for body in split:
                body.identifier

            if len(d.root) == 1:
                d.apply_obb()

            # store the X values of bounds
            ori = d.bounds.copy()
            # apply a translation
            d.apply_translation([10, 0])
            # X should have translated by 10.0
            assert g.np.allclose(d.bounds[:, 0] - 10, ori[:, 0])
            # Y should not have moved
            assert g.np.allclose(d.bounds[:, 1], ori[:, 1])

            if len(d.polygons_full) > 0 and len(d.vertices) < 150:
                g.log.info('Checking medial axis on %s',
                           d.metadata['file_name'])
                m = d.medial_axis()
                assert len(m.entities) > 0

            # shouldn't crash
            d.fill_gaps()

            # transform to first quadrant
            d.rezero()
            # run process manually
            d.process()

    def test_poly(self):
        p = g.get_mesh('2D/LM2.dxf')
        assert p.is_closed

        # one of the lines should be a polyline
        assert any(len(e.points) > 2 for e in p.entities if
                   isinstance(e, g.trimesh.path.entities.Line))

        # layers should match entity count
        assert len(p.layers) == len(p.entities)
        assert len(set(p.layers)) > 1

        count = len(p.entities)

        p.explode()
        # explode should have created new entities
        assert len(p.entities) > count
        # explode should have added some new layers
        assert len(p.entities) == len(p.layers)
        # all line segments should have two points now
        assert all(len(i.points) == 2 for i in p.entities if
                   isinstance(i, g.trimesh.path.entities.Line))
        # should still be closed
        assert p.is_closed
        # chop off the last entity
        p.entities = p.entities[:-1]
        # should no longer be closed
        assert not p.is_closed

        # fill gaps of any distance
        p.fill_gaps(g.np.inf)
        # should have fixed this puppy
        assert p.is_closed

    def test_text(self):
        """
        Do some checks on Text entities
        """
        p = g.get_mesh('2D/LM2.dxf')
        p.explode()
        # get some text entities
        text = [e for e in p.entities if
                isinstance(e, g.trimesh.path.entities.Text)]
        assert len(text) > 1

        # loop through each of them
        for t in text:
            # a spurious error we were seeing in CI
            if g.trimesh.util.is_instance_named(t, 'Line'):
                raise ValueError(
                    'type bases:',
                    [i.__name__ for i in g.trimesh.util.type_bases(t)])
        # make sure this doesn't crash with text entities
        g.trimesh.rendering.convert_to_vertexlist(p)

    def test_empty(self):
        # make sure empty paths perform as expected

        p = g.trimesh.path.Path2D()
        assert p.is_empty
        p = g.trimesh.path.Path3D()
        assert p.is_empty

        p = g.trimesh.load_path([[[0, 0, 1], [1, 0, 1]]])
        assert len(p.entities) == 1
        assert not p.is_empty

        b = p.to_planar()[0]
        assert len(b.entities) == 1
        assert not b.is_empty

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

    def test_color(self):
        p = g.get_mesh('2D/wrench.dxf')
        # make sure we have entities
        assert len(p.entities) > 0
        # make sure shape of colors is correct
        assert p.colors.shape == (len(p.entities), 4)
        color = [255, 0, 0, 255]
        # assign a color to the entity
        p.entities[0].color = color
        # make sure this is reflected in the path color
        assert g.np.allclose(p.colors[0], color)


class SplitTest(g.unittest.TestCase):

    def test_split(self):

        for fn in ['2D/ChuteHolderPrint.DXF',
                   '2D/tray-easy1.dxf',
                   '2D/sliding-base.dxf',
                   '2D/wrench.dxf',
                   '2D/spline_1.dxf']:
            p = g.get_mesh(fn)

            # make sure something was loaded
            assert len(p.root) > 0

            # split by connected
            split = p.split()

            # make sure split parts have same area as source
            assert g.np.isclose(p.area, sum(i.area for i in split))
            # make sure concatenation doesn't break that
            assert g.np.isclose(p.area, g.np.sum(split).area)

            # check that cache didn't screw things up
            for s in split:
                assert len(s.root) == 1
                assert len(s.path_valid) == len(s.paths)
                assert len(s.paths) == len(s.discrete)
                assert s.path_valid.sum() == len(s.polygons_closed)
                g.check_path2D(s)


class SectionTest(g.unittest.TestCase):

    def test_section(self):
        mesh = g.get_mesh('tube.obj')

        # check the CCW correctness with a normal in both directions
        for sign in [1.0, -1.0]:
            # get a cross section of the tube
            section = mesh.section(plane_origin=mesh.center_mass,
                                   plane_normal=[0.0, sign, 0.0])

            # Path3D -> Path2D
            planar, T = section.to_planar()

            # tube should have one closed polygon
            assert len(planar.polygons_full) == 1
            polygon = planar.polygons_full[0]
            # closed polygon should have one interior
            assert len(polygon.interiors) == 1

            # the exterior SHOULD be counterclockwise
            assert g.trimesh.path.util.is_ccw(
                polygon.exterior.coords)
            # the interior should NOT be counterclockwise
            assert not g.trimesh.path.util.is_ccw(
                polygon.interiors[0].coords)

            # should be a valid Path2D
            g.check_path2D(planar)


class CreationTests(g.unittest.TestCase):

    def test_circle_pattern(self):
        from trimesh.path import creation
        pattern = creation.circle_pattern(pattern_radius=1.0,
                                          circle_radius=0.1,
                                          count=4)
        assert len(pattern.entities) == 4
        assert len(pattern.polygons_closed) == 4
        assert len(pattern.polygons_full) == 4

        # should be a valid Path2D
        g.check_path2D(pattern)

    def test_circle(self):
        from trimesh.path import creation
        circle = creation.circle(radius=1.0, center=(1.0, 1.0))

        # it's a discrete circle
        assert g.np.isclose(circle.area, g.np.pi, rtol=0.01)
        # should be centered at 0
        assert g.np.allclose(
            circle.polygons_full[0].centroid, [
                1.0, 1.0], atol=1e-3)

        assert len(circle.entities) == 1
        assert len(circle.polygons_closed) == 1
        assert len(circle.polygons_full) == 1

        # should be a valid Path2D
        g.check_path2D(circle)

    def test_rect(self):
        from trimesh.path import creation

        # create a single rectangle
        pattern = creation.rectangle([[0, 0], [2, 3]])
        assert len(pattern.entities) == 1
        assert len(pattern.polygons_closed) == 1
        assert len(pattern.polygons_full) == 1
        assert g.np.isclose(pattern.area, 6.0)
        # should be a valid Path2D
        g.check_path2D(pattern)

        # make 10 untouching rectangles
        pattern = creation.rectangle(
            g.np.arange(40).reshape((-1, 2, 2)))
        assert len(pattern.entities) == 10
        assert len(pattern.polygons_closed) == 10
        assert len(pattern.polygons_full) == 10
        # should be a valid Path2D
        g.check_path2D(pattern)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
