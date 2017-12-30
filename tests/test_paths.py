import generic as g


class VectorTests(g.unittest.TestCase):

    def test_discrete(self):
        for d in g.get_2D():
            self.assertTrue(len(d.polygons_closed) == len(d.paths))

            # file_name should be populated, and if we have a DXF file
            # the layer field should be populated with layer names
            if d.metadata['file_name'][-3:] == 'dxf':
                self.assertTrue(len(d.metadata['layers']) == len(d.entities))

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
                self.assertTrue(circuit_test)

                is_ccw = g.trimesh.path.util.is_ccw(verts)
                if not is_ccw:
                    g.log.error('discrete %s not ccw!',
                                d.metadata['file_name'])
                # self.assertTrue(is_ccw)

            for i in range(len(d.paths)):
                self.assertTrue(d.polygons_closed[i].is_valid)
                self.assertTrue(d.polygons_closed[i].area > g.tol_path.zero)
            export_dict = d.export(file_type='dict')
            export_svg = d.export(file_type='svg')
            simple = d.simplify()
            split = d.split()
            g.log.info('Split %s into %d bodies, checking identifiers',
                       d.metadata['file_name'],
                       len(split))
            for body in split:
                body.identifier

            if len(d.root) == 1:
                d.apply_obb()

            if len(d.vertices) < 150:
                g.log.info('Checking medial axis on %s',
                           d.metadata['file_name'])
                m = d.medial_axis()

    def test_poly(self):
        p = g.get_mesh('2D/LM2.dxf')
        self.assertTrue(p.is_closed)
        self.assertTrue(any(len(i.points) > 2 for i in p.entities if
                            g.trimesh.util.is_instance_named(i, 'Line')))

        self.assertTrue(len(p.metadata['layers']) == len(p.entities))
        self.assertTrue(len(g.np.unique(p.metadata['layers'])) > 1)

        p.explode()
        self.assertTrue(all(len(i.points) == 2 for i in p.entities if
                            g.trimesh.util.is_instance_named(i, 'Line')))
        self.assertTrue(p.is_closed)
        p.entities = p.entities[:-1]
        self.assertFalse(p.is_closed)

        p.fill_gaps()
        self.assertTrue(p.is_closed)


class ArcTests(g.unittest.TestCase):

    def test_center(self):
        test_points = [[[0, 0], [1.0, 1], [2, 0]]]
        test_results = [[[1, 0], 1.0]]
        points = test_points[0]
        res_center, res_radius = test_results[0]
        center_info = g.trimesh.path.arc.arc_center(points)
        C, R, N, angle = (center_info['center'],
                          center_info['radius'],
                          center_info['normal'],
                          center_info['span'])

        self.assertTrue(abs(R - res_radius) < g.tol_path.zero)
        self.assertTrue(g.trimesh.util.euclidean(
            C, res_center) < g.tol_path.zero)

    def test_center_random(self):
        '''
        Test that arc centers work on well formed random points in 2D and 3D
        '''
        min_angle = g.np.radians(2)
        min_radius = .0001
        count = 1000

        center_3D = (g.np.random.random((count, 3)) - .5) * 50
        center_2D = center_3D[:, 0:2]
        radii = g.np.clip(g.np.random.random(count) * 100, min_angle, g.np.inf)

        angles = g.np.random.random((count, 2)) * \
            (g.np.pi - min_angle) + min_angle
        angles = g.np.column_stack((g.np.zeros(count),
                                    g.np.cumsum(angles, axis=1)))

        points_2D = g.np.column_stack((g.np.cos(angles[:, 0]), g.np.sin(angles[:, 0]),
                                       g.np.cos(angles[:, 1]), g.np.sin(
                                           angles[:, 1]),
                                       g.np.cos(angles[:, 2]), g.np.sin(angles[:, 2]))).reshape((-1, 6))
        points_2D *= radii.reshape((-1, 1))

        points_2D += g.np.tile(center_2D, (1, 3))
        points_2D = points_2D.reshape((-1, 3, 2))

        points_3D = g.np.column_stack((points_2D.reshape((-1, 2)),
                                       g.np.tile(center_3D[:, 2].reshape((-1, 1)),
                                                 (1, 3)).reshape(-1))).reshape((-1, 3, 3))

        for center, radius, three in zip(center_2D,
                                         radii,
                                         points_2D):
            info = g.trimesh.path.arc.arc_center(three)

            assert g.np.allclose(center, info['center'])
            assert g.np.allclose(radius, info['radius'])

        for center, radius, three in zip(center_3D,
                                         radii,
                                         points_3D):
            transform = g.trimesh.transformations.random_rotation_matrix()
            center = g.trimesh.transformations.transform_points([center], transform)[
                0]
            three = g.trimesh.transformations.transform_points(
                three, transform)

            info = g.trimesh.path.arc.arc_center(three)

            assert g.np.allclose(center, info['center'])
            assert g.np.allclose(radius, info['radius'])


class PolygonsTest(g.unittest.TestCase):

    def test_rasterize(self):
        test_radius = 1.0
        test_pitch = test_radius / 10.0
        polygon = g.Point([0, 0]).buffer(test_radius)
        (offset,
         grid,
         grid_points) = g.trimesh.path.polygons.rasterize_polygon(polygon=polygon,
                                                                  pitch=test_pitch)
        self.assertTrue(g.trimesh.util.is_shape(grid_points, (-1, 2)))

        grid_radius = (grid_points ** 2).sum(axis=1) ** .5
        pixel_diagonal = (test_pitch * (2.0**.5)) / 2.0
        contained = grid_radius <= (test_radius + pixel_diagonal)

        self.assertTrue(contained.all())


class SplitTest(g.unittest.TestCase):
    def test_split(self):
        for fn in ['2D/ChuteHolderPrint.DXF',
                   '2D/tray-easy1.dxf',
                   '2D/sliding-base.dxf',
                   '2D/wrench.dxf',
                   '2D/spline_1.dxf']:
            p = g.get_mesh(fn)

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


class ExportTest(g.unittest.TestCase):
    def test_svg(self):
        for d in g.get_2D():
            # export as svg string
            exported = d.export('svg')
            # load the exported SVG
            stream = g.trimesh.util.wrap_as_stream(exported)
            loaded = g.trimesh.load(stream, file_type='svg')

            # we only have line and arc primitives as SVG export and import
            if all(i.__class__.__name__ in ['Line',
                                            'Arc'] for i in d.entities):
                # perimeter should stay the same-ish on export/inport
                assert g.np.allclose(d.length,
                                     loaded.length,
                                     rtol=.01)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
