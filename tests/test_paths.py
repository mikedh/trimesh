import generic as g


class VectorTests(g.unittest.TestCase):

    def test_discrete(self):
        for d in g.get_2D():

            self.assertTrue(len(d.polygons_closed) == len(d.paths))

            for path in d.paths:
                verts = d.discretize_path(path)
                dists = g.np.sum((g.np.diff(verts, axis=0))**2, axis=1)**.5
                self.assertTrue(g.np.all(dists > g.tol_path.zero))
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


class PolygonsTest(g.unittest.TestCase):

    def test_rasterize(self):
        test_radius = 1.0
        test_pitch = test_radius / 10.0
        polygon = g.Point([0, 0]).buffer(test_radius)
        offset, grid, grid_points = g.trimesh.path.polygons.rasterize_polygon(polygon=polygon,
                                                                              pitch=test_pitch)
        self.assertTrue(g.trimesh.util.is_shape(grid_points, (-1, 2)))

        grid_radius = (grid_points ** 2).sum(axis=1) ** .5
        pixel_diagonal = (test_pitch * (2.0**.5)) / 2.0
        contained = grid_radius <= (test_radius + pixel_diagonal)

        self.assertTrue(contained.all())

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
