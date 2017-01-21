import generic as g


class SimplifyTest(g.unittest.TestCase):

    def polygon_simplify(self, polygon):
        path = g.trimesh.load_path(polygon)
        md5_pre = g.deepcopy(path.md5())

        simplified = path.simplify()

        # make sure the simplify call didn't alter our original mesh
        self.assertTrue(path.md5() == md5_pre)

        area_ratio = path.area / simplified.area
        arc_count = len(
            [i for i in simplified.entities if type(i).__name__ == 'Arc'])

        g.log.info('simplify area ratio was %f with %d arcs',
                   area_ratio, arc_count)

        self.assertTrue(abs(area_ratio - 1) < 1e-2)

    def test_simplify(self):
        for file_name in ['2D/cycloidal.dxf',
                          '2D/125_cycloidal.DXF',
                          '2D/spline_1.dxf']:
            for polygon in g.get_mesh(file_name).polygons_full:
                self.polygon_simplify(polygon)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
