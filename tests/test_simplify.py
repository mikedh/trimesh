import generic as g


class SimplifyTest(g.unittest.TestCase):

    def polygon_simplify(self, polygon, arc_count):

        # loading the polygon will make all arcs discrete
        path = g.trimesh.load_path(polygon)

        # save the md5 before doing operations
        md5_pre = g.deepcopy(path.md5())

        # this should return a copy of the path
        simplified = path.simplify()

        # make sure the simplify call didn't alter our original mesh
        assert path.md5() == md5_pre

        for garbage in range(2):
            # the simplified version shouldn't have lost area
            assert g.np.allclose(path.area,
                                 simplified.area,
                                 rtol=1e-3)

            # see if we fit as many arcs as existed in the original drawing
            new_count = sum(int(type(i).__name__ == 'Arc')
                            for i in simplified.entities)
            if new_count != arc_count:
                print(new_count, arc_count)

            if arc_count > 1:
                g.log.info('originally were {} arcs, simplify found {}'.format(
                    arc_count,
                    new_count))
                assert new_count > 0
                assert new_count <= arc_count

            # dump the cache to make sure bookkeeping wasn't busted or wrong
            simplified._cache.clear()

        # make sure the simplify call didn't alter our original mesh
        assert path.md5() == md5_pre

    def test_simplify(self):
        for file_name in ['2D/cycloidal.dxf',
                          '2D/125_cycloidal.DXF',
                          '2D/spline_1.dxf']:

            original = g.get_mesh(file_name)

            split = original.split()

            assert g.np.allclose(original.area,
                                 sum(i.area for i in split))

            for drawing in split:
                # we split so there should be only one polygon per drawing now
                assert len(drawing.polygons_full) == 1
                polygon = drawing.polygons_full[0]
                arc_count = sum(int(type(i).__name__ == 'Arc')
                                for i in drawing.entities)

                self.polygon_simplify(polygon=polygon,
                                      arc_count=arc_count)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
