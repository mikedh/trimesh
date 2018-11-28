try:
    from . import generic as g
except BaseException:
    import generic as g


class ExportTest(g.unittest.TestCase):

    def test_svg(self):
        for d in g.get_2D():
            # export as svg string
            exported = d.export(file_type='svg')
            # load the exported SVG
            stream = g.trimesh.util.wrap_as_stream(exported)

            if g.np.isclose(d.area, 0.0):
                continue

            loaded = g.trimesh.load(stream, file_type='svg')

            # we only have line and arc primitives as SVG
            # export and import
            if all(i.__class__.__name__ in ['Line', 'Arc']
                   for i in d.entities):
                # perimeter should stay the same-ish
                # on export/inport
                assert g.np.isclose(d.length,
                                    loaded.length,
                                    rtol=.01)

    def test_layer(self):
        from shapely.geometry import Point

        # create two disjoint circles and apply layers
        a = g.trimesh.load_path(Point([0, 0]).buffer(1))
        a.apply_layer('ACIRCLE')

        b = g.trimesh.load_path(Point([2, 0]).buffer(1))
        b.apply_layer('BCIRCLE')

        # combine two circles
        c = a + b

        # should be combined
        assert g.np.isclose(c.area, a.area + b.area)

        # export C with just layer of A
        aX = c.export(file_type='svg',
                      layers=['ACIRCLE'],
                      return_path=True)

        # export C with just layer of A
        cX = c.export(file_type='svg',
                      layers=None,
                      return_path=True)

        assert len(cX) > len(aX)

        # make
        aR = g.trimesh.load(g.io_wrap(c.export(file_type='dxf',
                                               layers=['ACIRCLE'])),
                            file_type='dxf')

        assert g.np.isclose(aR.area, a.area)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
