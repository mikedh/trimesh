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
                # on export/import
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
        aX = g.trimesh.load(g.io_wrap(
            c.export(file_type='svg',
                     layers=['ACIRCLE'])),
            file_type='svg')

        # export C with all layers
        cX = g.trimesh.load(g.io_wrap(
            c.export(file_type='svg',
                     layers=None)),
            file_type='svg')

        assert len(cX.entities) == len(c.entities)
        assert len(aX.entities) == 1

        # make
        aR = g.trimesh.load(g.io_wrap(c.export(file_type='dxf',
                                               layers=['ACIRCLE'])),
                            file_type='dxf')

        assert g.np.isclose(aR.area, a.area)

    def test_trans(self):
        from trimesh.path.exchange.svg_io import transform_to_matrices as tf

        # empty strings shouldn't have matrix
        assert len(tf('')) == 0

        # check translate with different whitespace
        a = tf('translate(1.1, 2.2      )')
        assert len(a) == 1
        assert g.np.allclose(a[0],
                             [[1, 0, 1.1],
                              [0, 1, 2.2],
                              [0, 0, 1]])
        a = tf(' translate(1.1    1.2   )       ')
        assert len(a) == 1
        assert g.np.allclose(a[0],
                             [[1, 0, 1.1],
                              [0, 1, 1.2],
                              [0, 0, 1]])

        a = tf(' translate(1.1    1.2   )       ' +
               'matrix (  {} {} {} {} {} {})'.format(*g.np.arange(6)))
        assert len(a) == 2
        # check the translate
        assert g.np.allclose(a[0],
                             [[1, 0, 1.1],
                              [0, 1, 1.2],
                              [0, 0, 1]])
        # check the matrix string
        assert g.np.allclose(a[1],
                             [[0, 2, 4],
                              [1, 3, 5],
                              [0, 0, 1]])


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
