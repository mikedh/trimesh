try:
    from . import generic as g
except BaseException:
    import generic as g


class ExportTest(g.unittest.TestCase):
    def test_svg(self):
        for d in g.get_2D():
            if g.np.isclose(d.area, 0.0):
                continue
            # export and reload the exported SVG
            loaded = g.trimesh.load(
                g.trimesh.util.wrap_as_stream(d.export(file_type="svg")), file_type="svg"
            )

            # we only have line and arc primitives as SVG
            # export and import
            if all(i.__class__.__name__ in ["Line", "Arc"] for i in d.entities):
                # perimeter should stay the same-ish
                # on export/import
                assert g.np.isclose(d.length, loaded.length, rtol=0.01)

                assert len(d.entities) == len(loaded.entities)

                path_str = g.trimesh.path.exchange.svg_io.export_svg(d, return_path=True)
                assert isinstance(path_str, str)
                assert len(path_str) > 0

    def test_layer(self):
        from shapely.geometry import Point

        # create two disjoint circles and apply layers
        a = g.trimesh.load_path(Point([0, 0]).buffer(1))
        a.apply_layer("ACIRCLE")

        b = g.trimesh.load_path(Point([2, 0]).buffer(1))
        b.apply_layer("BCIRCLE")

        assert id(a.entities[0]._metadata) != id(b.entities[0]._metadata)

        # combine two circles
        c = a + b

        # should be combined
        assert g.np.isclose(c.area, a.area + b.area)

        # export C with just layer of A
        aX = g.trimesh.load(
            g.io_wrap(c.export(file_type="svg", only_layers=["ACIRCLE"])), file_type="svg"
        )

        # export C with all layers
        cX = g.trimesh.load(
            g.io_wrap(c.export(file_type="svg", only_layers=None)), file_type="svg"
        )

        assert len(cX.entities) == len(c.entities)
        # should have skipped the layers
        assert len(aX.entities) == 1

        # make
        aR = g.trimesh.load(
            g.io_wrap(c.export(file_type="dxf", only_layers=["ACIRCLE"])), file_type="dxf"
        )

        assert g.np.isclose(aR.area, a.area)

    def test_trans(self):
        from trimesh.path.exchange.svg_io import transform_to_matrices as tf

        # empty strings shouldn't have matrix
        assert len(tf("")) == 0

        # check translate with different whitespace
        a = tf("translate(1.1, 2.2      )")
        assert len(a) == 1
        assert g.np.allclose(a[0], [[1, 0, 1.1], [0, 1, 2.2], [0, 0, 1]])
        a = tf(" translate(1.1    1.2   )       ")
        assert len(a) == 1
        assert g.np.allclose(a[0], [[1, 0, 1.1], [0, 1, 1.2], [0, 0, 1]])

        a = tf(
            " translate(1.1    1.2   )       "
            + "matrix (  {} {} {} {} {} {})".format(*g.np.arange(6))
        )
        assert len(a) == 2
        # check the translate
        assert g.np.allclose(a[0], [[1, 0, 1.1], [0, 1, 1.2], [0, 0, 1]])
        # check the matrix string
        assert g.np.allclose(a[1], [[0, 2, 4], [1, 3, 5], [0, 0, 1]])

    def test_roundtrip(self):
        """
        Check to make sure a roundtrip from both a Scene and a
        Path2D results in the same file on both sides
        """
        for fn in ["2D/250_cycloidal.DXF", "2D/tray-easy1.dxf"]:
            p = g.get_mesh(fn)
            assert isinstance(p, g.trimesh.path.Path2D)
            # load the exported SVG
            r = g.trimesh.load(
                g.trimesh.util.wrap_as_stream(p.export(file_type="svg")), file_type="svg"
            )
            assert isinstance(r, g.trimesh.path.Path2D)
            assert g.np.isclose(r.length, p.length)
            assert g.np.isclose(r.area, p.area)

            assert set(r.metadata.keys()) == set(p.metadata.keys())

            s = g.trimesh.scene.split_scene(p)
            as_svg = s.export(file_type="svg")
            assert isinstance(s, g.trimesh.Scene)
            r = g.trimesh.load(g.trimesh.util.wrap_as_stream(as_svg), file_type="svg")
            assert isinstance(r, g.trimesh.Scene)
            assert s.metadata == r.metadata

            # make sure every geometry name matches exactly
            assert set(s.geometry.keys()) == set(r.geometry.keys())

            # check to see if every geometry has the same metadata
            for geom in s.geometry.keys():
                a, b = s.geometry[geom], r.geometry[geom]
                assert a.metadata == b.metadata
                assert g.np.isclose(a.length, b.length)
                assert g.np.isclose(a.area, b.area)
                assert a.body_count == b.body_count

            assert r.metadata["file_path"].endswith(fn[3:])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
