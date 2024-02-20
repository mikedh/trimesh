try:
    from . import generic as g
except BaseException:
    import generic as g


class RasterTest(g.unittest.TestCase):
    def test_rasterize(self):
        p = g.get_mesh("2D/wrench.dxf")

        origin = p.bounds[0]
        pitch = p.extents.max() / 600
        resolution = g.np.ceil(p.extents / pitch).astype(int)

        # rasterize with filled
        filled = p.rasterize(
            origin=origin, pitch=pitch, resolution=resolution, fill=True, width=None
        )

        # rasterize just the outline
        outline = p.rasterize(
            origin=origin, pitch=pitch, resolution=resolution, fill=False, width=2.0
        )

        # rasterize both
        both = p.rasterize(
            origin=origin, pitch=pitch, resolution=resolution, fill=True, width=2.0
        )

        # rasterize with two-dimensional pitch
        pitch = p.extents / 600
        filled_2dpitch = p.rasterize(
            origin=origin, pitch=pitch, resolution=resolution, fill=True, width=None
        )

        # count the number of filled pixels
        fill_cnt = g.np.array(filled).sum()
        fill_2dpitch_cnt = g.np.array(filled_2dpitch).sum()
        both_cnt = g.np.array(both).sum()
        outl_cnt = g.np.array(outline).sum()

        # filled should have more than an outline
        assert fill_cnt > outl_cnt
        # filled+outline should have more than outline
        assert both_cnt > outl_cnt
        # filled+outline should have more than filled
        assert both_cnt > fill_cnt
        # A different pitch results in a different image
        assert fill_2dpitch_cnt != fill_cnt

    def test_nested(self):
        # make a test path with nested circles
        theta = g.np.linspace(0, g.np.pi * 2, 100)
        unit = g.np.column_stack((g.np.cos(theta), g.np.sin(theta)))
        radii = g.np.linspace(1.0, 10.0, 10)
        g.np.random.shuffle(radii)
        paths = []
        for R in radii:
            paths.append(g.trimesh.load_path(R * unit))
        path = g.trimesh.path.util.concatenate(paths)

        # split and extrude should both show 5 regions
        assert len(path.split()) == 5
        assert len(path.extrude(1.0)) == 5

        pitch = path.extents.max() / 1000
        origin = path.bounds[0] - pitch
        resolution = (g.np.ceil(path.extents / pitch) + 2).astype(int)

        # rasterize using the settings
        r = path.rasterize(pitch=pitch, origin=origin, resolution=resolution)
        # it's a boolean image so filled cells times
        # pitch area should be about the same as the area
        filled = g.np.array(r).sum() * pitch**2

        assert g.np.isclose(filled, path.area, rtol=0.01)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
