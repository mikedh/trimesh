try:
    from . import generic as g
except BaseException:
    import generic as g


class FillTest(g.unittest.TestCase):
    def test_fill(self):
        # a path closed with a bowtie, so the topology is wrong
        a = g.get_mesh("2D/broken_loop.dxf")
        assert len(a.paths) == 0
        # bowtie shouldn't require any connection distance
        a.fill_gaps(0.0)
        # topology should be good now
        assert len(a.paths) == 1
        # it is a rectangle
        assert g.np.isclose(a.area, g.np.prod(a.extents))

        # a path with a bowtie and a .05 gap
        b = g.get_mesh("2D/broken_pair.dxf")
        assert len(b.paths) == 0
        # should be too small to fill gap
        b.fill_gaps(0.01)
        assert len(b.paths) == 0
        # should be large enough to fill gap
        b.fill_gaps(0.06)
        assert len(b.paths) == 1
        # it is a rectangle
        assert g.np.isclose(b.area, g.np.prod(b.extents))


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
