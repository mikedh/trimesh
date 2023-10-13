try:
    from . import generic as g
except BaseException:
    import generic as g


class SplineTests(g.unittest.TestCase):
    def test_bezier_example(self):
        # path with a bunch of bezier spline
        p = g.get_mesh("2D/MIL.svg")
        # should have one body
        assert len(p.polygons_full) == 1

        # length of perimeter of exterior
        truth = 12696.6

        # perimeter should be about right if it was discretized properly
        if not g.np.isclose(p.polygons_full[0].exterior.length, truth, atol=100.0):
            raise ValueError(
                f"perimeter wrong: {truth} != {p.polygons_full[0].exterior.length}"
            )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
