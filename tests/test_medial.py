try:
    from . import generic as g
except BaseException:
    import generic as g


class MedialTests(g.unittest.TestCase):
    def test_medial(self):
        p = g.get_mesh("2D/wrench.dxf")
        assert p.is_closed

        assert len(p.polygons_full) == 1
        poly = p.polygons_full[0]
        medial = p.medial_axis()
        points = medial.vertices.view(g.np.ndarray)
        assert all(poly.contains(g.Point(v)) for v in points)

        # circles are a special case for medial axis
        poly = g.Point([0, 0]).buffer(1.0)
        # construct a Path2D from the polygon medial axis
        med = g.trimesh.path.Path2D(
            **g.trimesh.path.exchange.misc.edges_to_path(
                *g.trimesh.path.polygons.medial_axis(poly)
            )
        )
        # should have returned a single tiny line
        # with midpoint at origin
        assert len(med.vertices) == 2
        assert len(med.entities) == 1
        assert float(med.vertices.mean(axis=0).ptp()) < 1e-8


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
