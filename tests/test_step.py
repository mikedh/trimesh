try:
    import generic as g
except BaseException:
    from . import generic as g


class STEPTests(g.unittest.TestCase):
    def test_basic(self):
        try:
            import cascadio  # noqa
        except BaseException:
            g.log.error("failed to get cascadio!", exc_info=True)
            return

        s = g.get_mesh("box_sides.STEP")
        assert len(s.geometry) == 10

        b = g.get_mesh("featuretype.STEP")
        assert len(b.geometry) == 1
        m = next(iter(b.geometry.values()))
        m.merge_vertices(merge_tex=True, merge_norm=True)
        assert m.is_watertight


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
