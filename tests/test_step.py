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

    def test_merge_primitives(self):
        try:
            import cascadio  # noqa
        except BaseException:
            g.log.error("failed to get cascadio!", exc_info=True)
            return

        # a cube with faces in two colors is exported by cascadio
        # as one glTF mesh with two primitives that get merged
        for skip_materials in [False, True]:
            m = g.trimesh.load_mesh(
                g.os.path.join(g.dir_models, "two_color_cube.STEP"),
                skip_materials=skip_materials,
            )
            assert len(m.faces) == 12
            m.merge_vertices()
            assert m.is_watertight

        # two polylines in two colors are exported as two LINES
        # primitives, which have no faces to merge
        for skip_materials in [False, True]:
            s = g.trimesh.load_scene(
                g.os.path.join(g.dir_models, "two_color_lines.STEP"),
                skip_materials=skip_materials,
            )
            assert len(s.geometry) == 2
            assert all(isinstance(m, g.trimesh.path.Path3D) for m in s.geometry.values())


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
