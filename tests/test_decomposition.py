try:
    from . import generic as g
except BaseException:
    import generic as g


class DecompositionTest(g.unittest.TestCase):
    def test_convex_decomposition(self):
        try:
            import vhacdx  # noqa
        except ImportError:
            return

        mesh = g.get_mesh("quadknot.obj")
        meshes = mesh.convex_decomposition()
        assert len(meshes) > 1
        for m in meshes:
            assert m.is_watertight
            assert m.is_convex


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
