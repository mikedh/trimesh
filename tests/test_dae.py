try:
    from . import generic as g
except BaseException:
    import generic as g


class DAETest(g.unittest.TestCase):
    def test_duck(self):
        """
        Load a collada scene with pycollada.
        """

        scene = g.get_mesh('duck.dae')

        assert len(scene.geometry) == 1
        assert len(scene.graph.nodes_geometry) == 1


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
