try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import collada
except ImportError:
    collada = None


class DAETest(g.unittest.TestCase):
    def test_duck(self):
        """
        Load a collada scene with pycollada.
        """
        if collada is None:
            g.log.error('no pycollada to test!')
            return

        scene = g.get_mesh('duck.dae')

        assert len(scene.geometry) == 1
        assert len(scene.graph.nodes_geometry) == 1

    def test_shoulder(self):
        if collada is None:
            g.log.error('no pycollada to test!')
            return

        scene = g.get_mesh('shoulder.zae')
        assert len(scene.geometry) == 3
        assert len(scene.graph.nodes_geometry) == 3


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
