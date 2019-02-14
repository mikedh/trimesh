try:
    from . import generic as g
except BaseException:
    import generic as g


class DAETest(g.unittest.TestCase):
    def test_wam(self):
        """
        Load a collada scene with pycollada.
        """

        scene = g.get_mesh('barrett-wam.zae')

        assert len(scene.geometry) == 8
        assert len(scene.graph.nodes_geometry) == 8

        assert g.np.allclose(g.np.sort(scene.extents),
                             [0.3476, 0.36, 1.26096],
                             atol=.001)

        

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
