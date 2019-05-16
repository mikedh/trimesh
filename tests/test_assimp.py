try:
    from . import generic as g
except BaseException:
    import generic as g


class AssimpTest(g.unittest.TestCase):

    def test_wam(self):
        """
        Load a collada scene with assimp and verify
        it isn't totally crazy.
        """
        # the wam arm, on hold for now
        """
        path = g.os.path.join(g.dir_models, 'barrett-wam.zae')
        with open(path, 'rb') as f:
            archive = g.trimesh.util.decompress(f, file_type='zip')

        # load the WAM arm using pyassimp
        kwargs = g.trimesh.exchange.assimp.load_pyassimp(
            archive['barrett-wam.dae'], file_type='dae')

        scene = g.trimesh.exchange.load.load_kwargs(kwargs)

        assert len(scene.geometry) == 8
        assert len(scene.graph.nodes_geometry) == 8
        assert g.np.allclose(g.np.sort(scene.extents),
                             [0.3476, 0.36, 1.26096],
                             atol=.001)
        """
        pass

    def test_duck(self):
        # load the duck using pyassimp
        try:
            import pyassimp  # NOQA
        except ImportError:
            return

        file_path = g.os.path.join(g.dir_models, 'duck.dae')
        with open(file_path, 'rb') as f:
            kwargs = g.trimesh.exchange.assimp.load_pyassimp(
                f, file_type='dae')

        scene = g.trimesh.exchange.load.load_kwargs(kwargs)

        assert len(scene.geometry) == 1
        assert len(scene.graph.nodes_geometry) == 1


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
