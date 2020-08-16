try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import collada
except ImportError:
    collada = None
except BaseException:
    # TODO : REMOVE WHEN UPSTREAM RELEASE FIXED
    # https://github.com/pycollada/pycollada/pull/92
    g.log.error('DAE fix not pushed yet!')
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

    def test_export(self):
        if collada is None:
            g.log.error('no pycollada to test!')
            return

        a = g.get_mesh('ballA.off')
        r = a.export(file_type='dae')
        assert len(r) > 0

    def test_obj_roundtrip(self):
        # get a zipped-DAE scene
        s = g.get_mesh('duck.zae', force='mesh')
        with g.TemporaryDirectory() as root:
            # export using a file path so it can auto-create
            # a FilePathResolver to write the stupid assets
            path = g.os.path.join(root, 'duck.obj')
            s.export(path)
            # bring it back from outer space
            rec = g.trimesh.load(path, force='mesh')
        assert rec.visual.uv.ptp(axis=0).ptp() > 1e-5
        assert (s.visual.material.baseColorTexture.size ==
                rec.visual.material.image.size)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
