try:
    from . import generic as g
except BaseException:
    import generic as g


class AssimpTest(g.unittest.TestCase):
    def test_collada(self):
        # only test if pyassimp is installed
        try:
            import pyassimp
        except ImportError:
            g.log.error('no pyassimp to test', exc_info=True)
            return

        m = g.get_mesh('blue_cube.dae')

        assert m.visual.kind == 'vertex'

        # get the export as a string
        e = m.export(file_type='dae')

        r = g.trimesh.load(file_obj=g.trimesh.util.wrap_as_stream(e),
                           file_type='dae')

        assert r.visual.kind == 'vertex'

        # test mesh should all have same color
        assert g.np.allclose(r.visual.vertex_colors,
                             [0, 13, 96, 255])
        # original mesh should be all blueish
        assert g.np.allclose(m.visual.vertex_colors,
                             [0, 13, 96, 255])


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
