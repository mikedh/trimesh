try:
    from . import generic as g
except BaseException:
    import generic as g


class MFTest(g.unittest.TestCase):

    def test_3MF(self):
        # an assembly with instancing
        s = g.get_mesh('counterXP.3MF')
        # should be 2 unique meshes
        assert len(s.geometry) == 2
        # should be 6 instances around the scene
        assert len(s.graph.nodes_geometry) == 6
        assert all(m.is_volume for m in s.geometry.values())

        # a single body 3MF assembly
        s = g.get_mesh('featuretype.3MF')
        # should be 2 unique meshes
        assert len(s.geometry) == 1
        # should be 6 instances around the scene
        assert len(s.graph.nodes_geometry) == 1


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
