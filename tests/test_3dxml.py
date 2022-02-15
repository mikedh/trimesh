try:
    from . import generic as g
except BaseException:
    import generic as g


class DXMLTest(g.unittest.TestCase):

    def test_abaqus(self):
        # an assembly with instancing
        s = g.get_mesh('cube1.3dxml')

        # should be 1 unique meshes
        assert len(s.geometry) == 1


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
