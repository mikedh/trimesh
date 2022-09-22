try:
    from . import generic as g
except BaseException:
    import generic as g


class DepTest(g.unittest.TestCase):

    def test_deprecated(self):

        tests = [g.get_mesh('2D/wrench.dxf'),
                 g.trimesh.creation.box()]

        # todo : properly hash transform trees
        # so that copies of scenes hash the same
        # g.get_mesh('cycloidal.3DXML')]

        for m in tests:
            copy = m.copy()
            # the modern cool way of hashing
            assert hash(m) == hash(copy)
            assert m.__hash__() == copy.__hash__()

            # october 2023 deprecated ways of hashing
            # geometries
            assert m.md5() == copy.md5()
            assert m.crc() == copy.crc()
            assert m.hash() == copy.hash()
            # trackedarray
            assert m.vertices.md5() == copy.vertices.md5()
            assert m.vertices.hash() == copy.vertices.hash()
            assert m.vertices.crc() == copy.vertices.crc()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
