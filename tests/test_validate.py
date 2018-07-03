try:
    from . import generic as g
except BaseException:
    import generic as g


class ValidTest(g.unittest.TestCase):

    def test_validate(self):

        for validate in [True, False]:
            m = g.get_mesh('featuretype.STL', validate=validate)

            # reverse all face windings and normals to make volume negative
            m.invert()

            m._cache.verify()
            assert 'face_normals' in m._cache
            if validate:
                assert m.volume > 0.0
            else:
                assert m.volume < 0.0
            m._cache.verify()
            assert 'face_normals' in m._cache


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
