"""
Load all the meshes we can get our hands on and check things, stuff.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class OFFTests(g.unittest.TestCase):

    def test_comment(self):
        # see if we survive comments
        m = g.get_mesh('comments.off', process=False)
        assert m.is_volume
        assert m.vertices.shape == (8, 3)
        assert m.faces.shape == (12, 3)

    def test_whitespace(self):
        m = g.get_mesh('whitespace.off', process=False)
        assert m.is_volume
        assert m.vertices.shape == (8, 3)
        assert m.faces.shape == (12, 3)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
