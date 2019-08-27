"""
Load all the meshes we can get our hands on and check things, stuff.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class OFFTests(g.unittest.TestCase):

    def test_comment(self):
        m = g.get_mesh('comments.off')

        assert m.is_volume


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
