import generic as g


class ViewTest(g.unittest.TestCase):

    def test_JSHTML(self):

        import trimesh.scene.viewerJS as js

        m = g.get_mesh('featuretype.STL')
        s = m.scene()

        # a viewable scene
        html = js.scene_to_html(s)

        assert len(html) > 0

    def test_inNB(self):
        import trimesh.scene.viewerJS as js

        # should only return True inside
        # jypyter/IPython notebooks
        assert not js.in_notebook()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
