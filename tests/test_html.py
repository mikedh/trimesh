try:
    from . import generic as g
except BaseException:
    import generic as g


class ViewTest(g.unittest.TestCase):
    def test_JSHTML(self):
        import trimesh.viewer.notebook as js

        m = g.get_mesh("featuretype.STL")
        s = m.scene()

        # a viewable scene
        html = js.scene_to_html(s)
        assert len(html) > 0

        # check it out as an LXML document
        from lxml.html import fromstring

        h = fromstring(html)

        # should have some children
        children = list(h.body.iterchildren())
        assert len(children) >= 2

    def test_inNB(self):
        import trimesh.viewer.notebook as js

        # should only return True inside
        # jypyter/IPython notebooks
        assert not js.in_notebook()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
