try:
    from . import generic as g
except BaseException:
    import generic as g


class URDFTest(g.unittest.TestCase):
    """
    Test to make sure we can convert trimesh objects
    to pyglet objects for rendering.
    """

    def test_export(self):
        # not imported to trimesh by __init__
        # so we don't have a pyglet import attempt
        # unless we explicitly ask for it (like during
        # a viewer show() call)
        from trimesh.exchange import urdf

        mesh = g.get_mesh("featuretype.STL")

        out_dir = g.tempfile.mkdtemp()

        urdf.export_urdf(mesh=mesh, directory=out_dir)

        # make sure it wrote something
        files = g.os.listdir(out_dir)
        assert len(files) > 0

        # remove temporary directory
        g.shutil.rmtree(out_dir)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
