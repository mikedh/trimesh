try:
    from . import generic as g
except BaseException:
    import generic as g


class ViewerTest(g.unittest.TestCase):
    def test_viewer(self):
        # if the runner has not asked to include rendering exit
        if not g.include_rendering:
            return

        from pyglet import gl

        # set a GL config that fixes a depth buffer issue in xvfb
        # this should raise an exception if pyglet can't get a library
        window_conf = gl.Config(double_buffer=True, depth_size=24)

        # make a sphere
        mesh = g.trimesh.creation.icosphere()

        # get a scene object containing the mesh, this is equivalent to:
        # scene = trimesh.scene.Scene(mesh)
        scene = mesh.scene()

        # run the actual render call
        png = scene.save_image(resolution=[1920, 1080], window_conf=window_conf)

        assert len(png) > 0


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
