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

    def test_methods(self):
        if not g.include_rendering:
            return
        # set a GL config that fixes a depth buffer issue in xvfb
        # this should raise an exception if pyglet can't get a library
        from pyglet import gl
        from pyglet.window import key

        from trimesh.viewer.windowed import SceneViewer

        window_conf = gl.Config(double_buffer=True, depth_size=24)

        # try the various toggles we have implemented
        key_check = [key.W, key.Z, key.C, key.A, key.G, key.M, key.F]

        scene = g.get_mesh("cycloidal.3DXML")
        scene.add_geometry(g.get_mesh("fuze.obj"))
        scene.add_geometry(g.get_mesh("2D/spline.DXF"))

        v = SceneViewer(scene=scene, start_loop=False, window_conf=window_conf)
        v.on_draw()

        # see if these do anything
        for k in key_check:
            v.on_key_press(symbol=k, modifiers=None)
            v.on_draw()

        # run the quit key toggle
        v.on_key_press(symbol=key.Q, modifiers=None)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
