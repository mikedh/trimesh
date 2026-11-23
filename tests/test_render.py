try:
    from . import generic as g
except BaseException:
    import generic as g

from pathlib import Path


class RenderTest(g.unittest.TestCase):
    """
    Test to make sure we can convert trimesh objects
    to pyglet objects for rendering.
    """

    def test_args(self):
        # not imported to trimesh by __init__
        # so we don't have a pyglet import attempt
        # unless we explicitly ask for it (like during
        # a viewer show() call)
        from trimesh import rendering

        files = ["featuretype.STL", "fuze.obj", "points_bin.ply"]
        for file_name in files:
            m = g.get_mesh(file_name)
            args = rendering.convert_to_vertexlist(m)

            if isinstance(m, g.trimesh.Trimesh):
                # try turning smoothing off and on
                rendering.mesh_to_vertexlist(m, smooth_threshold=0)
                rendering.mesh_to_vertexlist(m, smooth_threshold=g.np.inf)

                P30 = m.section(plane_normal=[0, 0, 1], plane_origin=m.centroid)
                args = rendering.path_to_vertexlist(P30)
                args_auto = rendering.convert_to_vertexlist(P30)
                assert len(args) == 6
                assert len(args_auto) == len(args)

                P20, _T = P30.to_planar()
                args = rendering.path_to_vertexlist(P20)
                args_auto = rendering.convert_to_vertexlist(P20)
                assert len(args) == 6
                assert len(args_auto) == len(args)

        P21 = g.get_mesh("2D/wrench.dxf")
        args = rendering.path_to_vertexlist(P21)
        args_auto = rendering.convert_to_vertexlist(P21)
        assert len(args) == 6
        assert len(args_auto) == len(args)

        P22 = g.random((100, 2))
        args = rendering.points_to_vertexlist(P22)
        args_auto = rendering.convert_to_vertexlist(P22)
        assert len(args) == 6
        assert len(args_auto) == len(args)

        P31 = g.random((100, 3))
        args = rendering.points_to_vertexlist(P31)
        args_auto = rendering.convert_to_vertexlist(P31)
        assert len(args) == 6
        assert len(args_auto) == len(args)

        P32 = g.trimesh.points.PointCloud(P31)
        args = rendering.convert_to_vertexlist(P32)
        assert len(args) == 6
        assert len(args_auto) == len(args)

    def test_render(self):
        # importing trimesh or the GPU conversion loads no renderer or pyglet
        code = "import sys, trimesh, trimesh.viewer.conversion; print(*sys.modules)"
        loaded = g.subprocess.run(
            [g.sys.executable, "-c", code], capture_output=True, check=True, text=True
        ).stdout.split()
        assert not {"trimesh.viewer.render", "trimesh.interfaces.ffmpeg", "pyglet"} & set(
            loaded
        ), "nothing rendering imported"

        g.require_rendering()
        box = g.trimesh.creation.box().apply_translation([0, 0, 1])
        line = g.trimesh.load_path([[-1, -1, 0.5], [1, 1, 0.5]])
        points = g.trimesh.PointCloud([[0.7, -0.7, 0.5], [-0.7, 0.7, 0.5]])
        scene = g.trimesh.Scene([box, line, points])
        resolution = (96, 72)
        default = scene.save_image(resolution=resolution)
        # a path holds the same still `save_image` returns
        with g.TemporaryDirectory() as directory:
            path = g.os.path.join(directory, "still.png")
            assert scene.render(Path(path), resolution=resolution) is None
            with open(path, "rb") as file_obj:
                g.assert_images(g.png(file_obj.read()), g.png(default), "path", 2)
        default = g.png(default)
        assert default.shape == (72, 96, 4), "png"
        # every drawn `RenderSettings` field changes the image
        changed = {
            "quality": False,
            "exposure": 2.0,
            "bloom": 0.0,
            "ground_shadow": False,
        }
        changed.update(show_environment=True, background=[255, 0, 0, 255])
        changed.update(line_width=6.0, point_size=12.0)
        for key, value in changed.items():
            image = g.png(scene.render(resolution=resolution, **{key: value}))
            assert g.np.abs(image.astype(int) - default).max() > 8, f"{key} drawn"
        # bloom is part of quality
        fast = g.png(scene.render(resolution=resolution, quality=False))
        without = g.png(scene.render(resolution=resolution, quality=False, bloom=0.0))
        g.assert_images(without, fast, "no bloom without quality")
        # what isn't a viewer option raises rather than being dropped
        with g.pytest.raises(TypeError, match="bogus"):
            scene.render(resolution=resolution, bogus=True)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
