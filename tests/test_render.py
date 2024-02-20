try:
    from . import generic as g
except BaseException:
    import generic as g


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

                P20, T = P30.to_planar()
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


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
