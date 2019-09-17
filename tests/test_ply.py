try:
    from . import generic as g
except BaseException:
    import generic as g


class PlyTest(g.unittest.TestCase):

    def test_ply_dtype(self):
        # make sure all ply dtype strings are valid dtypes
        dtypes = g.trimesh.exchange.ply.dtypes
        for d in dtypes.values():
            # will raise if dtype string not valid
            g.np.dtype(d)

    def test_ply(self):
        m = g.get_mesh('machinist.XAML')

        assert m.visual.kind == 'face'
        assert m.visual.face_colors.ptp(axis=0).max() > 0

        export = m.export(file_type='ply')
        reconstructed = g.trimesh.load(g.trimesh.util.wrap_as_stream(export),
                                       file_type='ply')

        assert reconstructed.visual.kind == 'face'

        assert g.np.allclose(reconstructed.visual.face_colors,
                             m.visual.face_colors)

        m = g.get_mesh('reference.ply')

        assert m.visual.kind == 'vertex'
        assert m.visual.vertex_colors.ptp(axis=0).max() > 0

        export = m.export(file_type='ply')
        reconstructed = g.trimesh.load(g.trimesh.util.wrap_as_stream(export),
                                       file_type='ply')
        assert reconstructed.visual.kind == 'vertex'

        assert g.np.allclose(reconstructed.visual.vertex_colors,
                             m.visual.vertex_colors)

    def test_points(self):
        """
        Test reading point clouds from PLY files
        """
        m = g.get_mesh('points_ascii.ply')
        assert isinstance(m, g.trimesh.PointCloud)
        assert m.vertices.shape == (5, 3)

        m = g.get_mesh('points_bin.ply')
        assert m.vertices.shape == (5, 3)
        assert isinstance(m, g.trimesh.PointCloud)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
