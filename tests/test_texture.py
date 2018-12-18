try:
    from . import generic as g
except BaseException:
    import generic as g


class TextureTest(g.unittest.TestCase):

    def test_uv_to_color(self):
        try:
            import PIL.Image
        except ImportError:
            return

        n_vertices = 100
        uv = g.np.array([[0.25, 0.2], [0.4, 0.5]], dtype=float)
        texture = g.np.arange(96, dtype=g.np.uint8).reshape(8, 4, 3)
        colors = g.trimesh.visual.uv_to_color(uv, PIL.Image.fromarray(texture))

        colors_expected = [[75, 76, 77, 255], [51, 52, 53, 255]]

        g.np.testing.assert_allclose(colors, colors_expected, rtol=0, atol=0)

    def test_fuze(self):

        with g.serve_meshes() as address:
            # see if web resolvers work
            tex = g.trimesh.io.load.load_remote(
                url=address + '/fuze.obj')

            # this test should change when texture is actually implemented
            assert tex.visual.kind == 'vertex'
            # should be actual colors defined
            assert tex.visual.vertex_colors.ptp(axis=0).ptp() != 0

        # obj with texture, assets should be loaded
        # through a FilePathResolver
        m = g.get_mesh('fuze.obj')

        # this test should change when texture is actually implemented
        assert m.visual.kind == 'vertex'
        # should be actual colors defined
        assert m.visual.vertex_colors.ptp(axis=0).ptp() != 0

        # obj with texture, assets should be loaded
        # through a ZipResolver into a scene
        scene = g.get_mesh('fuze.zip')

        # zip files get loaded into a scene
        assert len(scene.geometry) == 1
        m = next(iter(scene.geometry.values()))

        # this test should change when texture is actually implemented
        assert m.visual.kind == 'vertex'
        # should be actual colors defined
        assert m.visual.vertex_colors.ptp(axis=0).ptp() != 0


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
