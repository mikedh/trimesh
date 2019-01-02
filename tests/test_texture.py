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
        colors = g.trimesh.visual.uv_to_color(
            uv, PIL.Image.fromarray(texture))

        colors_expected = [[75, 76, 77, 255], [51, 52, 53, 255]]

        assert (colors == colors_expected).all()
        

    def test_fuze(self):

        def check_fuze(fuze):
            """
            Check the classic textured mesh: a fuze bottle
            """
            # these loaded fuze bottles should have textures
            assert isinstance(
                fuze.visual, g.trimesh.visual.TextureVisuals)

            # image should be loaded with correct resolution
            assert fuze.visual.material.image.size == (1024, 1024)

            # UV coordinates should be unmerged correctly
            assert len(fuze.visual.uv) == 664

            # vertices should correspond with UV
            assert fuze.vertices.shape == (664, 3)
            
            # convert TextureVisuals to ColorVisuals
            viz = fuze.visual.to_color()
            assert viz.kind == 'vertex'
            # should be actual colors defined
            assert viz.vertex_colors.ptp(axis=0).ptp() != 0

        with g.serve_meshes() as address:
            # see if web resolvers work
            tex = g.trimesh.exchange.load.load_remote(
                url=address + '/fuze.obj', process=False)
            check_fuze(tex)

            # see if web+zip resolvers work
            scene = g.trimesh.exchange.load.load_remote(
                url=address + '/fuze.zip', process=False)

            # zip files get loaded into a scene
            assert len(scene.geometry) == 1
            # scene should just be a fuze bottle
            check_fuze(next(iter(scene.geometry.values())))

        # obj with texture, assets should be loaded
        # through a FilePathResolver
        m = g.get_mesh('fuze.obj', process=False)
        check_fuze(tex)

        # obj with texture, assets should be loaded
        # through a ZipResolver into a scene
        scene = g.get_mesh('fuze.zip', process=False)

        # zip files get loaded into a scene
        assert len(scene.geometry) == 1
        m = next(iter(scene.geometry.values()))

        check_fuze(m)

        # the PLY should have textures defined
        m = g.get_mesh('fuze.ply', process=False)
        check_fuze(m)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
