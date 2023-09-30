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

        # n_vertices = 100
        uv = g.np.array([[0.25, 0.2], [0.4, 0.5]], dtype=float)
        texture = g.np.arange(96, dtype=g.np.uint8).reshape(8, 4, 3)
        colors = g.trimesh.visual.uv_to_color(uv, PIL.Image.fromarray(texture))

        colors_expected = [[75, 76, 77, 255], [51, 52, 53, 255]]

        assert (colors == colors_expected).all()

    def test_bad_uv(self):
        # get a textured OBJ
        m = g.get_mesh("fuze.obj", force="mesh")
        # add malformed UV coordinates
        m.visual.uv = m.visual.uv[:100]
        m.merge_vertices()

    def test_order_kwarg(self):
        for file_name in ["ico4.obj", "ico4uv.obj"]:
            # get the location of the model file
            file_path = g.get_path(file_name)
            with open(file_path) as f:
                # get the raw ordered vertices from the file with basic string
                # ops
                v_raw = g.np.array(
                    [line[2:].split() for line in f if line.startswith("v ")],
                    dtype=g.np.float64,
                )

            # load them with maximal correspondence captain
            a = g.trimesh.load(file_path, process=False, maintain_order=True)
            # see if we have the same vertices
            assert g.np.allclose(a.vertices, v_raw)

            # load them without process and but without maintaining order
            a = g.trimesh.load(file_path, process=False, maintain_order=False)
            # vertex shape should not be the same
            assert a.vertices.shape != v_raw.shape

    def test_fuze(self):
        # create a local web server to test remote assets
        with g.serve_meshes() as address:
            # see if web resolvers work
            tex = g.trimesh.exchange.load.load_remote(
                url=address + "/fuze.obj", process=False
            )
            g.check_fuze(tex)

            # see if web + zip resolvers work
            scene = g.trimesh.exchange.load.load_remote(
                url=address + "/fuze.zip", process=False
            )

            # zip files get loaded into a scene
            assert len(scene.geometry) == 1
            # scene should just be a fuze bottle
            g.check_fuze(next(iter(scene.geometry.values())))

        # obj with texture, assets should be loaded
        # through a FilePathResolver
        m = g.get_mesh("fuze.obj", process=False)
        g.check_fuze(tex)

        # obj with texture, assets should be loaded
        # through a ZipResolver into a scene
        scene = g.get_mesh("fuze.zip", process=False)

        # zip files get loaded into a scene
        assert len(scene.geometry) == 1
        m = next(iter(scene.geometry.values()))
        g.check_fuze(m)

        # the PLY should have textures defined
        m = g.get_mesh("fuze.ply", process=False)
        g.check_fuze(m)

        # ASCII PLY should have textures defined
        m = g.get_mesh("fuze_ascii.ply", process=False)
        g.check_fuze(m)

        # textured meshes should subdivide OK-ish
        s = m.subdivide()
        assert len(s.visual.uv) == len(s.vertices)

        # load without doing the vertex separation
        # will look like garbage but represents original
        # and skips "disconnect vertices with different UV"
        b = g.get_mesh("fuze.ply", process=False, fix_texture=False)
        assert len(b.vertices) == 502
        assert len(b.visual.uv) == 502

    def test_upsize(self):
        """
        Texture images usually want to have sizes that are powers
        of two so resize textures up to the nearest power of two.
        """
        try:
            from PIL import Image
        except BaseException:
            g.log.warning("no PIL, not testing power_resize!")
            return

        # shortcut for the function
        resize = g.trimesh.visual.texture.power_resize

        img = Image.new("RGB", (10, 20))
        assert img.size == (10, 20)
        assert resize(img).size == (16, 32)
        assert resize(img, square=True).size == (32, 32)

        # check with one value on-size
        img = Image.new("RGB", (10, 32))
        assert img.size == (10, 32)
        assert resize(img).size == (16, 32)
        assert resize(img, square=True).size == (32, 32)

        # check early exit pathOA
        img = Image.new("RGB", (32, 32))
        assert img.size == (32, 32)
        assert resize(img).size == (32, 32)
        assert resize(img, square=True).size == (32, 32)

    def test_concatenate(self):
        # test concatenation with texture
        a = g.get_mesh("fuze.obj")
        b = a.copy()
        b.apply_translation([b.extents[0] * 1.25, 0, 0])

        c = a + b
        assert len(c.vertices) > len(a.vertices)
        assert len(c.visual.uv) == len(c.vertices)
        # should have deduplicated image texture
        assert g.np.allclose(c.visual.material.image.size, a.visual.material.image.size)

    def test_concatentate_multi(self):
        colors = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [100, 100, 100, 255],
        ]
        funcs = [
            g.trimesh.creation.box,
            g.trimesh.creation.icosphere,
            g.trimesh.creation.capsule,
        ]

        fuze = g.get_mesh("fuze.obj")
        fuze.apply_scale(1.0 / fuze.extents.max())
        fuze.apply_translation([-2, 0, 0] - fuze.bounds[0])

        meshes = []
        for i, color in enumerate(colors):
            for j, f in enumerate(funcs):
                m = f()
                m.visual.face_colors = color
                # convert color visual to texture
                m.visual = m.visual.to_texture()
                m.apply_translation([i * 2.2, j * 2.2, 0.0])
                meshes.append(m)

        c = g.trimesh.util.concatenate(meshes)
        assert isinstance(c.visual, g.trimesh.visual.TextureVisuals)
        assert len(c.faces) == sum(len(i.faces) for i in meshes)
        assert g.np.prod(c.visual.material.image.size) >= 4

        # convert texture back to color
        roundtrip = c.visual.to_color()
        assert roundtrip.kind == "vertex"
        vertex_c = roundtrip.vertex_colors
        # get the unique colors
        unique = vertex_c[g.trimesh.grouping.unique_rows(vertex_c)[0]]

        # roundtripped colors should be a superset of original colors
        assert {tuple(c) for c in unique}.issuperset({tuple(c) for c in colors})

    def test_to_tex(self):
        m = g.trimesh.creation.box()
        color = [255, 0, 0, 255]
        m.visual.face_colors = color
        # convert color visual to texture
        m.visual = m.visual.to_texture()
        # convert back to color
        m.visual = m.visual.to_color()
        assert g.np.allclose(m.visual.main_color, color)

    def test_uv_none(self):
        # setting UV coordinates to None should work
        m = g.get_mesh("fuze.obj")
        m.visual.uv = None
        assert m.visual.uv is None

        # should still be None on a copy
        c = m.copy()
        assert c.visual.uv is None

    def test_pbr_export(self):
        # try loading a textured box
        m = next(iter(g.get_mesh("BoxTextured.glb").geometry.values()))
        # make sure material copy doesn't crash
        m.visual.copy()

        with g.TemporaryDirectory() as d:
            # exports by path allow files to be written
            path = g.os.path.join(d, "box.obj")
            m.export(path)
            # try reloading
            r = g.trimesh.load(path)
            # make sure material survived
            assert r.visual.material.image.size == (256, 256)

    def test_pbr_material_fusion(self):
        # a scene with 5 cubes each with a different material
        # one having only simple color values
        # one with PBR material values
        # one with PBR textures
        # one with emissive textures
        # and one with specular glossiness color values
        m = g.get_mesh("pbr_cubes_emissive_spec_gloss.zip", force="mesh")

        assert isinstance(m, g.trimesh.Trimesh)

        mat_m = m.visual.material
        assert isinstance(mat_m, g.trimesh.visual.material.PBRMaterial)
        assert mat_m.baseColorTexture is not None
        assert mat_m.metallicRoughnessTexture is not None
        assert mat_m.emissiveTexture is not None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
