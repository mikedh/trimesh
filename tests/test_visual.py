try:
    from . import generic as g
except BaseException:
    import generic as g


class VisualTest(g.unittest.TestCase):
    def test_face_subset_texture_visuals(self):
        m = g.get_mesh("fuze.obj", force="mesh")

        random = g.np.random.default_rng(seed=0)
        face_index = random.choice(len(m.faces), len(m.triangles) // 2)
        idx = m.faces[g.np.unique(face_index)].flatten()

        ori = m.visual.uv[idx]
        check = m.visual.face_subset(face_index).uv

        tree = g.spatial.cKDTree(ori)
        distances, _index = tree.query(check, k=1)
        assert distances.max() < 1e-8

    def test_split_shares_texture(self):
        # every piece of a split must reference the source image:
        # a copy per piece runs out of memory on large textures
        from PIL import Image

        from trimesh.visual.material import PBRMaterial, SimpleMaterial

        box = g.trimesh.creation.box()
        m = g.trimesh.util.concatenate([box, box.copy().apply_translation([5, 0, 0])])
        uv = g.np.random.default_rng(0).random((len(m.vertices), 2))
        image = Image.new("RGBA", (64, 64))
        for material, key in [
            (SimpleMaterial(image=image), "image"),
            (PBRMaterial(baseColorTexture=image), "baseColorTexture"),
        ]:
            m.visual = g.trimesh.visual.TextureVisuals(uv=uv, material=material)
            parts = m.split(only_watertight=False)
            assert len(parts) == 2
            for p in parts:
                assert getattr(p.visual.material, key) is image
                # each vertex of a piece keeps the UV of its source vertex
                _, index = g.spatial.cKDTree(m.vertices).query(p.vertices)
                assert g.np.allclose(p.visual.uv, uv[index])

    def test_face_subset_color_visuals(self):
        import trimesh

        m = g.get_mesh("torus.STL")

        random = g.np.random.default_rng(seed=0)
        vertex_colors = random.integers(0, 255, size=(len(m.vertices), 3))
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=vertex_colors)

        face_index = random.choice(len(m.faces), len(m.triangles) // 2)
        idx = m.faces[g.np.unique(face_index)].flatten()

        ori = m.visual.vertex_colors[idx]
        check = m.visual.face_subset(face_index).vertex_colors

        tree = g.spatial.cKDTree(ori)
        distances, _index = tree.query(check, k=1)
        assert distances.max() < 1e-8

    # def test_face_subset_vertex_color(self):
    #     import trimesh
    #     m = g.get_mesh('torus.STL')
    #
    #     vertex_colors = trimesh.visual.VertexColor(mesh=m, vertex_color=)
    #     m.visual = trimesh.visual.VertexColor(mesh=m, colors=vertex_colors)
    #
    #     face_index = g.np.random.choice(len(m.faces), len(m.triangles) // 2)
    #     idx = m.faces[g.np.unique(face_index)].flatten()
    #
    #     ori = m.visual.vertex_colors[idx]
    #     check = m.visual.face_subset(face_index).vertex_colors
    #
    #     tree = g.spatial.cKDTree(ori)
    #     distances, index = tree.query(check, k=1)
    #     assert distances.max() < 1e-8

    def test_face_maintain_order(self):
        # chose a mesh that has the same number of vs and vts
        # to prevent conflict without unmerging when maintain_order=True
        mesh1 = g.get_mesh("capsule.obj", process=False, maintain_order=True)
        mesh2 = g.get_mesh("capsule.obj", process=False, maintain_order=False)
        colors1 = mesh1.visual.to_color()
        colors2 = mesh2.visual.to_color()
        g.np.testing.assert_allclose(colors1.vertex_colors, colors2.vertex_colors)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
