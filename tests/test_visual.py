try:
    from . import generic as g
except BaseException:
    import generic as g


class VisualTest(g.unittest.TestCase):
    def test_face_subset_texture_visuals(self):
        m = g.get_mesh("fuze.obj", force="mesh")

        face_index = g.np.random.choice(len(m.faces), len(m.triangles) // 2)
        idx = m.faces[g.np.unique(face_index)].flatten()

        ori = m.visual.uv[idx]
        check = m.visual.face_subset(face_index).uv

        tree = g.spatial.cKDTree(ori)
        distances, index = tree.query(check, k=1)
        assert distances.max() < 1e-8

    def test_face_subset_color_visuals(self):
        import trimesh

        m = g.get_mesh("torus.STL")

        vertex_colors = g.np.random.randint(0, 255, size=(len(m.vertices), 3))
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=vertex_colors)

        face_index = g.np.random.choice(len(m.faces), len(m.triangles) // 2)
        idx = m.faces[g.np.unique(face_index)].flatten()

        ori = m.visual.vertex_colors[idx]
        check = m.visual.face_subset(face_index).vertex_colors

        tree = g.spatial.cKDTree(ori)
        distances, index = tree.query(check, k=1)
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
        # to prevent confict without unmerging when maintain_order=True
        mesh1 = g.get_mesh("capsule.obj", process=False, maintain_order=True)
        mesh2 = g.get_mesh("capsule.obj", process=False, maintain_order=False)
        colors1 = mesh1.visual.to_color()
        colors2 = mesh2.visual.to_color()
        g.np.testing.assert_allclose(colors1.vertex_colors, colors2.vertex_colors)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
