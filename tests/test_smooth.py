try:
    from . import generic as g
except BaseException:
    import generic as g


class SmoothTest(g.unittest.TestCase):
    def test_crease_with_smooth_path_around_it(self):
        # The fold disappears at either end of the sheet.
        x, y = g.np.meshgrid(g.np.arange(-2, 3), g.np.arange(-6, 7), indexing="ij")
        z = 3 * g.np.maximum(x, 0) * g.np.exp(-(y**2) / 4)
        a = (g.np.arange(4)[:, None] * 13 + g.np.arange(12)).ravel()
        faces = g.np.vstack((g.np.c_[a, a + 13, a + 14], g.np.c_[a, a + 14, a + 1]))
        mesh = g.trimesh.Trimesh(
            g.np.c_[x.ravel(), y.ravel(), z.ravel()], faces, process=False
        )
        angle = g.np.radians(40)
        mesh.visual = g.trimesh.visual.TextureVisuals(uv=mesh.vertices[:, :2].copy())
        mesh.source.file_path = "fold.obj"
        smooth = g.trimesh.graph.smooth_shade(mesh, angle=angle, facet_minarea=None)
        center = g.np.all(smooth.vertices == [0, 0, 0], axis=1)
        assert center.sum() == 2
        left, right = smooth.vertex_normals[center]
        assert left @ right < g.np.cos(angle)
        assert g.np.all(smooth.vertices == [-1, 0, 0], axis=1).sum() == 1
        assert g.np.array_equal(smooth.visual.uv, smooth.vertices[:, :2])
        assert smooth.source.file_name == mesh.source.file_name
        assert g.np.array_equal(
            smooth.triangles,
            mesh.triangles[g.np.hstack(smooth.metadata["original_components"])],
        )

    def test_smooth(self):
        m = g.get_mesh("chair_model.obj", force="mesh")
        s = m.smooth_shaded

        ori = g.np.hstack((m.visual.uv, m.vertices))
        check = g.np.hstack((s.visual.uv, s.vertices))

        tree = g.spatial.cKDTree(ori)
        distances, _index = tree.query(check, k=1)
        assert distances.max() < 1e-8

        # g.texture_equal(m, s)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
