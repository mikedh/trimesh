try:
    from . import generic as g
except BaseException:
    import generic as g


class SmoothTest(g.unittest.TestCase):
    def test_smooth(self):
        m = g.get_mesh("chair_model.obj", force="mesh")
        s = m.smooth_shaded

        ori = g.np.hstack((m.visual.uv, m.vertices))
        check = g.np.hstack((s.visual.uv, s.vertices))

        tree = g.spatial.cKDTree(ori)
        distances, index = tree.query(check, k=1)
        assert distances.max() < 1e-8

        # g.texture_equal(m, s)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
