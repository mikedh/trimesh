try:
    from . import generic as g
except BaseException:
    import generic as g


class OpTest(g.unittest.TestCase):
    def test_add(self):
        # make sure different concatenation results return the same
        m = [g.trimesh.creation.box().apply_translation(v) for v in 2.0 * g.np.eye(3)]
        assert g.np.isclose(sum(m).volume, 3.0)
        assert g.np.isclose(g.np.sum(m).volume, 3.0)
        assert g.np.isclose((m[0] + m[1] + m[2]).volume, 3.0)
        assert g.np.isclose(g.trimesh.util.concatenate(m).volume, 3.0)

        p = g.get_mesh("2D/wrench.dxf")
        m = [p.copy().apply_translation(v) for v in p.extents.max() * 2.0 * g.np.eye(2)]
        m.append(p)

        area = 3.0 * p.area
        assert g.np.isclose(sum(m).area, area)
        assert g.np.isclose(g.np.sum(m).area, area)
        assert g.np.isclose((m[0] + m[1] + m[2]).area, area)
        assert g.np.isclose(g.trimesh.path.util.concatenate(m).area, area)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
