try:
    from . import generic as g
except BaseException:
    import generic as g


class SmoothTest(g.unittest.TestCase):
    def test_smooth(self):
        """
        Load a collada scene with pycollada.
        """
        m = g.trimesh.creation.icosahedron()
        m.vertices, m.faces = g.trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.1)

        s = m.copy()
        q = m.copy()
        f = m.copy()
        d = m.copy()
        b = m.copy()
        v = m.copy()

        assert m.is_volume

        # Equal Weights
        lap = g.trimesh.smoothing.laplacian_calculation(mesh=m, equal_weight=True)

        g.trimesh.smoothing.filter_laplacian(s, 0.5, 10, False, True, lap)
        g.trimesh.smoothing.filter_laplacian(q, 0.5, 10, True, True, lap)
        g.trimesh.smoothing.filter_humphrey(f, 0.1, 0.5, 10, lap)
        g.trimesh.smoothing.filter_taubin(d, 0.5, 0.53, 10, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(b, 0.5, 10, False, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(v, 0.5, 10, True, lap)

        assert s.is_volume
        assert q.is_volume
        assert f.is_volume
        assert d.is_volume
        assert b.is_volume
        assert v.is_volume

        assert g.np.isclose(s.volume, m.volume, rtol=0.1)
        assert g.np.isclose(q.volume, m.volume, rtol=0.1)
        assert g.np.isclose(f.volume, m.volume, rtol=0.1)
        assert g.np.isclose(d.volume, m.volume, rtol=0.1)
        assert g.np.isclose(b.volume, m.volume, rtol=0.1)
        assert g.np.isclose(v.volume, m.volume, rtol=0.1)

        s = m.copy()
        q = m.copy()
        f = m.copy()
        d = m.copy()
        b = m.copy()
        v = m.copy()

        # umbrella Weights
        lap = g.trimesh.smoothing.laplacian_calculation(m, equal_weight=False)

        g.trimesh.smoothing.filter_laplacian(s, 0.5, 10, False, True, lap)
        g.trimesh.smoothing.filter_laplacian(q, 0.5, 10, True, True, lap)
        g.trimesh.smoothing.filter_humphrey(f, 0.1, 0.5, 10, lap)
        g.trimesh.smoothing.filter_taubin(d, 0.5, 0.53, 10, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(b, 0.5, 10, False, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(v, 0.5, 10, True, lap)

        assert s.is_volume
        assert q.is_volume
        assert f.is_volume
        assert d.is_volume
        assert b.is_volume
        assert v.is_volume

        assert g.np.isclose(s.volume, m.volume, rtol=0.1)
        assert g.np.isclose(q.volume, m.volume, rtol=0.1)
        assert g.np.isclose(f.volume, m.volume, rtol=0.1)
        assert g.np.isclose(d.volume, m.volume, rtol=0.1)
        assert g.np.isclose(b.volume, m.volume, rtol=0.1)
        assert g.np.isclose(v.volume, m.volume, rtol=0.1)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
