try:
    from . import generic as g
except BaseException:
    import generic as g


class SmoothTest(g.unittest.TestCase):
    def test_smooth(self):
        """
        Load a collada scene with pycollada.
        """
        m = trimesh.creation.icosahedron()
        m.vertices,m.faces=trimesh.remesh.subdivide_to_size(m.vertices,m.faces,0.1)

        s=m.copy()
        f=m.copy()
        d=m.copy()

        assert m.is_volume

        # Equal Weights
        lap=trimesh.smoothing.laplacian_calculation(m,1)
        trimesh.smoothing.filter_laplacian(s,0.5,10,lap)
        trimesh.smoothing.filter_humphrey(f,0.1,0.5,10,lap)
        trimesh.smoothing.filter_taubin(d,0.5,0.53,10,lap)

        assert s.is_volume
        assert f.is_volume
        assert d.is_volume

        assert np.isclose(s.volume, m.volume,rtol=0.1)
        assert np.isclose(f.volume, m.volume,rtol=0.1)
        assert np.isclose(d.volume, m.volume,rtol=0.1)

        s=m.copy()
        f=m.copy()
        d=m.copy()

        # UMbrella Weights
        lap=trimesh.smoothing.laplacian_calculation(m,2)
        trimesh.smoothing.filter_laplacian(s,0.5,10,lap)
        trimesh.smoothing.filter_humphrey(f,0.1,0.5,10,lap)
        trimesh.smoothing.filter_taubin(d,0.5,0.53,10,lap)

        assert s.is_volume
        assert f.is_volume
        assert d.is_volume

        assert np.isclose(s.volume, m.volume,rtol=0.1)
        assert np.isclose(f.volume, m.volume,rtol=0.1)
        assert np.isclose(d.volume, m.volume,rtol=0.1)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()