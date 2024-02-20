try:
    from . import generic as g
except BaseException:
    import generic as g


class ReprTest(g.unittest.TestCase):
    def test_repr(self):
        m = g.trimesh.creation.icosphere()
        r = str(m)
        assert "trimesh.Trimesh" in r
        assert "vertices" in r
        assert "faces" in r

        s = m.scene()
        assert isinstance(s, g.trimesh.Scene)
        r = str(s)
        assert "Scene" in r
        assert "geometry" in r

        p = g.trimesh.PointCloud(m.vertices)
        r = str(p)
        assert "trimesh.PointCloud" in r
        assert "vertices" in r

        p = g.trimesh.path.creation.rectangle([[0, 0], [1, 1]])
        assert isinstance(p, g.trimesh.path.Path2D)
        r = str(p)
        assert "trimesh.Path2D" in r
        assert "entities" in r
        assert "vertices" in r

        p = p.to_3D()
        assert isinstance(p, g.trimesh.path.Path3D)
        r = str(p)
        assert "trimesh.Path3D" in r
        assert "entities" in r
        assert "vertices" in r


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
