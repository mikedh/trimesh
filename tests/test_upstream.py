"""
test_upstream.py
-------------------

Make sure our upstream dependencies are acting right.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g


class UpstreamTests(g.unittest.TestCase):
    def test_shapely(self):
        """
        conda installs of shapely started returning NaN on
        valid input so make sure our builds fail in that case
        """
        string = g.LineString([[0, 0], [1, 0]])
        assert g.np.isclose(string.length, 1.0)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
