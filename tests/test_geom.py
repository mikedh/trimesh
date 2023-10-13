"""
Load all the meshes we can get our hands on and check things, stuff.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class GeomTests(g.unittest.TestCase):
    def test_triangulate(self):
        from trimesh.geometry import triangulate_quads as tq

        # create some triangles and quads
        tri = (g.random((100, 3)) * 100).astype(g.np.int64)
        quad = (g.random((100, 4)) * 100).astype(g.np.int64)

        # should just exit early for triangles
        assert g.np.allclose(tri, tq(tri.tolist()))

        # should produce two triangles for each quad
        assert len(tq(quad)) == 2 * len(quad)

        # create a mixed list of triangles and quads
        mixed = tri.tolist()
        mixed.extend(quad.tolist())
        result = tq(mixed)

        # make sure the result has the right number of elements
        assert result.shape == (len(tri) + len(quad) * 2, 3)

        # called on empty arrays should be empty
        assert len(tq([])) == 0


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
