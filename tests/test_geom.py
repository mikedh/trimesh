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

    def test_triangulate_winding(self):
        """Ensure the fan path winds triangles the same as the quad path."""
        from trimesh.geometry import triangulate_quads
        from trimesh.triangles import area, normals

        # unit square on XY plane
        vertices = g.np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        )

        # quad path: produces 2 triangles
        quad_faces = triangulate_quads([[0, 1, 2, 3]])
        assert len(quad_faces) == 2

        # 5-gon: same square, edge 1->2 split by midpoint vertex 4
        vertices_5 = g.np.vstack([vertices, [[1, 0.5, 0]]])
        fan_faces = triangulate_quads([[0, 1, 4, 2, 3]])
        assert len(fan_faces) == 3

        # all normals should point +Z (consistent winding)
        quad_normals, quad_valid = normals(vertices[quad_faces])
        assert quad_valid.all()
        assert (quad_normals[:, 2] > 0).all()

        fan_normals, fan_valid = normals(vertices_5[fan_faces])
        assert fan_valid.all()
        assert (fan_normals[:, 2] > 0).all()

        # both should cover the unit square
        assert g.np.isclose(area(vertices[quad_faces]).sum(), 1.0)
        assert g.np.isclose(area(vertices_5[fan_faces]).sum(), 1.0)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
