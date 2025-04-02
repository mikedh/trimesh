try:
    from . import generic as g
except BaseException:
    import generic as g


class MargeTest(g.unittest.TestCase):
    def test_cube(self):
        """
        Test PointCloud object
        """

        m = g.trimesh.creation.box()

        assert m.vertices.shape == (8, 3)
        assert m.is_volume
        assert m.euler_number == 2

        # stack a bunch of unreferenced vertices
        m.vertices = g.np.vstack((m.vertices, g.random((10000, 3))))
        assert m.euler_number == 2
        assert m.vertices.shape == (10008, 3)
        assert m.referenced_vertices.sum() == 8
        copied = m.copy()

        # should remove unreferenced vertices
        m.merge_vertices()
        assert len(m.vertices) == 8
        assert m.is_volume
        assert m.euler_number == 2
        assert m.referenced_vertices.sum() == 8

        m.update_vertices(m.referenced_vertices)
        assert len(m.vertices) == 8
        assert m.is_volume
        assert m.euler_number == 2
        assert m.referenced_vertices.sum() == 8

        # check the copy with an int mask and see if inverse is created
        assert copied.referenced_vertices.sum() == 8
        assert copied.vertices.shape == (10008, 3)
        mask = g.np.nonzero(copied.referenced_vertices)[0]

        copied.update_vertices(mask)
        assert len(copied.vertices) == 8
        assert copied.is_volume
        assert copied.euler_number == 2
        assert copied.referenced_vertices.sum() == 8

    def test_caching(self):
        box = g.trimesh.util.concatenate([g.trimesh.creation.box(), g.trimesh.creation.box().apply_translation([1.000002, 0, 0])])

        # Check for consistent behavior
        for n_vertices, merge_norm in ((16, False), (12, True)):
            box_1 = box.copy()
            box_1.merge_vertices(digits_vertex=5, merge_norm=merge_norm)
            assert len(box_1.vertices) == n_vertices

            box_2 = box.copy()
            box_2.vertex_normals
            box_2.merge_vertices(digits_vertex=5, merge_norm=merge_norm)
            assert len(box_2.vertices) == n_vertices

            box_3 = box.copy()
            box_3.vertex_normals
            box_3._cache.delete('vertex_normals')
            box_3.merge_vertices(digits_vertex=5, merge_norm=merge_norm)
            assert len(box_3.vertices) == n_vertices


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
