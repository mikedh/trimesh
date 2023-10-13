try:
    from . import generic as g
except BaseException:
    import generic as g


class CreationTests(g.unittest.TestCase):
    def test_circle_pattern(self):
        from trimesh.path import creation

        pattern = creation.circle_pattern(pattern_radius=1.0, circle_radius=0.1, count=4)
        assert len(pattern.entities) == 4
        assert len(pattern.polygons_closed) == 4
        assert len(pattern.polygons_full) == 4

        # should be a valid Path2D
        g.check_path2D(pattern)

    def test_circle(self):
        from trimesh.path import creation

        circle = creation.circle(radius=1.0, center=(1.0, 1.0))

        # it's a discrete circle
        assert g.np.isclose(circle.area, g.np.pi, rtol=0.01)
        # should be centered at 0
        assert g.np.allclose(
            circle.polygons_full[0].centroid.coords, [1.0, 1.0], atol=1e-3
        )
        assert len(circle.entities) == 1
        assert len(circle.polygons_closed) == 1
        assert len(circle.polygons_full) == 1

        # should be a valid Path2D
        g.check_path2D(circle)

    def test_rect(self):
        from trimesh.path import creation

        # create a single rectangle
        pattern = creation.rectangle([[0, 0], [2, 3]])
        assert len(pattern.entities) == 1
        assert len(pattern.polygons_closed) == 1
        assert len(pattern.polygons_full) == 1
        assert g.np.isclose(pattern.area, 6.0)
        # should be a valid Path2D
        g.check_path2D(pattern)

        # make 10 untouching rectangles
        pattern = creation.rectangle(g.np.arange(40).reshape((-1, 2, 2)))
        assert len(pattern.entities) == 10
        assert len(pattern.polygons_closed) == 10
        assert len(pattern.polygons_full) == 10
        # should be a valid Path2D
        g.check_path2D(pattern)

    def test_grid(self):
        grid = g.trimesh.path.creation.grid(side=5.0)
        assert g.np.allclose(grid.extents, [10, 10, 0])
        # check grid along a plane
        grid = g.trimesh.path.creation.grid(
            side=10.0, plane_origin=[5.0, 0, 0], plane_normal=[1, 0, 0]
        )
        # make sure plane is applied correctly
        assert g.np.allclose(grid.extents, [0, 20, 20])
        assert g.np.allclose(grid.bounds, [[5, -10, -10], [5, 10, 10]])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
