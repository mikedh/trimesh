try:
    from . import generic as g
except BaseException:
    import generic as g


class ArcTests(g.unittest.TestCase):
    def test_center(self):
        from trimesh.path.arc import arc_center

        test_points = [[[0, 0], [1.0, 1], [2, 0]]]
        test_results = [[[1, 0], 1.0]]
        points = test_points[0]
        res_center, res_radius = test_results[0]
        center_info = arc_center(points)
        C, R, _N, _angle = (
            center_info["center"],
            center_info["radius"],
            center_info["normal"],
            center_info["span"],
        )

        assert abs(R - res_radius) < g.tol_path.zero
        assert g.trimesh.util.euclidean(C, res_center) < g.tol_path.zero
        # large magnitude arc failed some coplanar tests
        c = g.trimesh.path.arc.arc_center(
            [
                [30156.18, 1673.64, -2914.56],
                [30152.91, 1780.09, -2885.51],
                [30148.3, 1875.81, -2857.79],
            ]
        )
        assert len(c.center) == 3

    def test_center_random(self):
        from trimesh.path.arc import arc_center

        # Test that arc centers work on well formed random points in 2D and 3D
        min_angle = g.np.radians(2)
        count = 1000

        center_3D = (g.random((count, 3)) - 0.5) * 50
        center_2D = center_3D[:, 0:2]
        radii = g.np.clip(g.random(count) * 100, min_angle, g.np.inf)

        angles = g.random((count, 2)) * (g.np.pi - min_angle) + min_angle
        angles = g.np.column_stack((g.np.zeros(count), g.np.cumsum(angles, axis=1)))

        points_2D = g.np.column_stack(
            (
                g.np.cos(angles[:, 0]),
                g.np.sin(angles[:, 0]),
                g.np.cos(angles[:, 1]),
                g.np.sin(angles[:, 1]),
                g.np.cos(angles[:, 2]),
                g.np.sin(angles[:, 2]),
            )
        ).reshape((-1, 6))
        points_2D *= radii.reshape((-1, 1))
        points_2D += g.np.tile(center_2D, (1, 3))
        points_2D = points_2D.reshape((-1, 3, 2))
        points_3D = g.np.column_stack(
            (
                points_2D.reshape((-1, 2)),
                g.np.tile(center_3D[:, 2].reshape((-1, 1)), (1, 3)).reshape(-1),
            )
        ).reshape((-1, 3, 3))
        for center, radius, three in zip(center_2D, radii, points_2D):
            info = arc_center(three)

            assert g.np.allclose(center, info["center"])
            assert g.np.allclose(radius, info["radius"])

        for center, radius, three in zip(center_3D, radii, points_3D):
            transform = g.trimesh.transformations.random_rotation_matrix()
            center = g.trimesh.transformations.transform_points([center], transform)[0]
            three = g.trimesh.transformations.transform_points(three, transform)

            info = arc_center(three)

            assert g.np.allclose(center, info["center"])
            assert g.np.allclose(radius, info["radius"])

    def test_multiroot(self):
        """
        Test a Path2D object containing polygons nested in
        the interiors of other polygons.
        """
        inner = g.trimesh.creation.annulus(r_min=0.5, r_max=0.6, height=1.0)
        outer = g.trimesh.creation.annulus(r_min=0.9, r_max=1.0, height=1.0)
        m = inner + outer

        s = m.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, 0])
        p = s.to_planar()[0]

        assert len(p.polygons_closed) == 4
        assert len(p.polygons_full) == 2
        assert len(p.root) == 2
        g.check_path2D(p)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
