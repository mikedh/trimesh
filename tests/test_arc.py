try:
    from . import generic as g
except BaseException:
    import generic as g


@g.pytest.mark.parametrize("dimension", [2, 3])
@g.pytest.mark.parametrize("close", [False, True])
def test_discrete_endpoints(dimension, close):
    points = g.np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
    if dimension == 3:
        points = g.trimesh.transform_points(
            g.trimesh.util.stack_3D(points), next(g.random_transforms(1))
        )
    discrete = g.trimesh.path.arc.discretize_arc(points, close=close)
    assert discrete.shape[1] == dimension
    if close:
        assert g.np.array_equal(discrete[0], discrete[-1])
    else:
        assert g.np.array_equal(discrete[[0, -1]], points[[0, -1]])


def test_arclength():
    """
    Check our arc length against the discrete version
    """
    # make sure we actually check something
    checked = set()

    # loop through our 2D corpus
    for p in g.get_2D():
        for e in p.entities:
            # add the entity type
            checked.add(e.__class__.__name__)

            # discretize to an (n, 2) polyline
            d = e.discrete(vertices=p.vertices)
            if len(d) == 0:
                # didn't discretize, `Text` entities do this
                continue

            # get the length as a sum of line segment lengths
            d_l = g.np.linalg.norm(g.np.diff(d, axis=0), axis=1).sum()
            # get the reported length from the entity itself
            # this should be analytical for arcs, splines, etc
            e_l = e.length(vertices=p.vertices)

            # assert the two are within 0.1%
            assert g.np.isclose(d_l, e_l, rtol=0.001), f"{d_l} != {e_l}"

    # make sure we checked at least one arc and line
    assert "Arc" in checked
    assert "Line" in checked


def test_center():
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
    assert g.np.linalg.norm(C - res_center) < g.tol_path.zero
    # large magnitude arc failed some coplanar tests
    c = g.trimesh.path.arc.arc_center(
        [
            [30156.18, 1673.64, -2914.56],
            [30152.91, 1780.09, -2885.51],
            [30148.3, 1875.81, -2857.79],
        ]
    )
    assert len(c.center) == 3


def test_length():
    # open arcs are `radius * angle` and closed arcs the full circumference
    radius = 2.0
    for angle in [g.np.pi / 2, g.np.pi, g.np.pi * 1.5]:
        theta = g.np.array([0.0, angle / 2.0, angle])
        vertices = g.np.column_stack((g.np.cos(theta), g.np.sin(theta))) * radius
        arc = g.trimesh.path.entities.Arc([0, 1, 2])
        length = arc.length(vertices)
        assert g.np.isclose(length, radius * angle)
        # the discrete polyline is inscribed so it is slightly shorter
        chords = g.np.diff(arc.discrete(vertices), axis=0)
        polyline = g.np.linalg.norm(chords, axis=1).sum()
        assert polyline <= length and g.np.isclose(polyline, length, rtol=1e-3)

    circle = g.trimesh.path.creation.circle(radius=radius)
    assert g.np.isclose(circle.length, g.np.pi * 2.0 * radius)


def test_center_random():
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

    for center, radius, three, transform in zip(
        center_3D, radii, points_3D, g.random_transforms(len(radii), translate=0.0)
    ):
        center = g.trimesh.transformations.transform_points([center], transform)[0]
        three = g.trimesh.transformations.transform_points(three, transform)

        info = arc_center(three)

        assert g.np.allclose(center, info["center"])
        assert g.np.allclose(radius, info["radius"])


def test_multiroot():
    """
    Test a Path2D object containing polygons nested in
    the interiors of other polygons.
    """
    inner = g.trimesh.creation.annulus(r_min=0.5, r_max=0.6, height=1.0)
    outer = g.trimesh.creation.annulus(r_min=0.9, r_max=1.0, height=1.0)
    m = inner + outer

    s = m.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, 0])
    p = s.to_2D()[0]

    assert len(p.polygons_closed) == 4
    assert len(p.polygons_full) == 2
    assert len(p.root) == 2
    g.check_path2D(p)


def test_circle_is_closed():
    path = g.trimesh.path.creation.circle(radius=2, segments=8)
    path += g.trimesh.path.creation.circle(radius=1, segments=8)

    assert len(path.discrete) == 2
    for d in path.discrete:
        assert (d[0] == d[-1]).all(), g.np.linalg.norm(d[0] - d[-1])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()

    # test_circle_is_closed()
    test_arclength()
