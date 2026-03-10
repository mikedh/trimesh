import generic as g


def test_line_closed_constructor_normalizes_points():
    line = g.trimesh.path.entities.Line(points=[0, 1, 2], closed=True)

    assert line.closed
    assert line.points.tolist() == [0, 1, 2, 0]


def test_line_open_constructor_removes_duplicate_endpoint():
    line = g.trimesh.path.entities.Line(points=[0, 1, 2, 0], closed=False)

    assert not line.closed
    assert line.points.tolist() == [0, 1, 2]
