def test_line_closed():
    import numpy as np

    from trimesh.path.entities import Line

    e = Line(points=[0, 1, 2], closed=False)
    assert np.allclose(e.points, [0, 1, 2])

    e = Line(points=[0, 1, 2], closed=True)
    assert np.allclose(e.points, [0, 1, 2, 0])

    e = Line(points=[0, 1, 2, 0], closed=True)
    assert np.allclose(e.points, [0, 1, 2, 0])

    # should it really drop the last point... that seems weird
    e = Line(points=[0, 1, 2, 0], closed=False)
    assert np.allclose(e.points, [0, 1, 2, 0])


if __name__ == "__main__":
    import trimesh

    trimesh.util.attach_to_log()
    test_line_closed()
