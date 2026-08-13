import numpy as np

import trimesh


def test_line_line():

    from trimesh.path.intersections import line_line

    # simple 2D
    hit, point = line_line(origins=[[-10, 0], [0, -10]], directions=[[1, 0], [0, 1]])
    assert hit
    assert np.allclose(point, [0, 0])

    # simple 3D
    hit, point = line_line(
        origins=[[-10, 0, 0], [0, -10, 0]], directions=[[1, 0, 0], [0, 1, 0]]
    )
    assert hit
    assert np.allclose(point, [0, 0, 0])

    # simple 3D with normal
    hit, point = line_line(
        origins=[[-10, 0, 0], [0, -10, 0]],
        directions=[[1, 0, 0], [0, 1, 0]],
        plane_normal=[0, 0, 1],
    )
    assert hit
    assert np.allclose(point, [0, 0, 0])

    # simple 3D with wrong normal
    hit, point = line_line(
        origins=[[-10, 0, 0], [0, -10, 0]],
        directions=[[1, 0, 0], [0, 1, 0]],
        plane_normal=[1, 0, 0],
    )
    assert not hit
    assert point is None


if __name__ == "__main__":
    trimesh.util.attach_to_log()
    test_line_line()
