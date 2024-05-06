import numpy as np
from pyinstrument import Profiler
from shapely.geometry import Point, Polygon

import trimesh
from trimesh.creation import sweep_polygon

trimesh.tol.strict = True


def arc_2d(final_angle, d=1, start_angle=0, splits=20):
    xs = np.linspace(start_angle, final_angle, splits)
    return np.stack((d / 2 * np.cos(xs), d / 2 * np.sin(xs)), axis=1)


def test_simple_closed(h1=1, w=2, r=4.4):
    # a simple closed path
    square = Polygon([(-0.01, -h1 / 2), (w, -h1 / 2), (w, +h1 / 2), (-0.01, +h1 / 2)])
    circle = Point([0, 0]).buffer(1.0)

    # a simple square of side `r`
    path = [[0, 0, 0], [r, 0, 0], [r, r, 0], [0, r, 0], [0, 0, 0]]

    a = sweep_polygon(square, path)
    assert a.is_volume

    aa = sweep_polygon(square, path, connect=False)
    assert aa.is_volume

    # should have the same bounds but no longer a volume
    aaa = sweep_polygon(square, path, connect=False, cap=False)
    assert np.allclose(aa.bounds, aaa.bounds)
    assert not aaa.is_volume

    b = sweep_polygon(circle, path)
    assert b.is_volume


def test_simple_extrude(height=10):
    # make sure sweep behaves okay on a simple single segment path
    # this should be identical to the extrude operation
    circle = Point([0, 0]).buffer(1.0)

    # a simple extrusion
    path = [[0, 0, 0], [0, 0, height]]
    # will be running asserts inside function
    b = sweep_polygon(circle, path)

    # should be a straight extrude along Z
    expected = np.append(np.ptp(np.reshape(circle.bounds, (2, 2)), axis=0), height)
    assert np.allclose(expected, b.extents)

    # should be well constructed
    assert b.is_volume
    # volume should correspond to expected cylinder area
    assert np.isclose(b.volume, circle.area * height)


def test_simple_open():
    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 100)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10

    a = sweep_polygon(circle, path)
    assert a.is_volume


def test_spline_3D():
    circle = Point([0, 0]).buffer(1.0)
    # make a 3D noodle using spline smoothing
    path = trimesh.path.simplify.resample_spline(
        [[0, 0, 0], [4, 4, 0], [5, 5, 0], [10, 0, 10], [0, 20, 0]], smooth=0.25, count=100
    )

    a = sweep_polygon(circle, path)
    assert a.is_volume
    assert a.body_count == 1


def test_screw():
    h = 200
    d = 8
    h1 = 1
    w = 2
    d = d - w
    h2 = h1

    lead = 2
    splits_per_turn = 4 * 20
    polygon = np.array([(-0.01, -h1 / 2), (w, -h2 / 2), (w, +h2 / 2), (-0.01, +h1 / 2)])
    h_ = h + h1
    n = int(np.ceil(splits_per_turn * h_ / lead))
    turns = n / splits_per_turn
    xys = arc_2d(2 * np.pi * turns, d, splits=n)
    zs = np.linspace(0, h_, n)[:, None]
    spin_path = np.concatenate((xys, zs), axis=1)

    with Profiler() as P:
        x = sweep_polygon(Polygon(polygon), spin_path)
    P.print()

    assert x.is_volume
    # should have produced a result corresponding to the input dimensions
    assert np.allclose(x.extents[:2], d + w * 2, atol=0.01)
    assert np.allclose(x.extents[2], h + h1 * 2, atol=0.01)

    # check without capping: should be the same size just without caps
    o = sweep_polygon(Polygon(polygon), spin_path, cap=False, connect=False)
    assert np.allclose(x.bounds, o.bounds)
    assert not o.is_volume

    # try spinning along an angle
    sweep_polygon(
        Polygon(polygon), spin_path, angles=np.linspace(0.0, np.pi / 2, len(spin_path))
    )

    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 100)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10

    a = sweep_polygon(circle, path)
    assert a.is_volume


if __name__ == "__main__":
    test_simple_closed()
    test_simple_extrude()
    test_simple_open()
    test_screw()
    test_spline_3D()
