import numpy as np


def test_resample_original():
    # check to see if `include_original` works

    from shapely.geometry import Polygon

    from trimesh.path.traversal import resample_path

    ori = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)

    re = resample_path(ori, step=0.25, include_original=True)

    a, b = Polygon(ori), Polygon(re)
    assert np.isclose(a.area, b.area)
    assert np.isclose(a.length, b.length)


if __name__ == "__main__":
    test_resample_original()
