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


def test_resample_original_case():
    # check a case that was failing to insert the original points

    from scipy.spatial import KDTree
    from shapely.geometry import LineString

    from trimesh.path.traversal import PathSample

    test_path = np.array([[0, 0], [1, 0], [1.1, 0], [2, 0], [3, 0]], dtype=float)

    ps = PathSample(test_path)
    # note the length of the original path is 3, so this is sampling past the end
    a = ps.sample(np.arange(0, 5, 1.5), include_original=True)

    # get the distance to the closest point from our test path to the sample
    radius, index = KDTree(test_path).query(a)

    # assert sampling is increasing in index
    assert ((index[1:] - index[:-1]) >= 0).all()

    # the sample path includes points that didn't exist in the original
    # so we need to check that every index has at least one zero-radius
    radius_ok = np.zeros(len(test_path), dtype=bool)
    radius_ok[index[radius < 1e-12]] = True
    assert radius_ok.all(), "not every point was inserted!"

    # run a few more quick checks to make sure the path wasn't trashed
    a = LineString(test_path)
    b = LineString(a)

    assert np.isclose(a.length, b.length)
    assert np.allclose(a.bounds, b.bounds)


if __name__ == "__main__":
    test_resample_original()
    test_resample_original_case()


def test_sample_include_original_keeps_length():
    from trimesh.path.traversal import PathSample

    # segments of length 1, 2 and 3 so the path is 6 long
    path = np.array([[0, 0], [1, 0], [1, 2], [4, 2]], dtype=np.float64)
    sampler = PathSample(path)
    for distances in ([0.0, 2.0], [2.0], [1.0, 3.0, 6.0]):
        result = sampler.sample(np.array(distances), include_original=True)
        length = np.linalg.norm(np.diff(result, axis=0), axis=1).sum()
        assert np.isclose(length, 6.0)
