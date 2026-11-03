import numpy as np
import pytest

from trimesh.voxel import encoding as enc


@pytest.mark.parametrize("data", [[0, 0, 1, 1, 0], [1, 1], [0, 1], [1, 0]])
def test_binary_stripped_matches_dense(data):
    data = np.asarray(data, dtype=bool)
    binary = enc.BinaryRunLengthEncoding.from_dense(data)
    stripped, padding = binary.stripped
    expected, expected_padding = enc.DenseEncoding(data).stripped
    np.testing.assert_array_equal(stripped.dense, expected.dense)
    np.testing.assert_array_equal(padding, expected_padding)
    np.testing.assert_array_equal(np.pad(stripped.dense, padding[0]), data)


@pytest.mark.parametrize("indices", [[], [[1]], [[0], [1], [2]]])
def test_binary_gather_nd_preserves_query_axis(indices):
    data = np.array([False, True, False])
    indices = np.asarray(indices, dtype=np.int64).reshape((-1, 1))
    binary = enc.BinaryRunLengthEncoding.from_dense(data)
    actual = binary.gather_nd(indices)
    np.testing.assert_array_equal(actual, data[indices[:, 0]])
    assert actual.shape == (len(indices),)


@pytest.mark.parametrize("dtype", [bool, np.float32, np.int16])
def test_run_length_flip_preserves_value_dtype(dtype):
    data = np.array([0, 1, 1, 0, 0], dtype=dtype)
    encoded = enc.RunLengthEncoding.from_dense(data, dtype=dtype)
    flipped = encoded.flip(0)
    assert flipped.dtype == np.dtype(dtype)
    assert flipped.dense.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(flipped.dense, data[::-1])
