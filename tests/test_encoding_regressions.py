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


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_run_length_factories_infer_default_storage(dtype):
    rle = np.array([0, 2, 1, 3], dtype=dtype)
    brle = np.array([2, 3], dtype=dtype)
    expected = [0, 0, 1, 1, 1]
    for result in (
        enc.RunLengthEncoding.from_rle(rle),
        enc.RunLengthEncoding.from_brle(brle),
        enc.BinaryRunLengthEncoding.from_brle(brle),
        enc.BinaryRunLengthEncoding.from_rle(rle),
    ):
        np.testing.assert_array_equal(result.dense, expected)


@pytest.mark.parametrize(
    "runs,storage_dtype,value_dtype",
    [
        ([200, 2, 100, 3], np.uint8, np.uint8),
        ([100, 3, -50, 2], np.int8, np.int8),
        ([2, 3, 0, 2], np.int64, bool),
        ([10000, 10, -10000, 10], np.int64, np.float16),
    ],
)
def test_run_length_sum_uses_dense_arithmetic(runs, storage_dtype, value_dtype):
    encoded = enc.RunLengthEncoding(
        np.array(runs, dtype=storage_dtype), dtype=value_dtype
    )
    expected = encoded.dense.sum()
    assert encoded.sum == expected
    assert encoded.sum.dtype == expected.dtype
