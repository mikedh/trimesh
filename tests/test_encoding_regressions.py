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
