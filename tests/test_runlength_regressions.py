import numpy as np
import pytest

from trimesh.voxel import runlength as rl


@pytest.mark.parametrize("counts", [[2, 3], [2, 3, 4], [0, 3, 2, 0], [4, 0], []])
def test_binary_reverse_matches_dense(counts):
    counts = np.asarray(counts, dtype=np.uint8)
    reversed_counts = rl.brle_reverse(counts)
    np.testing.assert_array_equal(
        rl.brle_to_dense(reversed_counts), rl.brle_to_dense(counts)[::-1]
    )
    assert reversed_counts.dtype == counts.dtype
    np.testing.assert_array_equal(
        rl.brle_to_dense(rl.brle_reverse(reversed_counts)), rl.brle_to_dense(counts)
    )


@pytest.mark.parametrize("values", [[7, 9], [1.5, -2.5], ["free", "occupied"]])
def test_binary_decode_replacement_values(values):
    values = np.asarray(values)
    decoded = rl.brle_to_dense([2, 3, 1, 0], vals=values)
    np.testing.assert_array_equal(decoded, values[[0, 0, 1, 1, 1, 0]])
    assert decoded.dtype == values.dtype
