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
