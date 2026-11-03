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


def test_merge_byte_counts_without_overflow():
    brle = np.array([200, 0, 200, 3], dtype=np.uint8)
    assert rl.merge_brle_lengths(brle) == [400, 3]
    np.testing.assert_array_equal(rl.brle_to_brle(brle), [400, 3])

    rle = np.array([1, 200, 1, 200, 0, 3], dtype=np.uint8)
    values, lengths = rl.merge_rle_lengths(rle[::2], rle[1::2])
    assert values == [1, 0]
    assert lengths == [400, 3]
    np.testing.assert_array_equal(rl.rle_to_rle(rle), [1, 400, 0, 3])
    np.testing.assert_array_equal(rl.rle_to_brle(rle), [0, 400, 3, 0])
