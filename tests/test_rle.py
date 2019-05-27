try:
    from . import generic as g
except BaseException:
    import generic as g
from trimesh import rle
np = g.np


def random_rle_encoding(n=20, max_value=255, dtype=np.uint8):
    return (np.random.uniform(size=(n,),)*(max_value-1) + 1).astype(np.uint8)


def random_brle_encoding(n=20, max_value=255, dtype=np.uint8):
    return (np.random.uniform(size=(n,)) * (max_value-1) + 1).astype(dtype)


class RleTest(g.unittest.TestCase):
    """Tests should inherit from this class along with unittest.TestCase."""

    def test_rle_encode_decode(self):
        small = np.array([3]*500 + [5]*1000 + [2], dtype=np.uint8)
        rand = (np.random.uniform(size=(10000,)) > 0.05).astype(np.uint8)
        for original in [small, rand]:
            for dtype in [np.uint8, np.int64]:
                enc = rle.dense_to_rle(original, dtype=dtype)
                dec = rle.rle_to_dense(enc)
                np.testing.assert_equal(original, dec)

    def test_rle_decode_encode(self):
        small = [3, 5, 5, 10, 2, 1]
        rand = random_rle_encoding()
        for original in [small, rand]:
            for dtype in [np.uint8, np.int64]:
                dec = rle.brle_to_dense(original)
                enc = rle.dense_to_brle(dec, dtype=dtype)
                np.testing.assert_equal(original, enc)

    def test_merge_rle_lengths(self):
        v0, l0 = [5, 5, 2], [10, 10, 1]
        v1, l1 = [5, 2], [20, 1]
        v0, l0 = rle.merge_rle_lengths(v0, l0)
        np.testing.assert_equal(v0, v1)
        np.testing.assert_equal(l0, l1)

    def test_split_long_rle_lengths(self):
        v0, l0 = [5], [300]
        v1, l1 = [5, 5], [255, 45]

        v0, l0 = rle.split_long_rle_lengths(v0, l0, dtype=np.uint8)
        np.testing.assert_equal(v0, v1)
        np.testing.assert_equal(l0, l1)

        v0, l0 = [5, 2, 3], [10, 1000, 4]
        v1, l1 = [5, 2, 2, 2, 2, 3], [10, 255, 255, 255, 235, 4]

        v0, l0 = rle.split_long_rle_lengths(v0, l0, dtype=np.uint8)
        np.testing.assert_equal(v0, v1)
        np.testing.assert_equal(l0, l1)

    def test_rle_length(self):
        np.testing.assert_equal(
                rle.rle_length(
                    [0, 5, 1, 3, 0, 6]), rle.brle_length([5, 3, 6, 0]))

    def test_rle_to_brle(self):
        np.testing.assert_equal(
                rle.rle_to_brle([0, 5, 1, 3, 0, 10]),
                [5, 3, 10, 0])
        np.testing.assert_equal(
                rle.rle_to_brle([0, 5, 0, 3, 1, 10]),
                [8, 10])
        np.testing.assert_equal(
                rle.rle_to_brle([1, 5, 0, 3, 1, 10]),
                [0, 5, 3, 10])
        np.testing.assert_equal(
                rle.rle_to_brle([1, 5, 0, 2]),
                [0, 5, 2, 0])

    def test_rle_to_dense(self):
        np.testing.assert_equal(
            rle.rle_to_dense([5, 3, 4, 10]), [5]*3 + [4]*10)
        np.testing.assert_equal(
            rle.rle_to_dense([5, 300, 4, 100]), [5]*300 + [4]*100)

    def test_brle_encode_decode(self):
        small = np.array([False]*500 + [True]*1000 + [False], dtype=bool)
        rand = np.random.uniform(size=(10000,)) > 0.05
        for original in [small, rand]:
            for dtype in [np.uint8, np.int64]:
                enc = rle.dense_to_brle(original, dtype=dtype)
                dec = rle.brle_to_dense(enc)
                np.testing.assert_equal(original, dec)

    def test_brle_decode_encode(self):
        small = [3, 5, 5, 10, 2, 1]
        rand = random_brle_encoding()
        for original in [small, rand]:
            for dtype in [np.uint8, np.int64]:
                dec = rle.brle_to_dense(original)
                enc = rle.dense_to_brle(dec, dtype=dtype)
                np.testing.assert_equal(original, enc)

    def test_brle_logical_not(self):
        original = random_brle_encoding(dtype=np.int64)
        notted = rle.brle_logical_not(original)
        dense_notted = rle.brle_to_dense(notted)
        dense_original = rle.brle_to_dense(original)
        np.testing.assert_equal(
                dense_notted, np.logical_not(dense_original))

    def test_maybe_pad_brle(self):
        np.testing.assert_equal(rle.maybe_pad_brle([5], 0), [5, 0])
        np.testing.assert_equal(rle.maybe_pad_brle([5], 1), [0, 5])
        np.testing.assert_equal(rle.maybe_pad_brle([5, 3], 0), [5, 3])
        np.testing.assert_equal(rle.maybe_pad_brle([5, 3], 1), [0, 5, 3, 0])

    def test_merge_brle_lengths(self):
        np.testing.assert_equal(
                rle.merge_brle_lengths([10, 0, 10, 2]), [20, 2])
        np.testing.assert_equal(
                rle.merge_brle_lengths([10, 0, 10, 2]), [20, 2])
        np.testing.assert_equal(
                rle.merge_brle_lengths([10, 1, 10, 2]), [10, 1, 10, 2])
        np.testing.assert_equal(
                rle.merge_brle_lengths([0, 10, 2, 3]), [0, 10, 2, 3])

    def test_split_long_brle_lengths(self):
        np.testing.assert_equal(
                rle.split_long_brle_lengths([300, 600, 10], np.uint8),
                [255, 0, 45, 255, 0, 255, 0, 90, 10])

    def test_brle_split_merge(self):
        x = [300, 600, 10, 0]
        split = rle.split_long_brle_lengths(x, np.uint8)
        merged = rle.merge_brle_lengths(split)
        np.testing.assert_equal(merged, x)

    def test_brle_to_rle(self):
        brle_data = random_brle_encoding()
        brle_dense = rle.brle_to_dense(brle_data)
        rle_data = rle.brle_to_rle(brle_data)
        rle_dense = rle.rle_to_dense(rle_data)
        np.testing.assert_equal(brle_dense, rle_dense)
        np.testing.assert_equal(
                rle.brle_to_rle([0, 5, 2, 0]),
                [1, 5, 0, 2])

    def test_dense_to_brle(self):
        x = np.array([False]*300 + [True]*200 + [False]*1000)
        np.testing.assert_equal(rle.dense_to_brle(x), [300, 200, 1000, 0])
        np.testing.assert_equal(
                rle.dense_to_brle(x, np.uint8),
                [255, 0, 45, 200, 255, 0, 255, 0, 255, 0, 235, 0])

    def test_brle_to_dense(self):
        np.testing.assert_equal(
            rle.brle_to_dense(np.array([300, 200, 1000, 0], dtype=np.int64)),
            [False]*300 + [True]*200 + [False]*1000)
        np.testing.assert_equal(
            rle.brle_to_dense(np.array(
                [255, 0, 45, 200, 255, 0, 255, 0, 255, 0, 235, 0],
                dtype=np.int64)),
            [False]*300 + [True]*200 + [False]*1000)

    def test_brle_length(self):
        enc = random_brle_encoding(dtype=np.int64)
        dec = rle.brle_to_dense(enc)
        np.testing.assert_equal(len(dec), rle.brle_length(enc))

    def test_rle_mask(self):
        rle_data = random_rle_encoding()
        dense = rle.rle_to_dense(rle_data)
        mask = np.random.uniform(size=dense.shape) > 0.8
        expected = dense[mask]
        actual = tuple(rle.rle_mask(rle_data, mask))
        np.testing.assert_equal(actual, expected)

    def test_brle_mask(self):
        brle_data = random_brle_encoding()
        dense = rle.brle_to_dense(brle_data)
        mask = np.random.uniform(size=dense.shape) > 0.8
        expected = dense[mask]
        actual = tuple(rle.brle_mask(brle_data, mask))
        np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
