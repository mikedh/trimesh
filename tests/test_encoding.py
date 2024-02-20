try:
    from . import generic as g
except BaseException:
    import generic as g
enc = g.trimesh.voxel.encoding
rl = g.trimesh.voxel.runlength
np = g.np

shape = (10, 10, 10)
dense_data = np.random.uniform(size=shape) < 0.2
rle = enc.RunLengthEncoding.from_dense(dense_data.reshape((-1,)), dtype=bool).reshape(
    shape
)
brle = enc.BinaryRunLengthEncoding.from_dense(dense_data.reshape((-1,))).reshape(shape)
dense = enc.DenseEncoding(dense_data)
indices = np.column_stack(np.where(dense_data))
values = np.ones(shape=(indices.shape[0],), dtype=bool)
sparse = enc.SparseEncoding(indices, values, shape)

encodings = (
    dense,
    rle,
    brle,
    sparse,
)


class EncodingTest(g.unittest.TestCase):
    def _test_dense(self, encoding, data):
        np.testing.assert_equal(encoding.dense, data)

    def _test_rle(self, encoding, data):
        np.testing.assert_equal(encoding.run_length_data(), rl.dense_to_rle(data))

    def _test_brle(self, encoding, data):
        np.testing.assert_equal(encoding.binary_run_length_data(), rl.dense_to_brle(data))

    def _test_transpose(self, encoding, data, perm):
        encoding = encoding.transpose(perm)
        data = data.transpose(perm)
        self._test_dense(encoding, data)

    def _test_reshape(self, encoding, data, shape):
        encoding = encoding.reshape(shape)
        data = data.reshape(shape)
        self._test_dense(encoding, data)

    def _test_flat(self, encoding, data):
        self._test_dense(encoding.flat, data.reshape((-1,)))

    def _test_flipped(self, encoding, data, axes):
        if hasattr(axes, "__iter__"):
            for a in axes:
                data = np.flip(data, a)
        else:
            data = np.flip(data, axes)
        self._test_dense(encoding.flip(axes), data)

    def _test_composite(
        self,
        encoding,
        data,
        transpose=(0, 2, 1),
        reshape=(5, 2, -1),
        flatten=True,
        flip=(0, 2),
    ):
        def check():
            self._test_dense(encoding, data)

        if transpose is not None:
            encoding = encoding.transpose(transpose)
            data = data.transpose(transpose)
            check()
        if reshape is not None:
            encoding = encoding.reshape(reshape)
            data = data.reshape(reshape)
            check()
        if flip:
            encoding = encoding.flip(flip)
            if hasattr(flip, "__iter__"):
                for a in flip:
                    data = np.flip(data, a)
            else:
                data = np.flip(data, flip)
            check()
        if flatten:
            encoding = encoding.flat
            data = data.reshape((-1,))
            check()
        self._test_dense(encoding, data)
        self._test_rle(encoding, data)
        self._test_brle(encoding, data)

    def test_dense(self):
        for encoding in encodings:
            self._test_dense(encoding, dense_data)

    def test_rle(self):
        for encoding in encodings:
            self._test_rle(encoding.flat, dense_data.reshape((-1,)))

    def test_brle(self):
        for encoding in encodings:
            self._test_brle(encoding.flat, dense_data.reshape((-1,)))

    def test_flipped(self):
        axes = (
            0,
            1,
            2,
            (0,),
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 1, 2),
        )
        for encoding in encodings:
            for ax in axes:
                self._test_flipped(encoding, dense_data, ax)
            self.assertTrue(encoding.flip((0, 0)) is encoding)

    def test_transpose(self):
        perms = (
            (0, 2, 1),
            (0, 1, 2),
            (2, 1, 0),
        )
        for encoding in encodings:
            for perm in perms:
                self._test_transpose(encoding, dense_data, perm)
            perm = (0, 2, 1)
            if not isinstance(encoding, enc.DenseEncoding):
                self.assertTrue(encoding.transpose(perm).transpose(perm) is encoding)

    def test_flat(self):
        for encoding in encodings:
            self._test_dense(encoding.flat, dense_data.reshape((-1,)))

    def test_reshape(self):
        shapes = ((10, 10, 10), (5, 20, 10), (50, 4, 5), (-1, 4, 5))
        for encoding in encodings:
            for shape in shapes:
                self._test_dense(encoding.reshape(shape), dense_data.reshape(shape))

    def test_composite(self):
        for encoding in encodings:
            self._test_composite(encoding, dense_data)

    def test_dense_stripped(self):
        base_shape = (5, 5, 5)
        dense = np.ones(base_shape, dtype=bool)
        padding = [[2, 2], [2, 2], [2, 2]]
        dense = np.pad(dense, padding, mode="constant")
        encoding = enc.DenseEncoding(dense)
        stripped, calculated_padding = encoding.stripped
        np.testing.assert_equal(calculated_padding, padding)
        np.testing.assert_equal(stripped.shape, base_shape)
        np.testing.assert_equal(stripped.dense, 1)

    def test_sparse_stripped(self):
        box = g.trimesh.primitives.Box()
        box.apply_translation([0.5, 0.5, 0.5])  # center at origin
        box.apply_scale(5)  # 0 -> 5
        expected_sparse_indices = np.array(box.vertices)
        box.apply_translation([2, 2, 2])  # 2 -> 7
        sparse = np.array(box.vertices, dtype=int)
        encoding = enc.SparseBinaryEncoding(sparse, shape=(9, 9, 9))
        stripped, calculated_padding = encoding.stripped
        np.testing.assert_equal(stripped.sparse_indices, expected_sparse_indices)
        np.testing.assert_equal(calculated_padding, 2 * np.ones((3, 2), dtype=int))

    def test_empty_stripped(self):
        res = 10
        encoding = enc.DenseEncoding(np.zeros((res,) * 3, dtype=bool))
        stripped, calculated_padding = encoding.stripped
        self.assertEqual(stripped.size, 0)
        np.testing.assert_equal(calculated_padding, [[0, res], [0, res], [0, res]])

    def test_is_empty(self):
        res = 10
        empty = np.zeros((res,), dtype=bool)
        not_empty = np.zeros((res,), dtype=bool)
        not_empty[[1, 2, 5, 6, 7]] = True
        self.assertTrue(enc.DenseEncoding(empty).is_empty)
        self.assertFalse(enc.DenseEncoding(not_empty).is_empty)

        for cls in (
            enc.SparseEncoding,
            enc.RunLengthEncoding,
            enc.BinaryRunLengthEncoding,
        ):
            self.assertTrue(cls.from_dense(empty).is_empty)
            self.assertFalse(cls.from_dense(not_empty).is_empty)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
