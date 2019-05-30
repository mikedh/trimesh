try:
    from . import generic as g
except BaseException:
    import generic as g
from trimesh.voxel import encoding as enc
from trimesh.voxel import runlength as rl
np = g.np

shape = (10, 10, 10)
dense_data = np.random.uniform(size=shape) < 0.2
rle = enc.RunLengthEncoding.from_dense(
    dense_data.reshape((-1,)), dtype=bool).reshape(shape)
brle = enc.BinaryRunLengthEncoding.from_dense(
    dense_data.reshape((-1,))).reshape(shape)
dense = enc.DenseEncoding(dense_data)

encodings = (
    dense,
    rle,
    brle,
)


class EncodingTest(g.unittest.TestCase):

    def _test_dense(self, encoding, data):
        np.testing.assert_equal(encoding.dense, data)

    def _test_rle(self, encoding, data):
        np.testing.assert_equal(
            encoding.run_length_data(), rl.dense_to_rle(data))

    def _test_brle(self, encoding, data):
        np.testing.assert_equal(
            encoding.binary_run_length_data(),
            rl.dense_to_brle(data))

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
        if hasattr(axes, '__iter__'):
            for a in axes:
                data = np.flip(data, a)
        else:
            data = np.flip(data, axes)
        self._test_dense(encoding.flip(axes), data)

    def _test_composite(
            self, encoding, data, transpose=(0, 2, 1), reshape=(5, 2, -1),
            flatten=True, flip=(0, 2)):
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
            if hasattr(flip, '__iter__'):
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
            0, 1, 2,
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
                self.assertTrue(
                    encoding.transpose(perm).transpose(perm) is encoding)

    def test_flat(self):
        for encoding in encodings:
            self._test_dense(encoding.flat, dense_data.reshape((-1,)))

    def test_reshape(self):
        shapes = (
            (10, 10, 10),
            (5, 20, 10),
            (50, 4, 5),
            (-1, 4, 5)
        )
        for encoding in encodings:
            for shape in shapes:
                self._test_dense(
                    encoding.reshape(shape), dense_data.reshape(shape))

    def test_composite(self):
        for encoding in encodings:
            self._test_composite(encoding, dense_data)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
