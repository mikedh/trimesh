try:
    from . import generic as g
except BaseException:
    import generic as g

import typing

import numpy as np

from trimesh.typed import (
    ArrayLike,
    NDArray,
    NDArray1D,
    NDArray2D,
    NDArray3D,
    float64,
    int64,
)


# see if we pass mypy
def _check(values: ArrayLike) -> NDArray[int64]:
    return (np.array(values, dtype=float64) * 100).astype(int64)


def _run() -> NDArray[int64]:
    return _check(values=[1, 2])


class TypedTest(g.unittest.TestCase):
    def alias_args(self, alias):
        # an array alias is `numpy.ndarray[shape, numpy.dtype[scalar]]`
        assert typing.get_origin(alias) is np.ndarray
        shape, dtype = typing.get_args(alias)
        assert typing.get_origin(dtype) is np.dtype
        (scalar,) = typing.get_args(dtype)
        return typing.get_args(shape), scalar

    def test_dimension(self):
        # each alias must report the shape arity its name promises
        for alias, ndim in ((NDArray1D, 1), (NDArray2D, 2), (NDArray3D, 3)):
            shape, scalar = self.alias_args(alias[np.float64])
            assert shape == (int,) * ndim
            # dtype slot must be a concrete np.dtype, not a leaked TypeVar —
            # the numpy 2.5 regression beartype cannot reduce
            assert np.dtype(scalar) == np.dtype(np.float64)

    def test_scalar(self):
        # a spread of scalar kinds stay np.dtype-coercible through the alias
        for scalar in (np.float64, np.int64, np.uint8, np.bool_):
            _shape, got = self.alias_args(NDArray1D[scalar])
            assert np.dtype(got) == np.dtype(scalar)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
