from typing import Sequence, Union

import numpy as np

# NDArray: TypeAlias = ndarray
# ArrayLike: TypeAlias = Union[Sequence, ndarray]

try:
    from numpy.typing import NDArray
except BaseException:
    # NDArray = ndarray
    pass

# for input arrays we want to say "list[int], ndarray[int64], etc"
# all the integer types
IntLike = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intc,
    np.intp,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

FloatLike = Union[float, np.float16, np.float32, np.float64, np.float128, np.float_]
BoolLike = Union[bool, np.bool_]

ArrayLike = Sequence


__all__ = ["NDArray", "ArrayLike"]
