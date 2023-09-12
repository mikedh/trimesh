from typing import Sequence, Union

import numpy as np

# our default integer and floating point types
from numpy import float64, int64

try:
    from numpy.typing import NDArray
except BaseException:
    NDArray = Sequence

# for input arrays we want to say "list[int], ndarray[int64], etc"
# all the integer types
IntLike = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    int64,
    np.intc,
    np.intp,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

FloatLike = Union[float, np.float16, np.float32, float64, np.float_]
BoolLike = Union[bool, np.bool_]

ArrayLike = Sequence


__all__ = ["NDArray", "ArrayLike"]
