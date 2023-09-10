from typing import Any, Sequence, Union

import numpy as np
from numpy import float64, int64

# NDArray: TypeAlias = ndarray
# ArrayLike: TypeAlias = Union[Sequence, ndarray]

try:
    from numpy.typing import NDArray
except BaseException:
    # NDArray = ndarray
    pass

# for input arrays we want to say "list[int], ndarray[int64], etc"
IntLike = Union[int, np.int64]
FloatLike = Union[float, np.float64]
BoolLike = Union[bool, np.bool_]

ArrayLike = Sequence
# this should pass mypy eventually

def _check(values: ArrayLike[FloatLike]) -> NDArray[int64]:
    return (np.array(values, dtype=float64) * 100).astype(int64)

def _run() -> NDArray[int64]:
    return _check(values=[1, 2])
    

__all__ = ['NDArray', 'ArrayLike']
