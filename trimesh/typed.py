from typing import Any

import numpy as np
from numpy import float64, int64

# NDArray: TypeAlias = ndarray
# ArrayLike: TypeAlias = Union[Sequence, ndarray]

try:
    from numpy.typing import NDArray
except BaseException:
    NDArray = Any

# todo make this a generic List|ndarray
ArrayLike = NDArray

# this should pass mypy eventually
def _check(values: ArrayLike[float64]) -> NDArray[int64]:
    return (np.array(values, dtype=float64) * 100).astype(int64)

__all__ = ['NDArray', 'ArrayLike']
