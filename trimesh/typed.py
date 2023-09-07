from typing import Any, Union, List, TypeAlias, Sequence
from numpy import ndarray, float64, int64

import numpy as np


#NDArray: TypeAlias = ndarray
ArrayLike: TypeAlias = Union[Sequence, ndarray]

from numpy.typing import NDArray

def _check(values: ArrayLike[float64]) -> NDArray[int64]:
    return (np.array(values, dtype=float64) * 100).astype(int64)

