import numpy as np

from trimesh.typed import ArrayLike, NDArray, float64, int64


# see if we pass mypy
def _check(values: ArrayLike) -> NDArray[int64]:
    return (np.array(values, dtype=float64) * 100).astype(int64)


def _run() -> NDArray[int64]:
    return _check(values=[1, 2])
