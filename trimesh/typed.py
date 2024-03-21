from pathlib import Path
from typing import (
    IO,
    Any,
    BinaryIO,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# our default integer and floating point types
from numpy import float64, floating, int64, integer
from numpy.typing import ArrayLike, NDArray

# most loader routes take `file_obj` which can either be
# a file-like object or a file path
Loadable = Union[str, Path, IO, Dict]

IntLike = Union[int, integer]
FloatLike = Union[float, floating]

__all__ = [
    "NDArray",
    "ArrayLike",
    "Optional",
    "Sequence",
    "Iterable",
    "Loadable",
    "IO",
    "BinaryIO",
    "List",
    "Dict",
    "Any",
    "Tuple",
    "float64",
    "int64",
]
