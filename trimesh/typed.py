from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple, Union

# our default integer and floating point types
from numpy import float64, int64

try:
    from numpy.typing import ArrayLike, NDArray
except BaseException:
    NDArray = Sequence
    ArrayLike = Sequence

# most loader routes take `file_obj` which can either be
# a file-like object or a file path
Loadable = Union[str, Path, IO]

__all__ = [
    "NDArray",
    "ArrayLike",
    "Optional",
    "Loadable",
    "IO",
    "List",
    "Dict",
    "Tuple",
    "float64",
    "int64",
]
