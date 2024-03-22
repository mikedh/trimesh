from io import BytesIO, StringIO
from pathlib import Path
from sys import version_info
from typing import (
    IO,
    Any,
    BinaryIO,
    Optional,
    TextIO,
    Union,
)

# our default integer and floating point types
from numpy import float64, floating, int64, integer
from numpy.typing import ArrayLike, NDArray

if version_info >= (3, 9):
    # use PEP585 hints on newer python
    List = list
    Tuple = tuple
    Dict = dict
    from collections.abc import Iterable, Sequence
else:
    from typing import Dict, Iterable, List, Sequence, Tuple

# most loader routes take `file_obj` which can either be
# a file-like object or a file path, or sometimes a dict
Loadable = Union[str, Path, IO, BytesIO, StringIO, BinaryIO, TextIO, Dict, None]

# if you type a function argument as an `int` and then pass
# a value from a numpy array like `np.ones(10, dtype=np.int64)[0]`
# you will have a type error as `np.integer` does not
# inherit from `int`
# these wrappers union numpy integers and python integers
IntLike = Union[int, integer]
FloatLike = Union[float, floating]
Numeric = Union[int, integer, float, floating]

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
    "Numeric",
]
