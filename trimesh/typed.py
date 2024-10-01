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

from numpy import float64, floating, int64, integer, unsignedinteger

# requires numpy>=1.20
from numpy.typing import ArrayLike, NDArray

if version_info >= (3, 9):
    # use PEP585 hints on newer python
    List = list
    Tuple = tuple
    Dict = dict
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
else:
    from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

# most loader routes take `file_obj` which can either be
# a file-like object or a file path, or sometimes a dict

Stream = Union[IO, BytesIO, StringIO, BinaryIO, TextIO]
Loadable = Union[str, Path, Stream, Dict, None]

# numpy integers do not inherit from python integers, i.e.
# if you type a function argument as an `int` and then pass
# a value from a numpy array like `np.ones(10, dtype=np.int64)[0]`
# you may have a type error.
# these wrappers union numpy integers and python integers
Integer = Union[int, integer, unsignedinteger]

# Numbers which can only be floats and will not accept integers
# > isinstance(np.ones(1, dtype=np.float32)[0], floating) # True
# > isinstance(np.ones(1, dtype=np.float32)[0], float) # False
Floating = Union[float, floating]

# Many arguments take "any valid number" and don't care if it
# is an integer or a floating point input.
Number = Union[Floating, Integer]

__all__ = [
    "IO",
    "Any",
    "ArrayLike",
    "BinaryIO",
    "Dict",
    "Integer",
    "Iterable",
    "List",
    "Loadable",
    "NDArray",
    "Number",
    "Optional",
    "Sequence",
    "Stream",
    "Tuple",
    "float64",
    "int64",
    "Mapping",
    "Callable",
    "Hashable",
]
