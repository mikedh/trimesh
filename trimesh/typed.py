import io
import typing
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from io import IOBase
from pathlib import Path
from sys import version_info
from typing import (
    IO,
    Any,
    BinaryIO,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

import numpy
from numpy import dtype, float64, floating, generic, int64, integer, ndarray
from numpy.random import BitGenerator, Generator, SeedSequence
from numpy.typing import ArrayLike, DTypeLike, NDArray

if version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

# most loader routes take `file_obj` which can either be
# a file-like object or a file path, or sometimes a dict
# `IOBase` is the base of every stdlib stream and is included because
# concrete streams like `io.BytesIO` don't satisfy the `IO` protocol
# under beartype — https://github.com/beartype/beartype/issues/643
Stream: TypeAlias = IO[str] | IO[bytes] | IOBase
Loadable: TypeAlias = str | Path | Stream | dict | None

# for a function that returns "is this a file or not"
# but with typeguard-narrowing if the answer is yes
BoolIsFile: TypeAlias = TypeGuard[IO[Any]]

# numpy integers do not inherit from python integers, i.e.
# if you type a function argument as an `int` and then pass
# a value from a numpy array like `np.ones(10, dtype=np.int64)[0]`
# you may have a type error.
# these wrappers union numpy integers and python integers
Integer: TypeAlias = int | integer

# Numbers which can only be floats and will not accept integers
# > isinstance(np.ones(1, dtype=np.float32)[0], floating) # True
# > isinstance(np.ones(1, dtype=np.float32)[0], float) # False
Floating: TypeAlias = float | floating

# Many arguments take "any valid number" and don't care if it
# is an integer or a floating point input.
Number: TypeAlias = Floating | Integer

# the literals for specifying what viewer to use
ViewerType: TypeAlias = Callable | Literal["gl", "jupyter", "marimo"] | None

# literal for color maps we include in the library
ColorMapType: TypeAlias = Literal["viridis", "magma", "inferno", "plasma"]

# the literal for what graph backend engines are available
GraphEngineType: TypeAlias = Literal["networkx", "scipy"] | None

# what 3D boolean engines are available
BooleanEngineType: TypeAlias = Literal["manifold", "blender"] | None
# what 3D boolean operations can be passed to boolean functions
BooleanOperationType: TypeAlias = Literal["difference", "union", "intersection"]

# what are the supported methods for converting a mesh into voxels.
VoxelizationMethodsType: TypeAlias = Literal["subdivide", "ray", "binvox"]


@runtime_checkable
class HttpSessionLike(Protocol):
    """
    Structural type for an HTTP session.

    Matches `httpx.Client` and `requests.Session` so a resolver
    can take either without trimesh importing them directly.
    other duck-typed sessions are called as `get(url)` with no
    additional kwargs. async sessions like `aiohttp.ClientSession`
    can't be driven synchronously and are rejected at runtime.
    """

    def get(self, url: str, *args, **kwargs) -> Any: ...


# add numpy types like their `numpy.typing.NDArray`
# but with specific dimensionality, i.e. `NDArray2D[np.float64]`
DType = TypeVar("DType", bound=generic)
NDArray1D: TypeAlias = ndarray[tuple[int], dtype[DType]]
NDArray2D: TypeAlias = ndarray[tuple[int, int], dtype[DType]]
NDArray3D: TypeAlias = ndarray[tuple[int, int, int], dtype[DType]]

# anything `numpy.random.default_rng` can normalize into a `Generator`
# passing a `Generator` lets a caller thread one stream through nested
# calls -- `default_rng` hands it back rather than re-seeding it
Seed: TypeAlias = Integer | Sequence[int] | SeedSequence | Generator | BitGenerator | None


# DEPRECATED : these aliases will be removed after July 2028
# import them from `typing`, `io`, or `numpy` instead
List = list
Dict = dict
Tuple = tuple
Set = set
Optional = typing.Optional
Union = typing.Union
TextIO = typing.TextIO
BytesIO = io.BytesIO
StringIO = io.StringIO
BufferedRandom = io.BufferedRandom
unsignedinteger = numpy.unsignedinteger


__all__ = [
    "IO",
    "Any",
    "ArrayLike",
    "BinaryIO",
    "BoolIsFile",
    "Callable",
    "DTypeLike",
    "Floating",
    "Hashable",
    "HttpSessionLike",
    "Integer",
    "Iterable",
    "Literal",
    "Loadable",
    "Mapping",
    "NDArray",
    "NDArray1D",
    "NDArray2D",
    "NDArray3D",
    "Number",
    "Seed",
    "Self",
    "Sequence",
    "Stream",
    "ViewerType",
    "float64",
    "int64",
]
