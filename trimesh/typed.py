from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from sys import version_info
from typing import IO, Any, BinaryIO, Literal, TypeAlias, TypeGuard, TypeVar

from numpy import dtype, float64, floating, generic, int64, integer, ndarray
from numpy.typing import ArrayLike, DTypeLike, NDArray

if version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

# most loader routes take `file_obj` which can either be
# a file-like object or a file path, or sometimes a dict
Stream: TypeAlias = IO[str] | IO[bytes]
Loadable: TypeAlias = str | Path | Stream | dict | None

# narrowing return for `is_file` — hides the `TypeGuard` spelling
# behind an alias so the signature stays readable but checkers
# still narrow the argument to a stream after the check passes
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

# add numpy types like their `numpy.typing.NDArray`
# but with specific dimensionality
DType = TypeVar("DType", bound=generic)
NDArray1D: TypeAlias = ndarray[tuple[int], dtype[DType]]
NDArray2D: TypeAlias = ndarray[tuple[int, int], dtype[DType]]
NDArray3D: TypeAlias = ndarray[tuple[int, int, int], dtype[DType]]

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
    "Integer",
    "Iterable",
    "Loadable",
    "Mapping",
    "NDArray",
    "NDArray1D",
    "NDArray2D",
    "NDArray3D",
    "Number",
    "Self",
    "Sequence",
    "Stream",
    "ViewerType",
    "float64",
    "int64",
]
