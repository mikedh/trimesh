"""
boolean.py
-------------

Do boolean operations on meshes using either Blender or Manifold.
"""
import numpy as np

from . import interfaces
from .typed import Callable, NDArray, Optional, Sequence, Union


def difference(
    meshes: Sequence, engine: Optional[str] = None, check_volume: bool = True, **kwargs
):
    """
    Compute the boolean difference between a mesh an n other meshes.

    Parameters
    ----------
    meshes : sequence of trimesh.Trimesh
      Meshes to be processed.
    engine
      Which backend to use, i.e. 'blender' or 'manifold'
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.

    Returns
    ----------
    difference
      A `Trimesh` that contains `meshes[0] - meshes[1:]`
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")

    return _engines[engine](meshes, operation="difference", **kwargs)


def union(
    meshes: Sequence, engine: Optional[str] = None, check_volume: bool = True, **kwargs
):
    """
    Compute the boolean union between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.

    Returns
    ----------
    union
      A `Trimesh` that contains the union of all passed meshes.
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")

    return _engines[engine](meshes, operation="union", **kwargs)


def intersection(
    meshes: Sequence, engine: Optional[str] = None, check_volume: bool = True, **kwargs
):
    """
    Compute the boolean intersection between a mesh and other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.

    Returns
    ----------
    intersection
      A `Trimesh` that contains the intersection geometry.
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")
    return _engines[engine](meshes, operation="intersection", **kwargs)


def reduce_cascade(operation: Callable, items: Union[Sequence, NDArray]):
    """
    Call an operation function in a cascaded pairwise way against a
    flat list of items.

    This should produce the same result as `functools.reduce`
    if `operation` is commutable like addition or multiplication.
    This may be faster for an `operation` that runs with a speed
    proportional to its largest input, which mesh booleans appear to.

    The union of a large number of small meshes appears to be
    "much faster" using this method.

    This only differs from `functools.reduce` for commutative `operation`
    in that it returns `None` on empty inputs rather than `functools.reduce`
    which raises a `TypeError`.

    For example on `a b c d e f g` this function would run and return:
        a b
        c d
        e f
        ab cd
        ef g
        abcd efg
     -> abcdefg

    Where `functools.reduce` would run and return:
        a b
        ab c
        abc d
        abcd e
        abcde f
        abcdef g
     -> abcdefg

    Parameters
    ----------
    operation
      The function to call on pairs of items.
    items
      The flat list of items to apply operation against.
    """
    if len(items) == 0:
        return None
    elif len(items) == 1:
        # skip the loop overhead for a single item
        return items[0]
    elif len(items) == 2:
        # skip the loop overhead for a single pair
        return operation(items[0], items[1])

    for _ in range(int(1 + np.log2(len(items)))):
        results = []
        for i in np.arange(len(items) // 2) * 2:
            results.append(operation(items[i], items[i + 1]))

        if len(items) % 2:
            results.append(items[-1])

        items = results

    # logic should have reduced to a single item
    assert len(results) == 1

    return results[0]


# supported backend boolean engines
# preferred engines are at the top
_backends = [
    interfaces.manifold,
    interfaces.blender,
]

# test which engines are actually usable
_engines = {}
available_engines = ()
all_engines = ()
for e in _backends:
    all_engines = all_engines + e.name
    if e.exists:
        _engines[e.name] = e.boolean
        available_engines = available_engines + e.name
        if None not in _engines:
            # pick the first available engine as the default
            _engines[None] = e.boolean

def set_default_engine(engine):
    if engine not in available_engines:
        raise ValueError(f"Engine {engine} is not available on this system")
    _engines[None] = available_engines[engine].boolean
