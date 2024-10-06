"""
boolean.py
-------------

Do boolean operations on meshes using either Blender or Manifold.
"""

import numpy as np

from . import exceptions, interfaces
from .typed import Callable, NDArray, Optional, Sequence, Union

try:
    from manifold3d import Manifold, Mesh
except BaseException as E:
    Mesh = exceptions.ExceptionWrapper(E)
    Manifold = exceptions.ExceptionWrapper(E)


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


def boolean_manifold(
    meshes: Sequence,
    operation: str,
    check_volume: bool = True,
    **kwargs,
):
    """
    Run an operation on a set of meshes using the Manifold engine.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    operation
      Which boolean operation to do.
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")

    # Convert to manifold meshes
    manifolds = [
        Manifold(
            mesh=Mesh(
                vert_properties=np.array(mesh.vertices, dtype=np.float32),
                tri_verts=np.array(mesh.faces, dtype=np.uint32),
            )
        )
        for mesh in meshes
    ]

    # Perform operations
    if operation == "difference":
        if len(meshes) < 2:
            raise ValueError("Difference only defined over two meshes.")
        elif len(meshes) == 2:
            # apply the single difference
            result_manifold = manifolds[0] - manifolds[1]
        elif len(meshes) > 2:
            # union all the meshes to be subtracted from the final result
            unioned = reduce_cascade(lambda a, b: a + b, manifolds[1:])
            # apply the difference
            result_manifold = manifolds[0] - unioned
    elif operation == "union":
        result_manifold = reduce_cascade(lambda a, b: a + b, manifolds)
    elif operation == "intersection":
        result_manifold = reduce_cascade(lambda a, b: a ^ b, manifolds)
    else:
        raise ValueError(f"Invalid boolean operation: '{operation}'")

    # Convert back to trimesh meshes
    from . import Trimesh

    result_mesh = result_manifold.to_mesh()
    return Trimesh(vertices=result_mesh.vert_properties, faces=result_mesh.tri_verts)


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


# which backend boolean engines
_engines = {
    None: boolean_manifold,
    "manifold": boolean_manifold,
    "blender": interfaces.blender.boolean,
}
