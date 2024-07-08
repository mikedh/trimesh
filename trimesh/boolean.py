"""
boolean.py
-------------

Do boolean operations on meshes using either Blender or Manifold.
"""

import numpy as np

from . import exceptions, interfaces
from .typed import Iterable, Optional

try:
    from manifold3d import Manifold, Mesh
except BaseException as E:
    Mesh = exceptions.ExceptionWrapper(E)
    Manifold = exceptions.ExceptionWrapper(E)


def difference(
    meshes: Iterable, engine: Optional[str] = None, check_volume: bool = True, **kwargs
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
    meshes: Iterable, engine: Optional[str] = None, check_volume: bool = True, **kwargs
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

    result = _engines[engine](meshes, operation="union", **kwargs)
    return result


def intersection(
    meshes: Iterable, engine: Optional[str] = None, check_volume: bool = True, **kwargs
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
    meshes: Iterable,
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
        if len(meshes) != 2:
            raise ValueError("Difference only defined over two meshes.")

        result_manifold = manifolds[0] - manifolds[1]
    elif operation in ["union", "intersection"]:
        for _lvl in range(int(1 + np.log2(len(manifolds)))):
            results = []
            for i in np.arange(len(manifolds) // 2) * 2:
                if operation == "union":
                    results.append(manifolds[i] + manifolds[i + 1])
                else: # operation == "intersection":
                    results.append(manifolds[i] ^ manifolds[i + 1])
            if len(manifolds) % 2:
                results.append(manifolds[-1])
            manifolds = results
        result_manifold = manifolds[0]
    else:
        raise ValueError(f"Invalid boolean operation: '{operation}'")

    # Convert back to trimesh meshes
    from . import Trimesh

    result_mesh = result_manifold.to_mesh()
    out_mesh = Trimesh(vertices=result_mesh.vert_properties, faces=result_mesh.tri_verts)

    return out_mesh


# which backend boolean engines
_engines = {
    None: boolean_manifold,
    "manifold": boolean_manifold,
    "blender": interfaces.blender.boolean,
}
