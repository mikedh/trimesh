"""
boolean.py
-------------

Do boolean operations on meshes using either Blender or Manifold.
"""
import warnings

import numpy as np

try:
    from manifold3d import Manifold, Mesh
except BaseException as E:
    from .exceptions import ExceptionWrapper

    Mesh = ExceptionWrapper(E)
    Manifold = ExceptionWrapper(E)

from . import interfaces


def difference(meshes, engine=None, **kwargs):
    """
    Compute the boolean difference between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'

    Returns
    ----------
    difference : a - (other meshes), **kwargs for a Trimesh
    """
    result = _engines[engine](meshes, operation="difference", **kwargs)
    return result


def union(meshes, engine=None, **kwargs):
    """
    Compute the boolean union between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'

    Returns
    ----------
    union : a + (other meshes), **kwargs for a Trimesh
    """
    result = _engines[engine](meshes, operation="union", **kwargs)
    return result


def intersection(meshes, engine=None, **kwargs):
    """
    Compute the boolean intersection between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'

    Returns
    ----------
    intersection : **kwargs for a Trimesh object of the
                    volume that is contained by all meshes
    """
    result = _engines[engine](meshes, operation="intersection", **kwargs)
    return result


def boolean_manifold(meshes, operation, debug=False, **kwargs):
    """
    Run an operation on a set of meshes using the Manifold engine.
    """
    # Convert to manifold meshes
    manifolds = [
        Manifold.from_mesh(
            Mesh(
                vert_properties=np.asarray(mesh.vertices, dtype="float32"),
                tri_verts=np.asarray(mesh.faces, dtype="int32"),
            )
        )
        for mesh in meshes
    ]

    # Perform operations
    if operation == "difference":
        if len(meshes) != 2:
            raise ValueError("Difference only defined over two meshes.")

        result_manifold = manifolds[0] - manifolds[1]
    elif operation == "union":
        result_manifold = manifolds[0]

        for manifold in manifolds[1:]:
            result_manifold = result_manifold + manifold
    elif operation == "intersection":
        result_manifold = manifolds[0]

        for manifold in manifolds[1:]:
            result_manifold = result_manifold ^ manifold
    else:
        raise ValueError(f"Invalid boolean operation: '{operation}'")

    # Convert back to trimesh meshes
    from . import Trimesh

    result_mesh = result_manifold.to_mesh()
    out_mesh = Trimesh(vertices=result_mesh.vert_properties, faces=result_mesh.tri_verts)

    return out_mesh


def boolean_scad(*args, **kwargs):
    warnings.warn(
        "The OpenSCAD interface is deprecated, and Trimesh will instead"
        " use Manifold ('manifold'), which should be equivalent. In future versions"
        " of Trimesh, attempting to use engine 'scad' may raise an error.",
        DeprecationWarning,
        stacklevel=2,
    )
    return boolean_manifold(*args, **kwargs)


# which backend boolean engines
_engines = {
    None: boolean_manifold,
    "auto": boolean_manifold,
    "manifold": boolean_manifold,
    "scad": boolean_scad,
    "blender": interfaces.blender.boolean,
}
