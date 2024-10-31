import numpy as np

from .. import exceptions
from ..typed import Sequence
from ..util import reduce_cascade

exists = False

try:
    from manifold3d import Manifold, Mesh
    exists = True
except BaseException as E:
    Mesh = exceptions.ExceptionWrapper(E)
    Manifold = exceptions.ExceptionWrapper(E)


def boolean(
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
    from .. import Trimesh

    result_mesh = result_manifold.to_mesh()
    return Trimesh(vertices=result_mesh.vert_properties, faces=result_mesh.tri_verts)
