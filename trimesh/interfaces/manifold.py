import numpy as np

# Check if manifold is installed
try:
    import manifold3d
    exists=True

    from manifold3d import Mesh, Manifold
except ImportError:
    exists=False

def boolean(meshes, operation="difference", debug=False, **kwargs):
    """
    Run an operation on a set of meshes
    """
    # Convert to manifold meshes
    manifolds = [Manifold.from_mesh(
        Mesh(vert_properties=np.asarray(mesh.vertices, dtype="float32"), 
             tri_verts=np.asarray(mesh.faces, dtype="int32"))) 
        for mesh in meshes]
    
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

    from .. import Trimesh
    result_mesh = result_manifold.to_mesh()
    out_mesh = Trimesh(vertices=result_mesh.vert_properties, faces=result_mesh.tri_verts)

    return out_mesh
