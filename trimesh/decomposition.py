from typing import Dict, List

import numpy as np


def convex_decomposition(mesh) -> List[Dict]:
    """
    Compute an approximate convex decomposition of a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to be decomposed into convex parts

    Returns
    -------
    mesh_args : list
      List of **kwargs for Trimeshes that are nearly
      convex and approximate the original.
    """
    from pyVHACD import compute_vhacd

    # the faces are triangulated in a (len(face), ...vertex-index)
    # for vtkPolyData
    # i.e. so if shaped to four columns the first column is all 3
    faces = (
        np.column_stack((np.ones(len(mesh.faces), dtype=np.int64) * 3, mesh.faces))
        .ravel()
        .astype(np.uint32)
    )

    return [
        {"vertices": v, "faces": f.reshape((-1, 4))[:, 1:]}
        for v, f in compute_vhacd(mesh.vertices, faces)
    ]
