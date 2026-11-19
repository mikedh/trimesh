import numpy as np


def convex_decomposition(mesh, **kwargs) -> list[dict]:
    """
    Compute an approximate convex decomposition of a mesh.

    VHACD Parameters which can be passed as kwargs:

    Name                              Default
    -----------------------------------------
    maxConvexHulls                    64
    resolution                        400000
    minimumVolumePercentErrorAllowed  1.0
    maxRecursionDepth                 10
    shrinkWrap                        True
    fillMode                          "flood"
    maxNumVerticesPerCH               64
    asyncACD                          True
    minEdgeLength                     2
    findBestPlane                     False

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to be decomposed into convex parts
    **kwargs
      Passed to `trimkit.decompose` if installed, i.e. `maximum_hulls`,
      otherwise the VHACD keyword arguments above for `vhacdx`.

    Returns
    -------
    mesh_args : list
      List of **kwargs for Trimeshes that are nearly
      convex and approximate the original.
    """
    try:
        from trimkit import decompose

        # the VHACD names documented above as trimkit names them
        names = {
            "maxConvexHulls": "maximum_hulls",
            "maxNumVerticesPerCH": "maximum_vertices",
            "minimumVolumePercentErrorAllowed": "maximum_error",
        }
        hulls = decompose(
            mesh.vertices.view(np.ndarray),
            mesh.faces.view(np.ndarray),
            **{names.get(key, key): value for key, value in kwargs.items()},
        )
        return [
            {"vertices": hull_vertices, "faces": hull_faces}
            for hull_vertices, hull_faces in hulls
        ]
    except ImportError:
        from vhacdx import compute_vhacd

    # the faces are triangulated in a (len(face), ...vertex-index)
    # for vtkPolyData
    # i.e. so if shaped to four columns the first column is all 3
    faces = (
        np.column_stack((np.ones(len(mesh.faces), dtype=np.int64) * 3, mesh.faces))
        .ravel()
        .astype(np.uint32)
    )

    return [
        {"vertices": hull_vertices, "faces": hull_faces}
        for hull_vertices, hull_faces in compute_vhacd(mesh.vertices, faces, **kwargs)
    ]
