import numpy as np

from . import util


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

    With `trimkit` installed `maxConvexHulls`, `resolution`,
    `minimumVolumePercentErrorAllowed` and `maxNumVerticesPerCH`
    are passed to `trimkit.decompose` and the others to `vhacdx`.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to be decomposed into convex parts
    **kwargs : VHACD keyword arguments

    Returns
    -------
    mesh_args : list
      List of **kwargs for Trimeshes that are nearly
      convex and approximate the original.
    """
    # the VHACD names documented above as trimkit names them
    names = {
        "maxConvexHulls": "maximum_hulls",
        "resolution": "resolution",
        "minimumVolumePercentErrorAllowed": "maximum_error",
        "maxNumVerticesPerCH": "maximum_vertices",
    }
    if util.has_module("trimkit") and names.keys() >= kwargs.keys():
        from trimkit import decompose

        hulls = decompose(
            mesh.vertices.view(np.ndarray),
            mesh.faces.view(np.ndarray),
            **{names[key]: value for key, value in kwargs.items()},
        )
        return [{"vertices": v, "faces": f} for v, f in hulls]

    # vhacdx fallback: delete with its deprecation
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
        {"vertices": v, "faces": f}
        for v, f in compute_vhacd(mesh.vertices, faces, **kwargs)
    ]
