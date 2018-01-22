from . import interfaces


def convex_decomposition(mesh, engine=None, **kwargs):
    """
    Compute an approximate convex decomposition of a mesh.

    Parameters
    ----------
    mesh:   Trimesh object
    engine: string, which backend to use. Valid choice is 'vhacd'.

    Returns
    -------
    mesh_args: list, list of **kwargs for Trimeshes that are nearly
                     convex and approximate the original.
    """
    result = _engines[engine](mesh, **kwargs)
    return result


def decomposition_automatic(mesh, **kwargs):
    if interfaces.vhacd.exists:
        result = interfaces.vhacd.convex_decomposition(mesh, **kwargs)
    else:
        raise ValueError('No backends available for convex decomposition!')
    return result


_engines = {None: decomposition_automatic,
            'auto': decomposition_automatic,
            'vhacd': interfaces.vhacd.convex_decomposition}
