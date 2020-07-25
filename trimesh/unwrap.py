"""
unwrap.py
------------

Unwrap meshes with no texture using Blender's "smart unwrap."
"""
from .interfaces import blender


def unwrap(mesh, image=None, **kwargs):
    """
    Returns a Trimesh object equivalent to the current mesh where
    the vertices have been assigned uv texture coordinates.

    The vertices may be split into as many as necessary
    by the unwrapping algorithm, depending on how many uv maps
    they appear in.

    Requires blender.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Original mesh to unwrap
    image : None or PIL.Image
      Image to apply after unwrapping

    Returns
    --------
    unwrapped : trimesh.Trimesh
      Mesh with unwrapped uv coordinates
    """
    result = blender.unwrap(mesh=mesh, **kwargs)

    if image is not None:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        result.visual.material.image = image

    return result
