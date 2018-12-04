import numpy as np

from .color import to_rgba


def uv_to_color(uv, image):
    """
    Get the color in a texture image.

    Parameters
    -------------
    uv : (n, 2) float
      UV coordinates on texture image
    image : PIL.Image
      Texture image

    Returns
    ----------
    colors : (n, 4) float
      RGBA color at each of the UV coordinates
    """
    x = (uv[:, 0] * image.width).round().astype(int)
    y = ((1 - uv[:, 1]) * image.height).round().astype(int)

    image = np.asarray(image)
    colors = image[y, x]
    colors = to_rgba(colors)
    return colors
