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
    uv = np.asanyarray(uv, dtype=np.float64)

    # find pixel positions from UV coordinates
    x = (uv[:, 0] * (image.width - 1)).round().astype(int)
    y = ((1 - uv[:, 1]) * (image.height - 1)).round().astype(int)

    # wrap to image size in the manner of GL_REPEAT
    x %= image.width
    y %= image.height

    # access colors from pixel locations
    image = np.asarray(image)
    colors = image[y, x]

    # handle gray scale
    if colors.ndim == 1:
        colors = np.repeat(colors[:, None], 3, axis=1)
    # now ndim == 2
    if colors.shape[1] == 3:
        colors = to_rgba(colors)  # rgb -> rgba
    return colors
